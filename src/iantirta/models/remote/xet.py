

import logging
import os
import re
import secrets
import shutil
import stat
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, SoftFileLock

logger = logging.getLogger(__name__)

REPO_TYPE_DATASET = "dataset"
REPO_TYPE_SPACE = "space"
REPO_TYPE_MODEL = "model"
REPO_TYPE_KERNEL = "kernel"
REPO_TYPES_MAPPING = {
    "datasets": REPO_TYPE_DATASET,
    "spaces": REPO_TYPE_SPACE,
    "models": REPO_TYPE_MODEL,
    "kernels": REPO_TYPE_KERNEL,
}

_XET_HASH_REGEX = re.compile(r"[0-9a-f]{64}")
_REPO_DIR_REGEX = re.compile(rf"(?:{'|'.join(sorted(REPO_TYPES_MAPPING))})--.+")

SHARED_BLOBS_DIR_NAME = "blobs"
SHARED_BLOBS_MARKER_NAME = ".huggingface-shared-blobs"
SHARED_BLOBS_LAYOUT_VERSION = "1"
_LOCK_SUFFIX = ".lock"
_SOFT_LOCK_TIMEOUT = 10
_MANIFEST_SUFFIX = ".refs"
_MARKER_TMP_REGEX = re.compile(rf"{re.escape(SHARED_BLOBS_MARKER_NAME)}\.[0-9a-f]{{8}}\.tmp")


def shared_blobs_dir(cache_dir: str | Path) -> Path:
    """Return the path of the shared blob store inside a cache directory."""
    return Path(cache_dir) / SHARED_BLOBS_DIR_NAME


def shared_blob_path(cache_dir: str | Path, xet_hash: str) -> Path:
    """Return the store path for a given Xet file hash.

    Raises `ValueError` if `xet_hash` is not a valid Xet hash.
    """
    if _XET_HASH_REGEX.fullmatch(xet_hash) is None:
        raise ValueError(f"Invalid Xet file hash: '{xet_hash}'.")
    return shared_blobs_dir(cache_dir) / xet_hash[:2] / xet_hash


def _is_directory(path: Path) -> bool:
    # checks if path is a regular directory without following symlinks
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def _is_regular_file(path: Path) -> bool:
    # checks if path is a regular file without following symlinks
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def is_shared_blobs_dir(path: str | Path) -> bool:
    """Return whether `path` is an owned, supported shared blob store."""
    store_dir = Path(path)
    marker_path = store_dir / SHARED_BLOBS_MARKER_NAME
    if not _is_directory(store_dir) or not _is_regular_file(marker_path):
        return False
    try:
        return marker_path.read_text() == f"{SHARED_BLOBS_LAYOUT_VERSION}\n"
    except OSError:
        return False


def _is_usable_store_entry(store_path: Path, expected_size: int | None) -> bool:
    """Return whether `store_path` is a regular, readable payload of the expected size."""
    if expected_size is None:
        # Size unknown (HEAD without a content length): the entry cannot be validated.
        return False
    try:
        store_stat = store_path.lstat()
    except OSError:
        return False
    if not stat.S_ISREG(store_stat.st_mode):
        return False
    if store_stat.st_size != expected_size:
        logger.warning(
            f"Shared blob '{store_path}' has an unexpected size ({store_stat.st_size} instead of {expected_size}). "
            "Not using it."
        )
        return False
    if not os.access(store_path, os.R_OK):
        logger.warning(f"Shared blob '{store_path}' is not readable. Not using it.")
        return False
    return True


def _path_for_comparison(path: str | Path) -> Path:
    """Return an absolute path without the Windows extended-length prefix."""
    path_str = os.fspath(path)
    if path_str[:8].lower() == "\\\\?\\unc\\":
        path_str = f"\\\\{path_str[8:]}"
    elif path_str.startswith("\\\\?\\"):
        path_str = path_str[4:]
    return Path(os.path.abspath(path_str))


def _relative_blob_path(blob_path: str | Path, cache_dir: str | Path) -> str | None:
    blob_path = _path_for_comparison(blob_path)
    cache_dir = _path_for_comparison(cache_dir)
    try:
        relative_path = blob_path.relative_to(cache_dir)
    except ValueError:
        return None
    if (
        len(relative_path.parts) != 3
        or _REPO_DIR_REGEX.fullmatch(relative_path.parts[0]) is None
        or relative_path.parts[1] != "blobs"
        or not relative_path.parts[2]
    ):
        return None
    relative_str = relative_path.as_posix()
    return None if "\n" in relative_str or "\r" in relative_str else relative_str


def _lock_path_for_store_path(store_path: Path) -> Path:
    return store_path.with_name(f"{store_path.name}{_LOCK_SUFFIX}")


@contextmanager
def _shared_blob_lock(store_path: Path) -> Generator[None, None, None]:
    """Lock publication, reference creation, and GC for one content hash."""
    lock_path = _lock_path_for_store_path(store_path)
    # Create the lock file ourselves: `O_NOFOLLOW` refuses a planted symlink and 0o666 lets other users of a
    # shared cache take the same lock. `FileLock` alone would follow symlinks and apply the umask.
    flags = os.O_WRONLY | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lock_path, flags, 0o666)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f"Shared blob lock is not a regular file: '{lock_path}'.")
        if hasattr(os, "fchmod"):
            try:
                os.fchmod(fd, 0o666)
            except OSError:
                pass
    finally:
        os.close(fd)
    lock = FileLock(lock_path, mode=0o666)
    try:
        lock.acquire()
    except NotImplementedError:
        # SoftFileLock uses file existence as the lock, so it cannot reuse the flock file above.
        lock = SoftFileLock(f"{lock_path}.soft", mode=0o666)
        lock.acquire(timeout=_SOFT_LOCK_TIMEOUT)
    try:
        yield
    finally:
        try:
            lock.release()
        except OSError:
            pass


def _append_manifest_reference(manifest_path: Path, relative_blob_path: str) -> None:
    """Append and flush one reference before its symlink is made visible."""
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(manifest_path, flags, 0o666)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f"Shared blob manifest is not a regular file: '{manifest_path}'.")
        if hasattr(os, "fchmod"):
            try:
                os.fchmod(fd, 0o666)
            except OSError:
                pass
        line = f"{relative_blob_path}\n".encode()
        if os.write(fd, line) != len(line):
            raise OSError(f"Could not append a complete reference to '{manifest_path}'.")
        os.fsync(fd)
    finally:
        os.close(fd)


def _make_temporary_symlink(blob_path: Path, store_path: Path) -> Path:
    tmp_link = blob_path.with_name(f".{blob_path.name}.{secrets.token_hex(4)}.shared")
    relative_target = os.path.relpath(_path_for_comparison(store_path), start=_path_for_comparison(blob_path).parent)
    os.symlink(relative_target, tmp_link)
    return tmp_link


def _manifest_path_for_store_path(store_path: Path) -> Path:
    return store_path.with_name(f"{store_path.name}{_MANIFEST_SUFFIX}")


def try_link_from_shared_store(
    *, blob_path: str, xet_hash: str, cache_dir: str | Path, expected_size: int | None
) -> bool:
    """Materialize `blobs/<etag>` as a symlink to an existing store entry, if any.

    The reference manifest is flushed before the symlink becomes visible. Failures are
    best-effort misses and leave the regular download path untouched.
    """
    if (
        _XET_HASH_REGEX.fullmatch(xet_hash) is None
        or not is_shared_blobs_dir(shared_blobs_dir(cache_dir))
    ):
        return False
    store_path = shared_blob_path(cache_dir, xet_hash)
    relative_blob_path = _relative_blob_path(blob_path, cache_dir)
    if relative_blob_path is None:
        return False

    tmp_link: Path | None = None
    try:
        with _shared_blob_lock(store_path):
            if not _is_usable_store_entry(store_path, expected_size):
                return False

            tmp_link = _make_temporary_symlink(Path(blob_path), store_path)
            _append_manifest_reference(_manifest_path_for_store_path(store_path), relative_blob_path)
            os.replace(tmp_link, blob_path)
    except OSError as e:
        logger.debug(f"Could not symlink '{blob_path}' from shared blob store: {e}")
        return False
    finally:
        if tmp_link is not None:
            tmp_link.unlink(missing_ok=True)

    logger.debug(f"Blob '{blob_path}' reused from shared blob store (no download needed).")
    return True


def _cleanup_abandoned_marker_temps(store_dir: Path) -> bool:
    """Remove leftover marker temporaries.

    Returns whether the directory is empty afterwards, i.e. safe to mark. Any foreign content returns False.
    """
    expected_content = f"{SHARED_BLOBS_LAYOUT_VERSION}\n"
    entries = list(store_dir.iterdir())
    for entry in entries:
        if _MARKER_TMP_REGEX.fullmatch(entry.name) is None or not _is_regular_file(entry):
            return False
        try:
            if not expected_content.startswith(entry.read_text()):
                return False
        except (OSError, UnicodeError):
            return False
    for entry in entries:
        entry.unlink(missing_ok=True)
    return not any(store_dir.iterdir())


def _shared_blob_mode(cache_dir: str | Path) -> int:
    """Return a read-only mode accessible to users who can traverse the cache root."""
    if os.name == "nt":
        # Windows uses ACLs; 0444 would set the read-only attribute and block replacement and GC.
        return 0o666
    try:
        cache_mode = stat.S_IMODE(Path(cache_dir).stat().st_mode)
    except OSError:
        return 0o400
    return 0o400 | (0o040 if cache_mode & stat.S_IXGRP else 0) | (0o004 if cache_mode & stat.S_IXOTH else 0)


def _shared_directory_mode(cache_dir: str | Path) -> int:
    """Mirror cache-root access and inheritance bits on newly created store directories."""
    try:
        cache_mode = stat.S_IMODE(Path(cache_dir).stat().st_mode)
    except OSError:
        return 0o700
    return cache_mode & (0o777 | stat.S_ISGID | stat.S_ISVTX)


def _repair_shared_directory_mode(path: Path, cache_dir: str | Path) -> None:
    expected_mode = _shared_directory_mode(cache_dir)
    if stat.S_IMODE(path.lstat().st_mode) != expected_mode:
        path.chmod(expected_mode)


def _ensure_shared_blobs_dir(cache_dir: str | Path) -> bool:
    """Create and mark the shared store, refusing to adopt an unmarked directory."""
    store_dir = shared_blobs_dir(cache_dir)
    marker_path = store_dir / SHARED_BLOBS_MARKER_NAME
    try:
        store_dir.mkdir(exist_ok=True)
        if is_shared_blobs_dir(store_dir):
            return True
        if not _is_directory(store_dir) or not _cleanup_abandoned_marker_temps(store_dir):
            logger.debug(f"Refusing to use unmarked shared blob directory '{store_dir}'.")
            return False
        _repair_shared_directory_mode(store_dir, cache_dir)

        tmp_marker = marker_path.with_name(f"{marker_path.name}.{secrets.token_hex(4)}.tmp")
        try:
            tmp_marker.write_text(f"{SHARED_BLOBS_LAYOUT_VERSION}\n")
            tmp_marker.chmod(_shared_blob_mode(cache_dir))
            os.replace(tmp_marker, marker_path)
        finally:
            tmp_marker.unlink(missing_ok=True)
        return is_shared_blobs_dir(store_dir)
    except OSError as e:
        logger.debug(f"Could not initialize shared blob directory '{store_dir}': {e}")
        return False


def _ensure_prefix_dir(cache_dir: str | Path, xet_hash: str) -> Path | None:
    if not _ensure_shared_blobs_dir(cache_dir):
        return None
    prefix_dir = shared_blob_path(cache_dir, xet_hash).parent
    try:
        prefix_dir.mkdir(exist_ok=True)
    except OSError as e:
        logger.debug(f"Could not create shared blob prefix directory '{prefix_dir}': {e}")
        return None
    if not _is_directory(prefix_dir):
        logger.debug(f"Refusing to use non-directory shared blob prefix '{prefix_dir}'.")
        return None
    try:
        _repair_shared_directory_mode(prefix_dir, cache_dir)
    except OSError as e:
        logger.debug(f"Could not set shared blob prefix permissions on '{prefix_dir}': {e}")
        return None
    return prefix_dir


def _prepare_shared_blob_permissions(blob_path: Path, prefix_dir: Path, cache_dir: str | Path) -> None:
    """Make a payload immutable and readable according to the shared cache policy."""
    blob_mode = _shared_blob_mode(cache_dir)
    os.chmod(blob_path, blob_mode)
    if os.name == "nt" or not hasattr(os, "chown"):
        return
    target_gid = prefix_dir.stat().st_gid
    if blob_path.stat().st_gid == target_gid:
        return
    try:
        os.chown(blob_path, -1, target_gid)
    except OSError:
        # Without other-read, a wrong group makes the entry unreadable to other users: fall back to repo-local.
        if blob_mode & stat.S_IRGRP and not blob_mode & stat.S_IROTH:
            raise


def publish_blob_to_shared_store(
    *,
    blob_path: str,
    xet_hash: str,
    cache_dir: str | Path,
    expected_size: int | None,
    replace_existing: bool = False,
) -> bool:
    """Move a fresh Xet download into the store and replace its repo blob with a symlink.

    Best-effort: on failure the repo blob remains (or is restored as) a regular local
    file. Returns whether the repo blob was successfully shared.
    """
    # Publishing a payload whose size is unknown would create an entry no reader can validate.
    if expected_size is None or _XET_HASH_REGEX.fullmatch(xet_hash) is None:
        return False
    prefix_dir = _ensure_prefix_dir(cache_dir, xet_hash)
    relative_blob_path = _relative_blob_path(blob_path, cache_dir)
    if prefix_dir is None or relative_blob_path is None:
        return False

    blob_path_obj = Path(blob_path)
    store_path = shared_blob_path(cache_dir, xet_hash)
    manifest_path = _manifest_path_for_store_path(store_path)
    tmp_link: Path | None = None
    blob_moved = False
    try:
        with _shared_blob_lock(store_path):
            tmp_link = _make_temporary_symlink(blob_path_obj, store_path)
            store_is_usable = not replace_existing and _is_usable_store_entry(store_path, expected_size)
            _append_manifest_reference(manifest_path, relative_blob_path)
            if not store_is_usable:
                _prepare_shared_blob_permissions(blob_path_obj, prefix_dir, cache_dir)
                os.replace(blob_path_obj, store_path)
                blob_moved = True
            try:
                os.replace(tmp_link, blob_path_obj)
            except OSError:
                if blob_moved:
                    shutil.copyfile(store_path, blob_path_obj)  # restore under the lock, before GC can run
                    blob_moved = False
                raise
    except OSError as e:
        logger.debug(f"Could not publish '{blob_path}' to shared blob store: {e}")
        if blob_moved:
            raise OSError(f"Could not restore repo blob '{blob_path}' after shared-store failure") from e
        return False
    finally:
        if tmp_link is not None:
            tmp_link.unlink(missing_ok=True)

    logger.debug(f"Blob '{blob_path}' published to shared blob store.")
    return True
