
import re
from pathlib import Path
import os
import logging
import stat
import secrets
import shutil

from ._lock import file_lock

logger = logging.getLogger(__name__)


SHARED_DIR = "blobs"
SHARED_BLOBS_MARKER_NAME = ".huggingface-shared-blobs"
SHARED_BLOBS_LAYOUT_VERSION = "1"

_XET_HASH_RE = re.compile(r"[0-9a-f]{64}")
_MARKER_TMP_RE = re.compile(rf"{re.escape(
    SHARED_BLOBS_MARKER_NAME
)}\.[0-9a-f]{{8}}\.tmp")
_REPO_DIR_RE = re.compile(rf"(?:{'|'.join(
    ["datasets", "spaces", "models", "kernels"]
)})--.+")


def shared_blobs_dir(cache_dir: str | Path) -> Path:
    """Return the path of the shared blob store inside a cache directory."""
    return Path(cache_dir) / SHARED_DIR


def shared_blob_path(
    cache_dir: Path,
    xet_hash: str,
) -> Path:
    if not _XET_HASH_RE.fullmatch(xet_hash):
        raise ValueError(f"Invalid Xet hash: {xet_hash!r}")

    return cache_dir / SHARED_DIR / xet_hash[:2] / xet_hash


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


def _path_for_comparison(path: str | Path) -> Path:
    """Return an absolute path without the Windows extended-length prefix."""
    path_str = os.fspath(path)
    if path_str[:8].lower() == "\\\\?\\unc\\":
        path_str = f"\\\\{path_str[8:]}"
    elif path_str.startswith("\\\\?\\"):
        path_str = path_str[4:]
    return Path(os.path.abspath(path_str))


def _relative_blob_path(
    blob_path: str | Path,
    cache_dir: str | Path
) -> str | None:
    blob_path = _path_for_comparison(blob_path)
    cache_dir = _path_for_comparison(cache_dir)

    try:
        relative_path = blob_path.relative_to(cache_dir)
    except ValueError:
        return None

    if len(relative_path.parts) != 3:
        return None

    if relative_path.parts[1] != "blobs":
        return None

    if not relative_path.parts[2]:
        return None

    if not _REPO_DIR_RE.fullmatch(relative_path.parts[0]):
        return None

    relative_str = relative_path.as_posix()
    return (
        None
        if "\n" in relative_str
        or "\r" in relative_str
        else relative_str
    )


def _make_temporary_symlink(
    blob_path: Path,
    store_path: Path
) -> Path:
    tmp_link = blob_path.with_name(
        f".{blob_path.name}.{secrets.token_hex(4)}.shared"
    )
    relative_target = os.path.relpath(
        _path_for_comparison(store_path),
        start=_path_for_comparison(blob_path).parent
    )
    os.symlink(relative_target, tmp_link)
    return tmp_link


def _is_usable_store_entry(
    store_path: Path,
    expected_size: int | None
) -> bool:
    """Return whether `store_path` is a regular,
    readable payload of the expected size.
    """
    if expected_size is None:
        # Size unknown (HEAD without a content length):
        # the entry cannot be validated.
        return False

    try:
        store_stat = store_path.lstat()

    except OSError:
        return False

    if not stat.S_ISREG(store_stat.st_mode):
        return False

    if store_stat.st_size != expected_size:
        logger.warning(
            f"Shared blob '{store_path}' has an unexpected size "
            f"({store_stat.st_size} instead of {expected_size}). "
            "Not using it."
        )
        return False

    if not os.access(store_path, os.R_OK):
        logger.warning(
            f"Shared blob '{store_path}' is not readable. Not using it."
        )
        return False

    return True


def _get_lock_path(store_path: Path) -> Path:
    return store_path.with_name(f"{store_path.name}.lock")


def _get_manifest_path(store_path: Path) -> Path:
    return store_path.with_name(
        f"{store_path.name}.refs"
    )


def _update_manifest(
    manifest_path: Path,
    relative_blob_path: str
) -> None:
    """Append and flush one reference before its symlink is made visible."""
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    fd = os.open(manifest_path, flags, 0o666)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(
                "Shared blob manifest is not a "
                f"regular file: '{manifest_path}'."
            )

        if hasattr(os, "fchmod"):
            try:
                os.fchmod(fd, 0o666)
            except OSError:
                pass

        line = f"{relative_blob_path}\n".encode()
        if os.write(fd, line) != len(line):
            raise OSError(
                f"Could not append a complete reference to '{manifest_path}'."
            )

        os.fsync(fd)
    finally:
        os.close(fd)


def link_from_shared(
    *,
    blob_path: Path,
    xet_hash: str,
    cache_dir: Path,
    expected_size: int | None,
) -> bool:
    if not _XET_HASH_RE.fullmatch(xet_hash):
        return False

    shared = shared_blob_path(cache_dir, xet_hash)

    if not is_shared_blobs_dir(shared_blobs_dir(cache_dir)):
        return False

    relative_blob_path = _relative_blob_path(blob_path, cache_dir)
    if relative_blob_path is None:
        return False

    manifest_path = _get_manifest_path(shared)
    tmp_link: Path | None = None
    try:
        with file_lock(_get_lock_path(shared)):
            if not _is_usable_store_entry(shared, expected_size):
                return False

            tmp_link = _make_temporary_symlink(blob_path, shared)
            _update_manifest(manifest_path, relative_blob_path)
            os.replace(tmp_link, blob_path)
    except OSError as e:
        logger.debug(
            f"Could not symlink '{blob_path}' "
            f"from shared blob store: {e}"
        )
        return False
    finally:
        if tmp_link is not None:
            tmp_link.unlink(missing_ok=True)

    logger.debug(
        f"Blob '{blob_path}' reused from shared blob "
        "store (no download needed)."
    )
    return True


def _update_permissions(
    blob_path: Path,
    prefix_dir: Path,
    cache_dir: str | Path
) -> None:
    """Make a payload immutable and readable
    according to the shared cache policy.
    """
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
        # Without other-read, a wrong group
        # makes the entry unreadable to other users:
        # fall back to repo-local.
        if blob_mode & stat.S_IRGRP and not blob_mode & stat.S_IROTH:
            raise


def _shared_directory_mode(cache_dir: str | Path) -> int:
    """Mirror cache-root access and
    inheritance bits on newly created
    store directories.
    """
    try:
        cache_mode = stat.S_IMODE(Path(cache_dir).stat().st_mode)
    except OSError:
        return 0o700
    return cache_mode & (0o777 | stat.S_ISGID | stat.S_ISVTX)


def _repair_shared_directory_mode(path: Path, cache_dir: str | Path) -> None:
    expected_mode = _shared_directory_mode(cache_dir)
    if stat.S_IMODE(path.lstat().st_mode) != expected_mode:
        path.chmod(expected_mode)


def _cleanup_abandoned_marker_temps(store_dir: Path) -> bool:
    """Remove leftover marker temporaries.

    Returns whether the directory is empty
    afterwards, i.e. safe to mark. Any foreign
    content returns False.
    """
    expected_content = f"{SHARED_BLOBS_LAYOUT_VERSION}\n"
    entries = list(store_dir.iterdir())
    for entry in entries:
        if (
            _MARKER_TMP_RE.fullmatch(entry.name) is None
            or not _is_regular_file(entry)
        ):
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
    """Return a read-only mode accessible
    to users who can traverse the cache root.
    """
    if os.name == "nt":
        # Windows uses ACLs; 0444 would set the
        # read-only attribute and block replacement and GC.
        return 0o666
    try:
        cache_mode = stat.S_IMODE(Path(cache_dir).stat().st_mode)
    except OSError:
        return 0o400
    return (
        0o400
        | (
            0o040
            if cache_mode & stat.S_IXGRP
            else 0
        ) | (
            0o004
            if cache_mode & stat.S_IXOTH
            else 0
        )
    )


def _ensure_shared_blobs_dir(cache_dir: str | Path) -> bool:
    """Create and mark the shared store,
    refusing to adopt an unmarked directory.
    """
    store_dir = shared_blobs_dir(cache_dir)
    marker_path = store_dir / SHARED_BLOBS_MARKER_NAME
    try:
        store_dir.mkdir(exist_ok=True)
        if is_shared_blobs_dir(store_dir):
            return True
        if (
            not _is_directory(store_dir)
            or not _cleanup_abandoned_marker_temps(store_dir)
        ):
            logger.debug(
                "Refusing to use unmarked shared "
                f"blob directory '{store_dir}'."
            )
            return False
        _repair_shared_directory_mode(store_dir, cache_dir)

        tmp_marker = marker_path.with_name(
            f"{marker_path.name}.{secrets.token_hex(4)}.tmp"
        )
        try:
            tmp_marker.write_text(f"{SHARED_BLOBS_LAYOUT_VERSION}\n")
            tmp_marker.chmod(_shared_blob_mode(cache_dir))
            os.replace(tmp_marker, marker_path)
        finally:
            tmp_marker.unlink(missing_ok=True)
        return is_shared_blobs_dir(store_dir)
    except OSError as e:
        logger.debug(
            f"Could not initialize shared blob directory '{store_dir}': {e}"
        )
        return False


def _ensure_prefix_dir(cache_dir: str | Path, xet_hash: str) -> Path | None:
    if not _ensure_shared_blobs_dir(cache_dir):
        return None
    prefix_dir = shared_blob_path(cache_dir, xet_hash).parent
    try:
        prefix_dir.mkdir(exist_ok=True)
    except OSError as e:
        logger.debug(
            "Could not create shared blob "
            f"prefix directory '{prefix_dir}': {e}"
        )
        return None
    if not _is_directory(prefix_dir):
        logger.debug(
            "Refusing to use non-directory "
            f"shared blob prefix '{prefix_dir}'."
        )
        return None
    try:
        _repair_shared_directory_mode(prefix_dir, cache_dir)
    except OSError as e:
        logger.debug(
            "Could not set shared blob "
            f"prefix permissions on '{prefix_dir}': {e}"
        )
        return None
    return prefix_dir


def publish_to_shared(
    *,
    blob_path: Path,
    xet_hash: str,
    cache_dir: Path,
    expected_size: int | None,
    replace_existing: bool = False,
) -> bool:
    # Publishing a payload whose size is unknown
    # would create an entry no reader can validate.
    if expected_size is None or _XET_HASH_RE.fullmatch(xet_hash) is None:
        return False

    prefix_dir = _ensure_prefix_dir(cache_dir, xet_hash)
    relative_blob_path = _relative_blob_path(blob_path, cache_dir)

    if prefix_dir is None or relative_blob_path is None:
        return False

    shared = shared_blob_path(cache_dir, xet_hash)

    manifest_path = _get_manifest_path(shared)
    tmp_link: Path | None = None
    blob_moved = False
    try:
        with file_lock(_get_lock_path(shared)):
            tmp_link = _make_temporary_symlink(blob_path, shared)
            is_store_usable = (
                not replace_existing
                and _is_usable_store_entry(shared, expected_size)
            )
            _update_manifest(manifest_path, relative_blob_path)
            if not is_store_usable:
                _update_permissions(blob_path, prefix_dir, cache_dir)
                os.replace(blob_path, shared)
                blob_moved = True
            try:
                os.replace(tmp_link, blob_path)
            except OSError:
                if blob_moved:
                    # restore under the lock, before GC can run
                    shutil.copyfile(shared, blob_path)
                    blob_moved = False
                raise
    except OSError as e:
        logger.debug(
            f"Could not publish '{blob_path}' "
            f"to shared blob store: {e}"
        )
        if blob_moved:
            raise OSError(
                f"Could not restore repo blob '{blob_path}' "
                "after shared-store failure"
            ) from e
        return False
    finally:
        if tmp_link is not None:
            tmp_link.unlink(missing_ok=True)

    logger.debug(
        f"Blob '{blob_path}' published to shared blob store."
    )
    return True
