


import logging
import os
import shutil
import stat
import tempfile
import uuid
from pathlib import Path
from urllib.parse import quote

from . import _xet
from .hf import repo_folder_name
from ._cache import (
    CACHE_DIR,
    create_cache_tag,
    create_pointer,
    pointer_path,
    supports_symlink,
)
from ._http import http_download, request_follow_redirect
from ._lock import file_lock
from ._types import HFFileMeta
from ._shared import link_from_shared, publish_to_shared

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://huggingface.co"


def hf_hub_url(
    repo_id: str,
    filename: str,
    *,
    repo_type: str = "model",
    revision: str = "main",
    endpoint: str = "https://huggingface.co"
) -> str:
    return (
        f"{endpoint.rstrip('/')}/"
        f"{repo_id}/resolve/"
        f"{quote(revision, safe='')}/"
        f"{quote(filename)}"
    )


def get_hf_file_metadata(
    url: str,
) -> HFFileMeta:
    res = request_follow_redirect(
        "HEAD",
        url,
        headers={"Accept-Encoding": "identity"},
    )
    try:
        res.raise_for_status()
        headers = res.headers
        size_header = headers.get("X-Linked-Size") or headers.get("Content-Length")
        size = (
            int(size_header)
            if size_header is not None
            else None
        )

        etag = (headers.get("X-Linked-Etag") or headers.get("ETag"))
        if etag is not None:
            etag = etag.lstrip("W/").strip('"')

        commit_hash = headers.get("X-Repo-Commit")

        if not commit_hash:
            raise RuntimeError("Hugging Face did not return X-Repo-Commit.")

        if not etag:
            raise RuntimeError("Hugging Face did not return an ETag.")

        location = headers.get("Location") or res.url

        xet = _xet.parse_file_data(res)

        return HFFileMeta(
            commit_hash=commit_hash,
            etag=etag,
            location=location,
            size=size,
            xet=xet,
        )

    finally:
        res.close()


def _download_file(
    *,
    destination: Path,
    url: str,
    filename: str,
    metadata: HFFileMeta,
    headers: dict[str, str] | None,
) -> None:

    if destination.exists():
        return

    destination.parent.mkdir(parents=True,exist_ok=True,)

    incomplete_path = destination.with_suffix(".incomplete")
    tmp_path = incomplete_path.with_name(
        f"{incomplete_path.stem}."
        f"{uuid.uuid4().hex[:8]}."
        ".incomplete"
    )

    headers = headers or {}
    
    try:
        with tmp_path.open("wb") as f:
            if (metadata.xet is not None and _xet.available()):
                _xet.download(
                    metadata.xet,
                    tmp_path,
                    headers=headers,
                    expected_size=metadata.size,
                    displayed_filename=filename,
                )

            else:
                http_download(
                    url,
                    f,
                    expected_size=metadata.size,
                    headers=headers,
                )
        _chmod_and_move(tmp_path, destination)
    finally:
        tmp_path.unlink(missing_ok=True)


def _chmod_and_move(src: Path, dst: Path) -> None:
    """Set correct permission before moving a blob from tmp directory to cache dir.

    Do not take into account the `umask` from the process as there is no convenient way
    to get it that is thread-safe.

    See:
    - About umask: https://docs.python.org/3/library/os.html#os.umask
    - Thread-safety: https://stackoverflow.com/a/70343066
    - About solution: https://github.com/huggingface/huggingface_hub/pull/1220#issuecomment-1326211591
    - Fix issue: https://github.com/huggingface/huggingface_hub/issues/1141
    - Fix issue: https://github.com/huggingface/huggingface_hub/issues/1215
    """
    # Get umask by creating a temporary file in the folder containing the incomplete file.
    # We know this folder is writable since the incomplete file has just been written there.
    # Probing next to `dst` is not always possible: when downloading to a local dir, `dst` is the
    # final file location and its parents might not be writable (e.g. read-only root filesystem).
    # See https://github.com/huggingface/huggingface_hub/issues/4304.
    tmp_file = src.parent / f"tmp_{uuid.uuid4()}"
    try:
        tmp_file.touch()
        cache_dir_mode = Path(tmp_file).stat().st_mode
        os.chmod(str(src), stat.S_IMODE(cache_dir_mode))
    except OSError as e:
        logger.warning(
            f"Could not set the permissions on the file '{src}'. Error: {e}.\nContinuing without setting permissions."
        )
    finally:
        try:
            tmp_file.unlink()
        except OSError:
            # fails if `tmp_file.touch()` failed => do nothing
            # See https://github.com/huggingface/huggingface_hub/issues/2359
            pass

    if os.path.lexists(dst):
        # Replace the entry so a force download never writes through a shared symlink.
        _replace_no_matter_what(src, dst)
    else:
        shutil.move(str(src), str(dst), copy_function=_copy_no_matter_what)


def _replace_no_matter_what(src: Path, dst: Path) -> None:
    """Replace `dst` with `src`.

    Some mounts reject replace-over-existing: stage the new file next to `dst`, move the old entry aside
    and restore it if the final move fails.
    """
    try:
        os.replace(src, dst)
    except OSError:
        staged_dst = dst.with_name(f".{dst.name}.{uuid.uuid4().hex[:8]}.new")
        backup_dst = dst.with_name(f".{dst.name}.{uuid.uuid4().hex[:8]}.old")
        backup_holds_previous_entry = False
        try:
            shutil.move(str(src), str(staged_dst), copy_function=_copy_no_matter_what)
            os.rename(dst, backup_dst)
            backup_holds_previous_entry = True
            try:
                shutil.move(str(staged_dst), str(dst), copy_function=_copy_no_matter_what)
            except OSError as move_error:
                try:
                    if os.path.lexists(dst):
                        os.unlink(dst)
                    os.rename(backup_dst, dst)
                    backup_holds_previous_entry = False
                except OSError as restore_error:
                    raise OSError(
                        f"Could not restore previous destination '{dst}' from '{backup_dst}'"
                    ) from restore_error
                raise move_error
            try:
                backup_dst.unlink()
                backup_holds_previous_entry = False
            except OSError as cleanup_error:
                logger.warning(f"Could not remove previous destination backup '{backup_dst}': {cleanup_error}")
        finally:
            staged_dst.unlink(missing_ok=True)
            if not backup_holds_previous_entry:
                backup_dst.unlink(missing_ok=True)


def _copy_no_matter_what(src: str, dst: str) -> None:
    """Copy file from src to dst.

    If `shutil.copy2` fails, fallback to `shutil.copyfile`.
    """
    try:
        # Copy file with metadata and permission
        # Can fail e.g. if dst is an S3 mount
        shutil.copy2(src, dst)
    except OSError:
        # Copy only file content
        shutil.copyfile(src, dst)


def hf_hub_download(
    repo_id: str,
    filename: str,
    *,
    outdir: str | Path = CACHE_DIR,
    subfolder: str | None = None,
    revision: str = "main",
    endpoint: str = DEFAULT_ENDPOINT,
    headers: dict[str, str] | None = None,
    **kwargs,
) -> Path:
    """Main Entry hf download models."""
    outdir = Path(outdir).expanduser().resolve()

    if subfolder is not None:
        subfolder = subfolder.strip("/")
        if subfolder.strip():
            filename = f"{subfolder.strip('/')}/{filename.lstrip('/')}"

    if Path(filename).is_file():
        return Path(filename)
    
    if (outdir / filename).is_file():
        return outdir / filename

    storage_dir = outdir / repo_folder_name(repo_id)
    url = hf_hub_url(
        repo_id,
        filename,
        revision=revision,
        endpoint=endpoint,
    )

    metadata = get_hf_file_metadata(url)

    download_url = (
        url
        if metadata.xet is not None
        else metadata.location
    )
    if (metadata.xet is not None and metadata.size is None):
        raise RuntimeError("Xet file has no known size.")

    # cross-platform transcription of filename, to be used as a local file path.
    relative_filename = Path(*filename.split("/"))

    assert metadata.commit_hash
    assert metadata.etag
    

    blob = storage_dir / "blobs" / metadata.etag
    snapshot = pointer_path(
        storage_dir,
        os.fspath(relative_filename),
        metadata.commit_hash,   
    )

    if snapshot.is_file():
        return snapshot

    create_cache_tag(outdir)

    # Prevent parallel downloads of the same file with a lock.
    # etag could be duplicated across repos.
    # Note: the lock is best-effort to avoid downloading the same file twice. Cache correctness
    # does not depend on it: each download writes to a process-unique temporary file that is
    # atomically renamed into place (see `_download_to_tmp_and_move`).
    locks_dir = outdir / ".locks"
    lock_path = locks_dir / repo_folder_name(repo_id=repo_id) / f"{metadata.etag}.lock"

    shared_blob_hash = (
        metadata.xet.file_hash
        if metadata.xet is not None
        and supports_symlink(outdir)
        else None
    )
    with file_lock(lock_path):
        if snapshot.is_file():
            return snapshot

        blob_is_shared = False
        if not blob.is_file():
            if shared_blob_hash is not None and link_from_shared(
                blob_path=blob, xet_hash=shared_blob_hash, cache_dir=outdir, expected_size=metadata.size
            ):
                blob_is_shared = True
            else:
                _download_file(
                    destination=blob,
                    url=download_url,
                    filename=filename,
                    metadata=metadata,
                    headers=headers,
                )
                if shared_blob_hash is not None and metadata.xet is not None:
                    blob_is_shared = publish_to_shared(
                        blob_path=blob,
                        xet_hash=shared_blob_hash,
                        cache_dir=outdir,
                        expected_size=metadata.size,
                        replace_existing=False,
                    )

        if not snapshot.is_file():
            create_pointer(
                blob,
                snapshot,
                move_source=not blob_is_shared,
            )

    return snapshot
