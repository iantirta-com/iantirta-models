import os
from pathlib import Path
from urllib.parse import quote
from dataclasses import dataclass
import errno
import shutil

CACHE_DIR = Path("~/.cache/iantirta").expanduser().resolve()
CACHEDIR_TAG_CONTENT = (
    "Signature: 8a477f597d28d172789f06886806bc55\n"
    "# This file is a cache directory tag created by huggingface_hub.\n"
    "# For information about cache directory tags, see:\n"
    "#\thttps://bford.info/cachedir/\n"
)
SYMLINK_CACHE_SUPPORTED: dict[str, bool] = {}


def _create_cachedir_tag(cache_dir: Path) -> None:
    """Create a CACHEDIR.TAG file in ``cache_dir`` if one does not already exist.

    The tag follows the `Cache Directory Tagging Standard <http://www.brynosaurus.com/cachedir/>`_
    so that backup tools can recognize and skip cache directories.
    """
    tag_path = cache_dir / "CACHEDIR.TAG"
    if not tag_path.exists():
        try:
            tag_path.write_text(CACHEDIR_TAG_CONTENT)
        except OSError:
            pass


def support_symlink(
    cache_dir: str | Path = CACHE_DIR,
) -> bool:
    cache_dir = str(Path(cache_dir).expanduser().resolve())
    
    if cache_dir not in SYMLINK_CACHE_SUPPORTED:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=cache_dir) as tmpdir:
                src_path = Path(tmpdir) / "dummy_file_src"
                src_path.touch()
                dst_path = Path(tmpdir) / "dummy_file_dst"

                # Relative source path as in `_create_symlink``
                relative_src = os.path.relpath(src_path, start=os.path.dirname(dst_path))
                try:
                    os.symlink(relative_src, dst_path)
                    SYMLINK_CACHE_SUPPORTED[cache_dir] = True
                except OSError:
                    SYMLINK_CACHE_SUPPORTED[cache_dir] = False
        except ValueError:
            # Raised if src and dst are not on the same volume. Symlinks will still work on Linux/Macos.
            # See https://docs.python.org/3/library/os.path.html#os.path.commonpath
            SYMLINK_CACHE_SUPPORTED[cache_dir] = os.name != "nt"

    return SYMLINK_CACHE_SUPPORTED[cache_dir]


def _create_symlink(src: str, dst: str, new_blob: bool = False) -> None:
    """Create a symbolic link named dst pointing to src.
    """
    try:
        os.remove(dst)
    except OSError:
        pass

    abs_src = os.path.abspath(os.path.expanduser(src))
    abs_dst = os.path.abspath(os.path.expanduser(dst))
    abs_dst_folder = os.path.dirname(abs_dst)

    # Use relative_dst in priority
    try:
        relative_src = os.path.relpath(abs_src, abs_dst_folder)
    except ValueError:
        # Raised on Windows if src and dst are not on the same volume. This is the case when creating a symlink to a
        # local_dir instead of within the cache directory.
        # See https://docs.python.org/3/library/os.path.html#os.path.relpath
        relative_src = None

    try:
        commonpath = os.path.commonpath([abs_src, abs_dst])
        _support_symlink = support_symlink(commonpath)
    except PermissionError:
        # Permission error means src and dst are not in the same volume (e.g. destination path has been provided
        # by the user via `local_dir`. Let's test symlink support there)
        _support_symlink = support_symlink(abs_dst_folder)
    except OSError as e:
        # OS error (errno=30) means that the commonpath is readonly on Linux/MacOS.
        if e.errno == errno.EROFS:
            _support_symlink = support_symlink(abs_dst_folder)
        else:
            raise

    if _support_symlink:
        src_rel_or_abs = relative_src or abs_src
        try:
            os.symlink(src_rel_or_abs, abs_dst)
            return
        except FileExistsError:
            if os.path.islink(abs_dst) and os.path.realpath(abs_dst) == os.path.realpath(abs_src):
                # `abs_dst` already exists and is a symlink to the `abs_src` blob. It is most likely that the file has
                # been cached twice concurrently (exactly between `os.remove` and `os.symlink`). Do nothing.
                return
            else:
                # Very unlikely to happen. Means a file `dst` has been created exactly between `os.remove` and
                # `os.symlink` and is not a symlink to the `abs_src` blob file. Raise exception.
                raise
        except PermissionError:
            # Permission error means src and dst are not in the same volume (e.g. download to local dir) and symlink
            # is supported on both volumes but not between them. Let's just make a hard copy in that case.
            pass

    # Symlinks are not supported => let's move or copy the file.
    if new_blob:
        shutil.move(abs_src, abs_dst, copy_function=_copy_no_matter_what)
    else:
        shutil.copyfile(abs_src, abs_dst)


def repo_folder_name(
    repo_id: str,
    repo_type: str = "model"
) -> str:
    """Return a serialized version of a hf.co
    repo name and type, safe for disk storage
    as a single non-nested folder.

    Example: models--julien-c--EsperBERTo-small
    """
    parts = [f"{repo_type}s", *repo_id.split("/")]
    return "--".join(parts)


def hf_hub_url(
    repo_id: str,
    filename: str,
    *,
    repo_type: str = "model",
    revision: str = "main",
    endpoint: str = "https://huggingface.co"
) -> str:
    url = "/{repo_id}/resolve/{revision}/{filename}".format(
        repo_id=repo_id,
        revision=quote(revision, safe=""),
        filename=quote(filename)
    )
    return endpoint.rstrip("/") + url


def _get_pointer_path(
    storage_dir: Path,
    relative_filename: str,
    revision: str = "main",
) -> Path:
    # Using `os.path.abspath` instead of `Path.resolve()` to avoid resolving symlinks
    snapshot_path = storage_dir / "snapshots"
    pointer_path = snapshot_path / revision / relative_filename
    if (
        Path(os.path.abspath(str(snapshot_path)))
        not in Path(os.path.abspath(str(pointer_path))).parents
    ):
        raise ValueError(
            "Invalid pointer path: cannot create pointer path in snapshot folder if"
            f" `storage_folder='{storage_dir}'`, `revision='{revision}'` and"
            f" `relative_filename='{relative_filename}'`."
        )
    return pointer_path


@dataclass(frozen=True)
class HFFileMeta:
    commit_hash: str | None
    etag: str | None
    location: str
    size: int | None
    xet_fd: XetFileData | None


def get_hf_file_metadata(
    url: str,
):
    
    try:
        import hf_xet
        from .xet import _parse_xet_fd
    except:
        _parse_xet_fd = lambda r: None

    res = http_request(
        "HEAD",
        url,
        headers={"Accept-Encoding": "identity"},
    )
    res.raise_for_status()

    size_str = res.headers.get("X-Linked-Size") or res.headers.get("Content-Length")
    size = int(size_str) if size_str is not None else None

    etag = (
        res.headers.get("X-Linked-Etag")
        or res.headers.get("ETag")
    )

    if etag is not None:
        etag = etag.lstrip("W/").strip('"')

    return HFFileMeta(
        commit_hash=res.headers.get("X-Repo-Commit"),
        location=(
            res.headers.get("Location")
            or str(res.request.url)
        ),
        size=size,
        etag=etag,
        xet_fd=_parse_xet_fd(res)
    )


def hf_hub_download(
    repo_id: str,
    filename: str,
    *,
    outdir: str | Path = CACHE_DIR,
    subfolder: str | None = None,
    **kwargs,
) -> Path:
    """Main Entry hf download models."""
    outdir = Path(outdir).expanduser().resolve()

    if subfolder is not None:
        filename = f"{subfolder}/{filename}"

    if (outdir / filename).is_file():
        return outdir / filename

    storage_dir = outdir / repo_folder_name(repo_id)
    url = hf_hub_url(repo_id, filename)
    url_to_download = url
    
    try:
        metadata = get_hf_file_metadata(url)
    except:
        raise

    assert metadata.commit_hash
    assert metadata.etag
    if metadata.size is None and metadata.xet_fd is not None:
        raise RuntimeError()
    if metadata.xet_fd is None and url != metadata.location:
        url_to_download = metadata.location

    # cross-platform transcription of filename, to be used as a local file path.
    relative_filename = os.path.join(*filename.split("/"))

    blob_path = storage_dir / "blobs" / metadata.etag
    pointer_path = _get_pointer_path(
        storage_dir,
        relative_filename,
        metadata.commit_hash,   
    )

    if pointer_path.is_file():
        return pointer_path

    blob_path.mkdir(parents=True, exist_ok=True)
    pointer_path.mkdir(parents=True, exist_ok=True)

    _create_cachedir_tag(outdir)

    # Prevent parallel downloads of the same file with a lock.
    # etag could be duplicated across repos.
    # Note: the lock is best-effort to avoid downloading the same file twice. Cache correctness
    # does not depend on it: each download writes to a process-unique temporary file that is
    # atomically renamed into place (see `_download_to_tmp_and_move`).
    locks_dir = outdir / ".locks"
    lock_path = locks_dir / repo_folder_name(repo_id=repo_id) / f"{metadata.etag}.lock"

    if blob_path.is_file():
        with WeakFileLock(lock_path):
            if not pointer_path.is_file():
                _create_symlink(str(blob_path), str(pointer_path), new_blob=False)
            return pointer_path

    # Xet hash of the blob in the shared store, None when the store must not be used for this download
    # (see `utils/_shared_blobs.py`).
    shared_blob_hash = (
        metadata.xet_fd.file_hash
        if metadata.xet_fd is not None
        and are_symlinks_supported(outdir)
        else None
    )
    with WeakFileLock(lock_path):
        if (
            shared_blob_hash is not None
            and not blob_path.is_file()
        )