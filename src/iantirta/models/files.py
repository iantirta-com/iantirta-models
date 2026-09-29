# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

import os
import errno
import hashlib
import tempfile
import uuid
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from tqdm import tqdm

HF_HUB_ENDPOINT = "https://huggingface.co"

CACHE_HOME = Path("~/.cache/iantirta/hf").expanduser().resolve()
CACHEDIR_TAG_CONTENT = (
    "Signature: 8a477f597d28d172789f06886806bc55\n"
    "# This file is a cache directory tag created by huggingface_hub.\n"
    "# For information about cache directory tags, see:\n"
    "#\thttps://bford.info/cachedir/\n"
)

_CHUNK_SIZE = 128 * 1024
ALL_HF_REPO_TYPES = [
    None, "model", "dataset",
    "space", "kernel"
]


def http_download(
    url: str,
    destination: Path,
    hash_prefix: str | None = None,
    progress: bool = True,
) -> Path:
    """Download a URL to a local file atomically.

    The content is first written to a temporary file in the destination
    directory. Once the download succeeds, the temporary file is renamed
    to ``destination``. This prevents a partially downloaded file from
    being mistaken for a complete file.

    Parameters
    ----------
    url:
        URL of the resource to download.

    destination:
        Path where the downloaded file should be stored. Parent directories
        are created automatically.

    hash_prefix:
        Optional hexadecimal SHA-256 digest prefix used to verify the
        downloaded content. If provided, the beginning of the computed
        SHA-256 digest must match this value.

    progress:
        Whether to display a download progress bar.

    Returns
    -------
    pathlib.Path
        The resolved destination path.

    Raises
    ------
    ValueError
        If ``hash_prefix`` is empty or the downloaded content does not
        match the supplied hash prefix.

    OSError
        If the destination cannot be created or written.

    urllib.error.URLError
        If the resource cannot be downloaded.

    urllib.error.HTTPError
        If the server returns an HTTP error response.
    """
    destination = Path(destination).expanduser().resolve()

    if hash_prefix is not None:
        if not hash_prefix:
            raise ValueError("hash_prefix must not be empty")

        hash_prefix = hash_prefix.lower()

        if any(char not in "0123456789abcdef" for char in hash_prefix):
            raise ValueError("hash_prefix must contain hexadecimal characters")

    destination.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_path = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".partial",
    )

    temporary_file = Path(temporary_path)

    try:
        with open(fd, "w+b", closefd=True) as file:
            request = Request(
                url,
                headers={
                    "User-Agent": "iantirta-models",
                },
            )

            with urlopen(request) as response:
                content_length = response.headers.get("Content-Length")
                file_size = (
                    int(content_length)
                    if content_length is not None
                    else None
                )

                sha256 = (
                    hashlib.sha256()
                    if hash_prefix is not None
                    else None
                )

                with tqdm(
                    total=file_size,
                    disable=not progress,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    while chunk := response.read(_CHUNK_SIZE):
                        file.write(chunk)

                        if sha256 is not None:
                            sha256.update(chunk)

                        pbar.update(len(chunk))

            file.flush()

        if sha256 is not None:
            digest = sha256.hexdigest()

            if not digest.startswith(hash_prefix):
                raise ValueError(
                    "Downloaded file has an invalid SHA-256 hash "
                    f'(expected prefix "{hash_prefix}", got "{digest}")'
                )

        temporary_file.replace(destination)

    finally:
        temporary_file.unlink(missing_ok=True)

    return destination


def _get_target_path(
    filename: str,
    path_or_repo: str | Path,
    *,
    revision: str = "main",
) -> Path:
    """Return the local cache path for a repository file.

    Parameters
    ----------
    filename:
        File name inside the model repository.

    path_or_repo:
        Hugging Face repository identifier, such as
        ``"facebook/mms-1b-all"``.

    revision:
        Repository revision used to distinguish cached files belonging to
        different revisions.

    Returns
    -------
    pathlib.Path
        Path where the file should be cached.
    """
    cache_home = CACHE_HOME.expanduser()

    repo = str(path_or_repo).strip("/")
    repo_key = "--".join(repo.split("/"))

    revision_key = revision.replace("/", "--")

    target_dir = (
        cache_home / 
        f"models--{repo_key}" /
        revision_key
    )

    try:
        cache_home.mkdir(parents=True, exist_ok=True)

        if not (tag_path := cache_home / "CACHEDIR.TAG").exists():
            tag_path.write_text(
                CACHEDIR_TAG_CONTENT,
                encoding="utf-8",
            )
    except OSError:
        # The actual download will report the relevant filesystem error
        # if the cache cannot be used.
        pass

    return target_dir / filename


def ensure_file(
    path_or_repo: str | Path,
    filename: str | None = None,
    *,
    endpoint: str = HF_HUB_ENDPOINT,
    revision: str = "main",
    progress: bool = True,
) -> Path:
    """Resolve a local repository file or download a Hugging Face file.

    If ``path_or_repo`` is a local directory, ``filename`` is resolved
    relative to that directory and must already exist.

    Otherwise ``path_or_repo`` is treated as a Hugging Face repository
    identifier and the requested file is downloaded into the Iantirta
    model cache. Existing cached files are reused.

    Parameters
    ----------
    path_or_repo:
        Either a local directory or a Hugging Face repository identifier,
        for example ``"facebook/mms-1b-all"``.

    filename:
        Repository-relative filename, such as ``"config.json"`` or
        ``"model.safetensors"``. Required for both local-directory and
        remote-repository usage.

    endpoint:
        Base URL of the model hub.

    revision:
        Repository revision to download from. This can be a branch, tag,
        or commit identifier.

    progress:
        Whether to display a download progress bar for remote files.

    Returns
    -------
    pathlib.Path
        Local path to the requested file.

    Raises
    ------
    ValueError
        If ``filename`` is not provided.

    FileNotFoundError
        If a local directory is supplied but the requested file does not
        exist.

    NotADirectoryError
        If ``path_or_repo`` exists locally but is not a directory.

    urllib.error.URLError
        If a remote file cannot be downloaded.

    urllib.error.HTTPError
        If the model hub returns an HTTP error response.
    """
    if filename is None:
        raise ValueError("filename is required")

    local_path = Path(path_or_repo).expanduser()

    if local_path.exists():
        if not local_path.is_dir():
            raise NotADirectoryError(
                f"Expected a directory, got: {local_path}"
            )

        target = local_path / filename

        if not target.is_file():
            raise FileNotFoundError(
                f"File not found: {target}"
            )

        return target

    repo = str(path_or_repo).strip("/")

    if not repo:
        raise ValueError("path_or_repo must not be empty")

    target_path = _get_target_path(
        filename,
        repo,
        revision=revision,
    )

    if target_path.is_file():
        return target_path

    encoded_repo = quote(repo, safe="/")
    encoded_revision = quote(revision, safe="/")
    encoded_filename = quote(filename, safe="/")

    url = (
        f"{endpoint.rstrip('/')}/"
        f"{encoded_repo}/resolve/"
        f"{encoded_revision}/"
        f"{encoded_filename}"
    )

    return http_download(
        url,
        target_path,
        progress=progress,
    )


def _get_hf_url(
    repo_id: str,
    filename: str,
    *,
    subfolder: str | None = None,
    repo_type: str | None = None,
    revision: str | None = None,
    endpoint: str | None = None,
) -> str:
    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"

    if repo_type not in constants.REPO_TYPES_WITH_KERNEL:
        raise ValueError("Invalid repo type")

    if repo_type in constants.REPO_TYPES_URL_PREFIXES:
        repo_id = constants.REPO_TYPES_URL_PREFIXES[repo_type] + repo_id  # type: ignore

    if revision is None:
        revision = constants.DEFAULT_REVISION
    url = constants.HUGGINGFACE_CO_URL_TEMPLATE.format(
        repo_id=repo_id, revision=quote(revision, safe=""), filename=quote(filename)
    )
    # Update endpoint if provided
    if endpoint is not None and url.startswith(constants.ENDPOINT):
        url = endpoint + url[len(constants.ENDPOINT) :]
    return url

    
def _get_metadata_or_catch_error(
    *,
    repo_id: str,
    filename: str,
    repo_type: str,
    revision: str,
    endpoint: str | None,
    etag_timeout: float | None,
    headers: dict[str, str],  # mutated inplace!
    token: bool | str | None,
    local_files_only: bool,
    relative_filename: str | None = None,  # only used to store `.no_exists` in cache
    storage_folder: str | None = None,  # only used to store `.no_exists` in cache
    retry_on_errors: bool = False,
    tree_cache_folder: str | None = None,  # if set, read the on-disk tree listing to skip the HEAD call
) -> (
    # Either an exception is caught and returned
    tuple[None, None, None, None, None, Exception]
    |
    # Or the metadata is returned as
    # `(url_to_download, etag, commit_hash, expected_size, xet_file_data, None)`
    tuple[str, str, str, int | None, XetFileData | None, None]
):
    if local_files_only:
        return (
            None,
            None,
            None,
            None,
            None,
            ConnectionError(
                f"Cannot access file since 'local_files_only=True' as been set. (repo_id: {repo_id}, repo_type: {repo_type}, revision: {revision}, filename: {filename})"
            ),
        )
    if tree_cache_folder is not None:
        if revision is not None and revision != "main":
            raise YetToImplement("using commit hash for revision is not supported.")
 

def _hf_dl_cache(
    *,
    # Destination
    cache_dir: Path,
    # File info
    repo_id: str,
    filename: str,
    repo_type: str,
    revision: str,
    # HTTP info
    endpoint: str | None,
    etag_timeout: float,
    headers: dict[str, str],
    token: bool | str | None,
    # Additional options
    local_files_only: bool,
    force_download: bool,
    tqdm_class: type[base_tqdm] | None,
    dry_run: bool,
) -> Path:
    repo_folder_name = f"{repo_type}s" + "--".join(repo.split("/")
    locks_dir = cache_dir / ".locks"
    storage_folder = cache_dir / repo_folder_name

    # cross-platform transcription of filename, to be used as a local file path.
    relative_filename = os.path.join(*filename.split("/"))

    if revision is not None and revision != "main":
        raise YetToImplement("using commit hash for revision is not supported.")

    (
        url, etag, commit_hash,
        expected_size, xet_fd,
        head_call_error
    ) = _get_metadata_or_catch_error(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
        endpoint=endpoint,
        etag_timeout=etag_timeout,
        headers=headers,
        token=token,
        local_files_only=local_files_only,
        storage_folder=storage_folder,
        relative_filename=relative_filename,
        tree_cache_folder=storage_folder,
    )


def hf_dl(
    repo_id: str,
    filename: str,
    *,
    subfolder: str | None = None,
    repo_type: str | None = None,
    revision: str | None = "main",
    library_name: str | None = None,
    library_version: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    user_agent: dict | str | None = None,
    force_download: bool = False,
    etag_timeout: float = 10,
    token: bool | str | None = None,
    local_files_only: bool = False,
    headers: dict[str, str] | None = None,
    endpoint: str | None = None,
    tqdm_class: type[tqdm] | None = None,
    dry_run: bool = False,
) -> Path:
    
    if revision is None:
        revision = "main"
    elif not isinstance(revision, str):
        raise YetToImplement(f"using {type(revision)} is not supported.")
    
    if cache_dir is None:
        cache_dir = CACHE_HOME
    cache_dir = Path(cache_dir).expanduser().resolve()

    if local_dir is not None:
        local_dir = Path(local_dir).expanduser().resolve()

    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        # This is used to create a URL, and not a local path, hence the forward slash.
        filename = f"{subfolder}/{filename}"

    if repo_type is None:
        repo_type = "model"

    if repo_type not in ALL_HF_REPO_TYPES:
        raise ValueError(
            f"Invalid repo type: {repo_type}. Accepted repo types are: {str(ALL_REPO_TYPES)}"
        )

    if local_dir is not None:
        raise YetToImplement("download to local dir is not supported.")
    else:
        return _hf_dl_cache(
            # Destination
            cache_dir=cache_dir,
            # File info
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            # HTTP info
            endpoint=endpoint,
            etag_timeout=etag_timeout,
            headers=headers,
            token=token,
            # Additional options
            local_files_only=local_files_only,
            force_download=force_download,
            tqdm_class=tqdm_class,
            dry_run=dry_run,
        )


def cached_file(
    path_or_repo_id: str | Path,
    filename: str,
    **kwargs,
) -> Path | None:
    file = cached_files(
        path_or_repo_id=path_or_repo_id,
        filenames=[filename],
        **kwargs
    )
    file = (
        file[0]
        if file is not None
        else file
    )
    return file


def cached_files(
    path_or_repo_id: str | Patg,
    filenames: list[str],
    
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    proxies: dict[str, str] | None = None,
    token: bool | str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    subfolder: str = "",
    
    repo_type: str | None = None,
    user_agent: str | dict[str, str] | None = None,
    
    _raise_exceptions_for_gated_repo: bool = True,
    _raise_exceptions_for_missing_entries: bool = True,
    _raise_exceptions_for_connection_errors: bool = True,
    
    tqdm_class: type | None = None,
    
    **deprecated_kwargs,
) -> list[str] | None:
    
    full_filenames = [
        os.path.join(subfolder, file)
        for file in filenames
    ]
    existing_files = []
    for filename in full_filenames:
        if (path_or_repo_id := Path(path_or_repo_id)).is_dir():
            if not (resolved_file := path_or_repo_id / filename).is_file():
                if (
                    _raise_exceptions_for_missing_entries
                    and filename != os.path.join(subfolder, "config.json")
                ):
                    revision_ = "main" if revision is None else revision
                    raise OSError(
                        f"{path_or_repo_id} does not appear to have a file named {filename}. Checkout "
                        f"'https://huggingface.co/{path_or_repo_id}/tree/{revision_}' for available files."
                    )
                else:
                    continue
            existing_files.append(resolved_file)

    if path_or_repo_id.is_dir():
        return (
            existing_files
            if existing_files
            else None
        )

    
    def finalize(
        resolved_files: list[str | None]
    ) -> list[str] | None:
        # If there are any missing file and the flag is active, raise
        if any(
            file is None
            for file in resolved_files
        ) and _raise_exceptions_for_missing_entries:
            missing_entries = [
                original
                for original, resolved in zip(
                    full_filenames,
                    resolved_files
                ) if resolved is None
            ]
            # Last escape
            if (
                len(resolved_files) == 1
                and missing_entries[0] == os.path.join(
                    subfolder, "config.json"
                )
            ):
                return None
            # Now we raise for missing entries
            revision_ = "main" if revision is None else revision
            msg = (
                f"a file named {missing_entries[0]}"
                if len(missing_entries) == 1
                else f"files named {(*missing_entries,)}"
            )
            raise OSError(
                f"{path_or_repo_id} does not appear to have {msg}. Checkout 'https://huggingface.co/{path_or_repo_id}/tree/{revision_}'"
                " for available files."
            )

        # Remove potential missing entries (we can silently remove them at this point based on the flags)
        resolved_files = [
            file
            for file in resolved_files
            if file is not None
        ]
        # Return `None` if the list is empty, coherent with other Exception when the flag is not active
        resolved_files = None if len(resolved_files) == 0 else resolved_files

        return resolved_files

    if cache_dir is None:
        cache_dir = CACHE_HOME

    file_counter = 0
    if revision is not None and revision != "main":
        raise YetToImplement("using commit hash for revision is not supported.")

    if file_counter == len(full_filenames):
        return finalize(existing_files)

    try:
        if len(full_filenames) == 1:
            hf_dl()
        else:
            hf_snapshot_dl()
    except Exception as e:
        if isinstance(e, PermissionError):
            raise OSError(
                f"PermissionError at {e.filename} when downloading {path_or_repo_id}. "
                "Check cache directory permissions. Common causes: 1) another user is downloading the same model (please wait); "
                "2) a previous download was canceled and the lock file needs manual removal."
            ) from e
        elif isinstance(e, OSError) and e.errno == errno.EROFS:
            # Unlike EACCES (errno 13), which Python maps to PermissionError,
            # EROFS (errno 30) is a plain OSError that does NOT match `isinstance(e, PermissionError)`.
            # Without this guard it would fall through to the stale-cache recovery block below,
            # silently returning an old cached file even when a newer revision exists on the Hub.
            # Re-raise so callers can detect the read-only condition and retry with a writable path.
            raise
        elif isinstance(e, ValueError):
            raise OSError(f"{e}") from e

        # Now we try to recover if we can find all files correctly in the cache
        resolved_files = [
            _get_cache_file_to_return(
                path_or_repo_id,
                filename,
                cache_dir,
                commit_hash or revision,
                repo_type
            ) for filename in full_filenames
        ]
        if all(
            file is not None
            for file in resolved_files
        ):
            return resolved_files

    resolved_files = [
        _get_cache_file_to_return(
            path_or_repo_id,
            filename,
            cache_dir,
            commit_hash or revision
        ) for filename in full_filenames
    ]
    return finalize(resolved_files)
