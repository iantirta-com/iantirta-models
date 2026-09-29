# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

import errno
import hashlib
import logging
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote
from urllib.request import Request, urlopen

import requests
from tqdm import tqdm

from iantirta.models.exceptions import YetToImplement

logger = logging.getLogger(__name__)

HF_HUB_ENDPOINT = "https://huggingface.co"
HF_URL_TEMPLATE = "/{repo_id}/resolve/{revision}/{filename}"
ALL_HF_REPO_TYPES = [
    None, "model", "dataset",
    "space", "kernel"
]

CACHE_HOME = Path("~/.cache/iantirta/hf").expanduser().resolve()
CACHEDIR_TAG_CONTENT = (
    "Signature: 8a477f597d28d172789f06886806bc55\n"
    "# This file is a cache directory tag created by huggingface_hub.\n"
    "# For information about cache directory tags, see:\n"
    "#\thttps://bford.info/cachedir/\n"
)

_CHUNK_SIZE = 128 * 1024



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


@dataclass(slots=True)
class HFDownloadKwargs:
    # Destination
    cache_dir: str | Path | None = None
    local_dir: str | Path | None = None

    # File info
    subfolder: str | None = None
    repo_type: Literal["model", "dataset", "space", "kernel"] | None = None
    revision: str | None = "main"

    # HTTP info
    etag_timeout: float = 10
    endpoint: str = HF_HUB_ENDPOINT
    token: bool | str | None = None
    headers: dict[str, str] | None = None

    # Additional options
    local_files_only: bool = False
    tqdm_class: type[tqdm] | None = None
    dry_run: bool = False
    force_download: bool = False

    # error
    _raise_exceptions_for_gated_repo: bool = True
    _raise_exceptions_for_missing_entries: bool = True
    _raise_exceptions_for_connection_errors: bool = True
    retry_on_errors: bool = False

    user_agent: dict | str | None = None
    
    library_name: str | None = None
    library_version: str | None = None
    
    @classmethod
    def from_dict(
        cls,
        options: dict,
        *,
        unknown: bool = False
    ) -> HFDownloadKwargs | tuple[HFDownloadKwargs, dict[str, Any]]:
        if unknown:
            cls_vars = [f.name for f in fields(cls)]
            unused = {}
            for k, v in options.items():
                if k not in cls_vars:
                    unused.setdefault(k, v)
            options = {k: v for k, v in options.items() if k not in unused}
            return cls(**options), unused
        return cls(**options)


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
    path_or_repo_id: str | Path,
    filenames: list[str],
    options: HFDownloadKwargs | dict | None = None,
    proxies: dict[str, str] | None = None,
    **kwargs,
) -> list[str] | None:
    if not isinstance(options, HFDownloadKwargs):
        if options and isinstance(options, dict):
            options, unknown = HFDownloadKwargs.from_dict(options, unknown=True)
        elif kwargs and isinstance(kwargs, dict):
            options, unknown = HFDownloadKwargs.from_dict(kwargs, unknown=True)
        else:
            options = HFDownloadKwargs()
            unknown = kwargs
    if unknown:
        logger.warning(f"Unused Kwargs: {unknown}")

    if len(filenames) > 1:
        raise YetToImplement("multiple download is not supported.")
    filename = filenames[0]

    if not isinstance(options.revision, str):
        raise YetToImplement(f"using {type(options.revision)} is not supported.")
    elif options.revision != "main":
        raise YetToImplement("using revision other than 'main' is not supported.")

    if options.subfolder == "":
        options.subfolder = None
    if options.subfolder is not None:
        raise YetToImplement("using subfolder is not supported.")

    if options.repo_type is None:
        options.repo_type = "model"
    if options.repo_type not in ALL_HF_REPO_TYPES:
        raise ValueError(
            f"Invalid repo type: {options.repo_type}. Accepted repo types are: {ALL_HF_REPO_TYPES!s}"
        )

    if options.cache_dir is None:
        options.cache_dir = CACHE_HOME
    options.cache_dir = Path(options.cache_dir).expanduser().resolve()

    if options.local_dir is not None:
        raise YetToImplement("using local dir is not supported.")

    if options.token is not None:
        raise YetToImplement("Using token to download is not supported.")

    if options.dry_run:
        raise YetToImplement("Dry run is not supported.")

    if options.local_files_only:
        raise YetToImplement("local file only is not supported.")

    url = options.endpoint.rstrip("/") + HF_URL_TEMPLATE.format(
        repo_id=path_or_repo_id,
        revision=quote(options.revision, safe=""),
        filename=quote(filename)
    )
    
    parts = [f"{options.repo_type}s", *path_or_repo_id.split("/")]
    repo_folder_name: str = "--".join(parts)
    
    storage_folder: Path = options.cache_dir / repo_folder_name
    target_path = storage_folder / options.revision / filename

    if target_path.is_file():
        return [target_path]

    try:
        options.cache_dir.mkdir(parents=True, exist_ok=True)

        if not (tag_path := options.cache_dir / "CACHEDIR.TAG").exists():
            tag_path.write_text(
                CACHEDIR_TAG_CONTENT,
                encoding="utf-8",
            )
    except OSError:
        pass

    return [http_download(
        url,
        target_path,
        progress=True,
    )]
