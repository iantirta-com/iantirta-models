import json
import os
from pathlib import Path

import requests

from ._cache import try_to_load_from_cache
from ._http import request_follow_redirect
from .api import hf_hub_download, hf_hub_url

__all__ = [
    "cached_file",
    "cached_files",
    "has_file",
]


def _get_cache_file_to_return(
    path_or_repo_id: str,
    full_filename: str,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    repo_type: str | None = None,
):
    # We try to see if we have a cached version (not up to date):
    resolved_file = try_to_load_from_cache(
        path_or_repo_id,
        full_filename,
        cache_dir=cache_dir,
        revision=revision,
        repo_type=repo_type,
    )
    if resolved_file is not None:
        return resolved_file

    return None


def cached_file(
    path_or_repo_id: str | os.PathLike,
    filename: str,
    **kwargs,
) -> str | None:
    file = cached_files(
        path_or_repo_id=path_or_repo_id,
        filenames=[filename],
        **kwargs
    )
    file = file[0] if file is not None else file
    return file


def cached_files(
    path_or_repo_id: str | os.PathLike,
    filenames: list[str],
    subfolder: str = "",
    revision: str = "main",
    cache_dir: str | Path | None = None,
    **kwargs,
) -> list[str] | None:

    if not revision:
        revision = "main"

    subfolder = subfolder.strip("/")

    # Add folder to filenames
    full_filenames = [
        os.path.join(subfolder, file)
        for file in filenames
    ]

    if Path(path_or_repo_id).is_dir():
        existing_files = []
        for filename in full_filenames:
            resolved_file = os.path.join(path_or_repo_id, filename)
            if (
                not Path(resolved_file).is_file()
                and filename != os.path.join(subfolder, "config.json")
            ):
                raise OSError(
                    f"{path_or_repo_id} does not appear to "
                    f"have a file named {filename}. Checkout "
                    f"'https://huggingface.co/{path_or_repo_id}"
                    f"/tree/{revision}' for available files."
                )
            existing_files.append(resolved_file)
        return existing_files if existing_files else None

    def finalize(
        resolved_files: list[str | None]
    ) -> list[str] | None:
        if any(file is None for file in resolved_files):
            missing_entries = [
                original
                for original, resolved in zip(
                    full_filenames, resolved_files
                )
                if resolved is None
            ]
            
            # Last escape
            if (
                len(resolved_files) == 1
                and missing_entries[0] == os.path.join(subfolder, "config.json")
            ):
                return None

            # Now we raise for missing entries
            revision_ = (
                "main"
                if revision is None
                else revision
            )
            msg = (
                f"a file named {missing_entries[0]}"
                if len(missing_entries) == 1
                else f"files named {(*missing_entries,)}"
            )
            raise OSError(
                f"{path_or_repo_id} does not appear to have {msg}. "
                f"Checkout 'https://huggingface.co/{path_or_repo_id}"
                f"/tree/{revision_}' for available files."
            )
        
        # Remove potential missing entries (we can silently remove them at this point based on the flags)
        resolved_files = [
            file
            for file in resolved_files
            if file is not None
        ]
        # Return `None` if the list is empty, coherent with other Exception when the flag is not active
        resolved_files = (
            None
            if len(resolved_files) == 0
            else resolved_files
        )

        return resolved_files

    # Using commit hash revision for cache?

    try:
        if len(full_filenames) == 1:
            result = hf_hub_download(
                path_or_repo_id,
                filenames[0],
                subfolder=None if len(subfolder) == 0 else subfolder,
                revision=revision,
                cache_dir=cache_dir,
            )
            result = str(result)
        else:
            raise NotImplementedError()
    except Exception:  # noqa: TRY203
        # try to recover?
        raise

    # resolved_files = [
    #     _get_cache_file_to_return(
    #         path_or_repo_id,
    #         filename,
    #         cache_dir,
    #         revision
    #     )
    #     for filename in full_filenames
    # ]
    # return finalize(resolved_files)
    return finalize([result])


def has_file(
    path_or_repo: str | os.PathLike,
    filename: str,
    revision: str | None = None,
    proxies: dict[str, str] | None = None,
    token: bool | str | None = None,
    *,
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
    repo_type: str | None = None,
    **deprecated_kwargs,
):
    """
    Checks if a repo contains a given file without downloading it. Works for remote repos and local folders.

    If offline mode is enabled, checks if the file exists in the cache.

    <Tip warning={false}>

    This function will raise an error if the repository `path_or_repo` is not valid or if `revision` does not exist for
    this repo, but will return False for regular connection errors.

    </Tip>
    """
    # If path to local directory, check if the file exists
    if os.path.isdir(path_or_repo):
        return os.path.isfile(os.path.join(path_or_repo, filename))

    # Else it's a repo => let's check if the file exists in local cache or on the Hub

    # Check if file exists in cache
    # This information might be outdated so it's best to also make a HEAD call (if allowed).
    cached_path = try_to_load_from_cache(
        repo_id=path_or_repo,
        filename=filename,
        revision=revision,
        repo_type=repo_type,
        cache_dir=cache_dir,
    )
    has_file_in_cache = isinstance(cached_path, str)

    # If local_files_only, don't try the HEAD call
    if local_files_only:
        return has_file_in_cache

    # Check if the file exists
    try:
        res = request_follow_redirect(
            "HEAD",
            hf_hub_url(path_or_repo, filename=filename, revision=revision, repo_type=repo_type),
            headers={},
        )
        res.raise_for_status()
        return True
    except requests.exceptions.ProxyError:
        # Actually raise for those subclasses of ConnectionError
        raise
    except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout):
        return has_file_in_cache
    except Exception:
        raise


def get_checkpoint_shard_files(
    pretrained_model_name_or_path,
    index_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    local_files_only=False,
    token=None,
    user_agent=None,
    revision=None,
    subfolder="",
    tqdm_class=None,
    **deprecated_kwargs,
):
    """
    For a given model:

    - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a model ID on the
      Hub
    - returns the list of paths to all the shards, as well as some metadata.

    For the description of each arg, see [`PreTrainedModel.from_pretrained`]. `index_filename` is the full path to the
    index (downloaded and cached if `pretrained_model_name_or_path` is a model ID on the Hub).
    """
    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")

    with open(index_filename, encoding="utf-8") as f:
        index = json.loads(f.read())

    shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()

    # First, let's deal with local folder.
    if os.path.isdir(pretrained_model_name_or_path):
        shard_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in shard_filenames]
        return shard_filenames, sharded_metadata

    # At this stage pretrained_model_name_or_path is a model identifier on the Hub. Try to get everything from cache,
    # or download the files
    cached_filenames = cached_files(
        pretrained_model_name_or_path,
        shard_filenames,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        local_files_only=local_files_only,
        token=token,
        user_agent=user_agent,
        revision=revision,
        subfolder=subfolder,
        tqdm_class=tqdm_class,
    )

    return cached_filenames, sharded_metadata
