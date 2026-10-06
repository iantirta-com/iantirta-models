import os
from pathlib import Path

from ._cache import try_to_load_from_cache
from .api import hf_hub_download

__all__ = [
    "cached_file",
    "cached_files",
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
    except Exception:
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
