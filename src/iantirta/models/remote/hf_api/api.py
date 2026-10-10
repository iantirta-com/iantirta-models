import logging
import uuid
from pathlib import Path
from urllib.parse import quote

import requests

from . import _xet, hf_constant
from .errors import (
    FileMetadataError,
    HFHubHTTPError,
    RemoteEntryNotFoundError,
    RevisionNotFoundError,
    RevisionResolutionError,
)
from .mixin import HFCachedFile, HFHTTPApi
from .types import HFFileMetadata, ModelInfo, ResolvedRevision, XetFileData

logger = logging.getLogger(__name__)


class HFApi(HFHTTPApi, _xet.HFXetAPI):

    def get_cache_file(
        self,
        repo_id: str,
        repo_type: str,
        filename: str | None = None,
        revision: str | None = None,
        **kwargs,
    ) -> HFCachedFile:
        return HFCachedFile.get_cached_file(repo_id=repo_id, repo_type=repo_type, filename=filename, revision=revision, **kwargs)

    def get_file_metadata(self, url: str, headers: dict[str, str] | None = None) -> HFFileMetadata:
        headers = headers or {}
        headers["Accept-Encoding"] = "identity"  # prevent any compression => we want to know the real size of the file

        res = self.request_follow_redirect("HEAD", url)
        self.raise_for_status(res)

        return HFFileMetadata.from_response(res)

    def hf_hub_download(
        self,
        repo_id: str,
        filename: str,
        *,
        revision: str = "main",
        repo_type: str = "model",
        subfolder: str = "",
        cache_dir: str | None = None,
        local_only: bool | None = None,
    ) -> Path:
        self.repo_id = repo_id

        if not revision:
            revision = "main"
        elif isinstance(revision, ResolvedRevision):
            revision = revision.resolved

        if subfolder:
            filename = f"{subfolder.strip('/')}/{filename.lstrip('/')}"

        if not repo_type:
            repo_type = "model"

        cache_file: HFCachedFile = self.get_cache_file(
            repo_type=repo_type,
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            revision=revision
        )

        if local_only and cache_file.revision_is_commit and cache_file.pointer_path.exists():
            return cache_file.pointer_path

        url = self.hf_hub_url(repo_id, filename, repo_type=repo_type, revision=revision)
        try:
            file_metadata: HFFileMetadata = self.get_file_metadata(url,)
        except RemoteEntryNotFoundError as e:
            if cache_file.storage_dir is not None and cache_file.filename is not None:
                # Cache non existant of the file
                commit_hash = e.response.headers.get(hf_constant.HUGGINGFACE_HEADER_X_REPO_COMMIT)
                if commit_hash is not None:
                    cache_file.revision = commit_hash
                    cache_file.cache_no_exists()
            raise
        except RevisionNotFoundError as e:
            raise
        except Exception as e:
            if isinstance(e, (requests.exceptions.ConnectTimeout, requests.Timeout, requests.HTTPError, HFHubHTTPError)):
                # Try to get from cache file
                if local_only != False and not cache_file.revision_is_commit and cache_file.ref_path.is_file():
                    self.revision = cache_file.ref_path.read_text()
                    if cache_file.pointer_path.exists():
                        return cache_file.pointer_path
                raise
            else:
                raise

        # Commit hash must exist
        if file_metadata.commit_hash is None:
            raise FileMetadataError(
                f"Response from {url} is missing the '{hf_constant.HUGGINGFACE_HEADER_X_REPO_COMMIT}' header, so it"
                " does not seem to be served by a Hugging Face Hub endpoint. If HF_ENDPOINT is set, check that it"
                " points to a Hub-compatible endpoint. Otherwise, check your firewall and proxy settings and make"
                " sure your SSL certificates are updated."
            )

        # Etag must exist
        # If we don't have any of those, raise an error.
        if file_metadata.etag is None:
            raise FileMetadataError(
                "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
            )

        # Xet downloads require a known size, but regular HTTP downloads can recover it from the GET response.
        if file_metadata.size is None and file_metadata.xet_file_data is not None and _xet.available():
            raise FileMetadataError("Distant resource does not have a Content-Length.")

        if file_metadata.xet_file_data is None and url != file_metadata.location:
            url = file_metadata.location

        cache_file.revision = file_metadata.commit_hash
        cache_file.etag = file_metadata.etag

        if local_only != False and cache_file.pointer_path.is_file():
            return cache_file.pointer_path

        cache_file.pointer_path.parent.mkdir(parents=True, exist_ok=True)
        cache_file.blob_path.parent.mkdir(parents=True, exist_ok=True)
        cache_file.lock_path.parent.mkdir(parents=True)

        if local_only != False and cache_file.blob_path.exists():
            with cache_file.FileLock(cache_file.lock_path):
                if not cache_file.pointer_path.exists():
                    cache_file.create_symlink(cache_file.blob_path, cache_file.pointer_path, new_blob=False)
                return cache_file.pointer_path

        # Local file doesn't exist or etag isn't a match => retrieve file from remote (or cache)

        # Xet hash of the blob in the shared store, None when the store must not be used for this download
        # (see `utils/_shared_blobs.py`).
        shared_blob_hash = (
            file_metadata.xet_file_data.file_hash
            if file_metadata.xet_file_data is not None and cache_file(cache_file.cache_dir)
            else None
        )
        cache_file.xet_hash = shared_blob_hash
    
        with cache_file.FileLock(cache_file.lock_path):
            reused_blob_from_store = (
                shared_blob_hash is not None
                and local_only != False
                and not cache_file.blob_path.exists()
                and cache_file.is_shared_blob_exist(expected_size=file_metadata.size)
            )

            blob_is_shared = reused_blob_from_store
            if not reused_blob_from_store:
                will_download = local_only != False or not cache_file.blob_path.exists()
                self.cache_download(
                    incomplete_path=cache_file.blob_path.with_suffix(".incomplete"),
                    destination_path=cache_file.blob_path,
                    url=url,
                    cache_file=cache_file,
                    expected_size=file_metadata.size,
                    xet_file_data=file_metadata.xet_file_data
                )

                if shared_blob_hash is not None and will_download and _xet.available():
                    blob_is_shared = cache_file.store_shared_blob(
                        expected_size=file_metadata.size,
                        replace_existing=local_only == False
                    )

            if not cache_file.pointer_path.exists():
                cache_file.create_symlink(cache_file.blob_path, cache_file.pointer_path, new_blob = not blob_is_shared)

        return cache_file.pointer_path

    def cache_download(
        self,
        *,
        url: str,
        incomplete_path: Path,
        destination_path: Path,
        cache_file: HFCachedFile,
        expected_size: int | None,
        local_only: bool | None = None,
        xet_file_data: XetFileData | None = None
    ) -> None:
        """Download content from a URL to a destination path."""
        if destination_path.exists() and local_only != False:
            return
        assert cache_file.filename is not None, "Downloading require a filename"
        tmp_path = incomplete_path.with_name(f"{incomplete_path.stem}.{uuid.uuid4().hex[:8]}.incomplete")
        tmp_path = cache_file.as_extended_path(tmp_path)

        try:
            with tmp_path.open("wb") as f:
                logger.debug(f"Downloading '{cache_file.filename}' to '{tmp_path}'")

                if expected_size is not None:  # might be None if HTTP header not set correctly
                    # Check disk space in both tmp and destination path
                    cache_file.check_disk_space(expected_size, tmp_path.parent)
                    cache_file.check_disk_space(expected_size, destination_path.parent)

                if xet_file_data is not None and _xet.available():
                    logger.debug("Xet Storage is enabled for this repo. Downloading file from Xet Storage..")
                    self.xet_download(
                        incomplete_path=tmp_path,
                        xet_file_data=xet_file_data,
                        expected_size=expected_size,
                        displayed_filename=cache_file.filename,
                    )
                else:
                    self.download(
                        url=url,
                        temp_file=f,
                        expected_size=expected_size,
                        displayed_filename=cache_file.filename,
                    )
            logger.debug(f"Download complete. Moving file to {destination_path}")
            cache_file.chmod_and_move(tmp_path, destination_path)
        finally:
            # No-op on success (file has been moved). On failure, do not keep a partial file around:
            # it could not be reused anyway since the temporary name is unique to this download.
            tmp_path.unlink(missing_ok=True)

    def model_info(self, repo_id: str, *, revision: str ="main"):
        if not revision:
            revision = "main"

        path = f"/api/models/{repo_id}/revision/{quote(revision, safe='')}"
        res = self.request("get", path, params={})
        self.raise_for_status(res)
        data = res.json()
        return ModelInfo(**data)

    def repo_info(self, repo_id: str, *, repo_type: str = "model", revision: str = "main"):
        """Get the info object for a given repo of a given type."""
        match repo_type:
            case None | "model":
                fn = self.model_info
            case _:
                raise ValueError("Unsupported repo type.")
        return fn(repo_id, revision=revision)

    def resolve_revision(
        self,
        repo_id: str,
        *,
        repo_type: str = "model",
        revision: str = "main",
        local_only: bool = False,
    ) -> ResolvedRevision:
        if not repo_type:
            repo_type = "model"

        if not revision:
            revision = "main"

        cache_file = self.get_cache_file(
            repo_type=repo_type,
            repo_id=repo_id,
        )

        if isinstance(revision, ResolvedRevision):
            # A commit hash means nothing outside of the repo it was resolved for. `_repo_id=None` means the repo is
            # unknown (instance built by hand), in which case it is assumed to fit any repo.
            if revision._repo_id is None or (revision._repo_id, revision._repo_type) == (repo_id, repo_type):
                return revision  # already resolved for this repo => nothing to do
            revision = revision.initial  # resolved for another repo => resolve what was initially requested
        if revision is not None:
            cache_file.revision = revision
            if cache_file.revision_is_commit:
                return ResolvedRevision(resolved=revision, initial=revision, repo_id=repo_id, repo_type=repo_type)

        error: Exception | None = None
        if not local_only:
            try:
                sha = self.repo_info(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=revision,
                ).sha
                assert sha is not None, "Repo info returned from server must have a revision sha."
                cache_file.revision = sha
                return ResolvedRevision(
                    resolved=sha,
                    initial=revision,
                    repo_id=repo_id,
                    repo_type=repo_type,
                )
            except HFHubHTTPError as e:
                if e.response.status_code < 500:
                    raise
                else:
                    error = e

        # At this point, it is not available online or offline
        if cache_file.ref_path.is_file():
            if error is not None:
                logger.warning(f"Could not reach the Huggingface Hub ({error}). Using cached commit hash for '{repo_id}'.")
            return ResolvedRevision(
                resolved=cache_file.ref_path.read_text().strip(),
                initial=revision,
                repo_id=repo_id,
                repo_type=repo_type
            )

        reason = (
            "'local_files_only=True' is set"
            if error is None
            else f"the Huggingface Hub could not be reached ({error.__class__.__name__}: {error})"
        )
        raise RevisionResolutionError(
            f"Cannot resolve revision '{revision}' for {repo_type} '{repo_id}':"
            f" {reason} and no matching entry was found in the local cache ('{cache_file.ref_path}')."
        ) from error
    
    def list_repo_tree(
        self,
        repo_id: str,
        subfolder: str,
        *,
        revision: str = "main",
        repo_type: str = "model",
        recursive: bool = False,
        expand: bool = False,
    ):
        revision = revision or "main"
        repo_type = repo_type or "model"
        encoded_subfolder = (
            "/" + quote(subfolder, safe="")
            if subfolder
            else ""
        )
        path = (
            f"api/{repo_type}s/{repo_id}/"
            f"tree/{revision}{encoded_subfolder}"
        )
        self.paginate(
            path=path,
            params={
                "recursive": recursive, "expand": expand
            }
        )


if __name__ == "__main__":
    from rich import inspect
    # repo_id = "Qwen/Qwen3-ASR-1.7B-hf"
    repo_id = "facebook/mms-1b-all"
    api = HFApi()
    inspect(api.resolve_revision(repo_id, local_only=False))
    cache_file = api.get_cache_file(repo_id=repo_id, repo_type="model")
    inspect(cache_file)