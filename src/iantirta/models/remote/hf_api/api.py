from pathlib import Path
from urllib.parse import quote
from typing import Any

from .mixin import HFCachedFIle, HFHTTPApi
from .types import HFFileMetadata, ResolvedRevision


class HFApi(HFHTTPApi):

    # {"repo_id": {"repo_type": {"filename": HFCachedFIle}}}
    self.cached_files: dict[str, dict[str, Any]]

    def get_file_metadata(self, url: str, headers: dict[str, str]) -> HFFileMetadata:
        headers["Accept-Encoding"] = "identity"  # prevent any compression => we want to know the real size of the file

        res = self.request_follow_redirect("HEAD", url)

        return HFFileMetadata.from_response(res)

    def hf_hub_download(
        self,
        repo_id: str,
        filename: str,
        *,
        revision: str = "main",
        repo_type: str = "model",
        subfolder: str = "",
        cache_dir: str | None = None
    ) -> str:
        self.repo_id = repo_id

        if not revision:
            revision = "main"
        elif isinstance(revision, ResolvedRevision):
            revision = revision.resolved

        if subfolder:
            filename = f"{subfolder.strip('/')}/{filename.lstrip('/')}"

        if not repo_type:
            repo_type = "model"
        repo_type = repo_type

        cache_file = HFCachedFIle(
            repo_type=repo_type,
            repo_id=repo_id,
            filename=filename,
            cached_dir=cache_dir
        )

    def resolve_revision(
        self,
        repo_id: str,
        *,
        repo_type: str = "model",
        revision: str = "main",
    ):
        if not repo_type:
            repo_type = "model"

        if self.cached_files.get(repo_id):
            pass

        if isinstance(revision, ResolvedRevision):
            # A commit hash means nothing outside of the repo it was resolved for. `_repo_id=None` means the repo is
            # unknown (instance built by hand), in which case it is assumed to fit any repo.
            if revision._repo_id is None or (revision._repo_id, revision._repo_type) == (repo_id, repo_type):
                return revision  # already resolved for this repo => nothing to do
            revision = revision.initial  # resolved for another repo => resolve what was initially requested
        if revision is not None and _COMMIT_HASH_RE.fullmatch(revision):
            return ResolvedRevision(resolved=revision, initial=revision, repo_id=repo_id, repo_type=repo_type)

        try:
            sha = self.repo_info(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
            ).sha
            return ResolvedRevision(
                resolved=sha,
                initial=revision,
                repo_id=repo_id,
                repo_type=repo_type,
            )
        except HFHTTPHubError as e:
            if e.status_code < 500:
                raise
            else:
                error = e
        
    
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
