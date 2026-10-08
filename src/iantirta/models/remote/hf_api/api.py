from pathlib import Path
from urllib.parse import quote

from .mixin import HFCachedFIle, HFHTTPApi
from .types import HFFileMetadata


class HFApi(HFHTTPApi, HFCachedFIle):

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
        self.repo_type = repo_type

        if not cache_dir:
            cache_dir = self.cache_dir
        self.cache_dir = Path(cache_dir).expanduser().resolve()

        return self._cache_download()
        
    
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
