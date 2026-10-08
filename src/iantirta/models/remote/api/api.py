from urllib.parse import quote


class HFApi(HTTPMixin):
    def __init__(
        self,
    ):
        self.endpoint = "https://huggingface.co"

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
