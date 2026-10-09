import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import requests
from typing_extensions import Self

from . import hf_constant

REPO_API_RE = re.compile(
    r"""
        # staging or production endpoint
        ^https://[^/]+
        (
            # on /api/repo_type/repo_id
            /api/(models|datasets|spaces)/(.+)
            |
            # or /repo_id/resolve/revision/...
            /(.+)/resolve/(.+)
        )
    """,
    flags=re.VERBOSE,
)

BUCKET_API_RE = re.compile(
    r"""
        # staging or production endpoint
        ^https?://[^/]+
        # on /api/buckets/...
        /api/buckets/
    """,
    flags=re.VERBOSE,
)

# Regex to extract the job_id from a (scheduled) job API URL.
# Matches /api/jobs/{namespace}/{job_id}[/...] and /api/scheduled-jobs/{namespace}/{job_id}[/...].
_JOB_ID_FROM_URL_RE = re.compile(r"^https?://[^/]+/api/(?:scheduled-jobs|jobs)/[^/]+/([^/?]+)")

# Regex to extract repo_type and repo_id from API URLs.
# Captures: group(1) = repo_type plural (models/datasets/spaces), group(2) = first path segment, group(3) = optional second segment.
_REPO_ID_FROM_URL_RE = re.compile(r"^https?://[^/]+/api/(models|datasets|spaces)/([^/?]+)(?:/([^/?]+))?")

# Regex to extract repo_type and repo_id from download URLs: /[{repo_type}s/]{repo_id}/resolve/...
# Captures: group(1) = optional repo_type plural (datasets/spaces/kernels), group(2) = repo_id.
_REPO_ID_FROM_RESOLVE_URL_RE = re.compile(
    r"^https?://[^/]+/(?:(datasets|spaces|kernels)/)?([^/?]+(?:/[^/?]+)?)/resolve/"
)

# Regex to extract bucket_id (namespace/name) from bucket API URLs.
_BUCKET_ID_FROM_URL_RE = re.compile(r"^https?://[^/]+/api/buckets/([^/?]+/[^/?]+)")

# Sub-paths that follow a repo_id in API URLs (not part of the repo name).
_REPO_URL_SUBPATHS = {"resolve", "tree", "blob", "raw", "refs", "commit", "discussions", "settings", "revision"}

# Regex to check if the revision IS directly a commit_hash
_COMMIT_HASH_RE = re.compile(r"[0-9a-f]{40}")

_REPO_DIR_RE = re.compile(rf"(?:{'|'.join(sorted(hf_constant.REPO_TYPES_MAPPING))})--.+")

_MARKER_TMP_RE = re.compile(rf"{re.escape(hf_constant.SHARED_BLOBS_MARKER_NAME)}\.[0-9a-f]{{8}}\.tmp")


@dataclass(frozen=True)
class XetFileData:
    file_hash: str
    refresh_route: str

    @classmethod
    def from_response(cls, res: requests.Response) -> "XetFileData":
        try:
            file_hash = res.headers[hf_constant.HUGGINGFACE_HEADER_X_XET_HASH]

            if hf_constant.HUGGINGFACE_HEADER_LINK_XET_AUTH_KEY in res.links:
                refresh_route = res.links[hf_constant.HUGGINGFACE_HEADER_LINK_XET_AUTH_KEY]["url"]
            else:
                refresh_route = res.headers[hf_constant.HUGGINGFACE_HEADER_X_XET_REFRESH_ROUTE]
        except KeyError:
            return None
        return cls(
            file_hash=file_hash,
            refresh_route=refresh_route,
        )


@dataclass(frozen=True)
class HFFileMetadata:
    """Data structure containing information about a file versioned on the Hub.
    """

    commit_hash: str | None
    etag: str | None
    location: str
    size: int | None
    xet_file_data: XetFileData | None

    @classmethod
    def from_response(cls, res: requests.Response) -> "HFFileMetadata":
        etag = (res.headers.get(hf_constant.HUGGINGFACE_HEADER_X_LINKED_ETAG) or res.headers.get("ETag"))
        if etag is not None:
            etag = etag.lstrip("W/").strip('"')

        size_header = res.headers.get(hf_constant.HUGGINGFACE_HEADER_X_LINKED_SIZE) or res.headers.get("Content-Length")
        size = (
            int(size_header)
            if size_header is not None
            else None
        )

        return cls(
            commit_hash=res.headers.get(hf_constant.HUGGINGFACE_HEADER_X_REPO_COMMIT),
            location=res.headers.get("Location") or str(res.request.url),
            xet_file_data=XetFileData.from_response(res),
            etag=etag,
            size=size,
        )


class ResolvedRevision(str):
    """A git revision that has already been resolved to a commit hash.

    `ResolvedRevision` is a `str` subclass, so it can be passed to any `huggingface_hub` method taking a `revision`
    argument. Its string value is the revision initially requested by the user (e.g. `"main"`, `"refs/pr/4"`),
    which keeps URLs and error messages readable, while `.resolved` holds the commit hash it points to.

    Instances are built by [`HfApi.resolve_revision`], which also caches the `revision` -> `commit hash` mapping
    in the local cache (`refs/` folder).

    A commit hash only means something for the repo it was resolved against, so an instance also remembers that
    repo. Re-resolving it for another repo is not an error: the revision initially requested is resolved again
    (see [`HfApi.resolve_revision`]).

    Attributes:
        initial (`str` or `None`):
            The revision initially requested by the user. If `None`, the string value defaults to `"main"`.
        resolved (`str`):
            The commit hash that `initial` resolves to.

    Example:
    ```python
    >>> revision = resolve_revision("openai-community/gpt2")
    >>> revision
    ResolvedRevision(initial=None, resolved='607a30d783dfa663caf39e06633721c8d4cfcd7e')
    >>> revision == "main"  # it's a string
    True
    >>> revision.resolved
    '607a30d783dfa663caf39e06633721c8d4cfcd7e'
    ```
    """

    initial: str | None
    resolved: str
    _repo_id: str | None
    _repo_type: str

    def __new__(
        cls,
        resolved: str,
        initial: str | None = None,
        repo_id: str | None = None,
        repo_type: str | None = None,
    ) -> Self:
        revision = super().__new__(cls, initial if initial is not None else "main")
        revision.initial = initial
        revision.resolved = resolved
        # The repo `resolved` belongs to. `None` means unknown, in which case it is assumed to fit any repo.
        revision._repo_id = repo_id
        revision._repo_type = repo_type or hf_constant.REPO_TYPE_MODEL
        return revision

    def __reduce__(self):
        # without this, pickle/copy rebuild the instance from its string value only, losing the attributes
        return self.__class__, (self.resolved, self.initial, self._repo_id, self._repo_type)

    def __repr__(self) -> str:
        return f"ResolvedRevision(initial={self.initial!r}, resolved={self.resolved!r})"


@dataclass
class ModelInfo:
    id: str         # repo_id
    author: str | None
    # base_models: list[str] | None
    # card_data: ModelCardData | None
    # children_model_count: int | None
    config: dict | None
    created_at: datetime | None
    disabled: bool | None
    downloads: int | None
    # downloads_all_time: int | None
    # eval_results: list[EvalResultEntry] | None
    gated: Literal["auto", "manual", False] | None
    # gguf: dict | None
    # inference: Literal["warm"] | None
    # inference_provider_mapping: list[InferenceProviderMapping] | None
    last_modified: datetime | None
    library_name: str | None        # Transformers
    likes: int | None
    # mask_token: str | None
    # model_index: dict | None
    pipeline_tag: str | None
    private: bool | None
    # resource_group: dict | None
    # safetensors: SafeTensorsInfo | None
    # security_repo_status: dict | None
    sha: str | None
    # siblings: list[RepoSibling] | None
    # spaces: list[str] | None
    tags: list[str] | None
    # transformers_info: TransformersInfo | None
    # trending_score: int | None
    used_storage: int | None
    # widget_data: Any | None

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)