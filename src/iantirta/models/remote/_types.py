

import os
from dataclasses import dataclass
from typing import TypedDict


@dataclass(frozen=True, slots=True)
class XetFileData:
    file_hash: str
    refresh_route: str


@dataclass(frozen=True)
class HFFileMeta:
    commit_hash: str | None
    etag: str | None
    location: str
    size: int | None
    xet: XetFileData | None


class DownloadKwargs(TypedDict, total=False):
    cache_dir: str | os.PathLike | None
    force_download: bool
    proxies: dict[str, str] | None
    local_files_only: bool
    token: str | bool | None
    revision: str | None
    subfolder: str
    tqdm_class: type | None
