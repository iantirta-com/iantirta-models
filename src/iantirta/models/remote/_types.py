

from dataclasses import dataclass


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


