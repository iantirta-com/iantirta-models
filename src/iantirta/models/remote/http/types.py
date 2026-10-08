import re
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property

# Regex patterns for parsing rate limit headers
# e.g.: "api";r=0;t=55 --> resource_type="api", r=0, t=55
_RATELIMIT_RE = re.compile(r"\"(?P<resource_type>\w+)\"\s*;\s*r\s*=\s*(?P<r>\d+)\s*;\s*t\s*=\s*(?P<t>\d+)")
# e.g.: "fixed window";"api";q=500;w=300 --> q=500, w=300
_RATELIMIT_POLICY_RE = re.compile(r"q\s*=\s*(?P<q>\d+).*?w\s*=\s*(?P<w>\d+)")
# Regex to get filename from a "Content-Disposition" header for CDN-served files
_HEADER_FILENAME_RE = re.compile(r'filename="(?P<filename>.*?)";')
# Regex to parse HTTP Range header
_RANGE_RE = re.compile(r"\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*", re.IGNORECASE)


class HTTPStatusCode(IntEnum):
    REQUEST_TIMEOUT = 408
    TOO_MANY_REQUESTS = 429
    INTERNAL_SERVER_ERROR = 500
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504

    @classmethod
    def get_retryable_codes(cls) -> tuple[int, ...]:
        """Return all status code values as a tuple of integers."""
        return tuple(code.value for code in cls)

    @classmethod
    def is_retryable(cls, status_code: int) -> bool:
        """O(1) check if a given status code exists in this enum."""
        return status_code in cls._value2member_map_

class RateLimitInfo:
    def __init__(self) -> None:
        self.resource_type: str | None = None
        self.remaining: int | None = None
        self.reset_in_seconds: int | None = None
        self.limit: int | None = None
        self.window_seconds: int | None = None

    def add_policy(self, policy_value: str) -> None:
        if match := _RATELIMIT_POLICY_RE.search(policy_value):
            self.limit = int(match.group("q"))
            self.window_seconds = int(match.group("w"))

    def add_ratelimit(self, ratelimit_value: str) -> None:
        if match := _RATELIMIT_RE.search(ratelimit_value):
            self.resource_type = match.group("resource_type")
            self.remaining = int(match.group("r"))
            self.reset_in_seconds = int(match.group("t"))


class HTTPHeader:
    def __init__(self, headers: dict[str, str]) -> None:
        self._headers = {k.lower(): v for k, v in headers.items()}
        self.ratelimit_info: RateLimitInfo = RateLimitInfo()
        self.retry_after: int | None = None
        self.total_file_size: int | None = None
        self.filename: str | None = None

        self._parse_headers()

    def _parse_file_length(self):
        if self._headers.get("Content-Encoding", "identity").lower() != "identity":
            return # gzip/br/deflate/zstd etc

        if content_range := self._headers.get("Content-Range"):
            self.total_file_size = int(content_range.rsplit("/")[-1])
        elif content_length := self._headers.get("Content-Length"):
            self.total_file_size = int(content_length)

    def _parse_headers(self):
        if "retry-after" in self._headers:
            val = self._headers["retry-after"].strip()
            if val.isdigit():
                self.retry_after = int(val)

        if "ratelimit" in self._headers:
            self.ratelimit_info.add_ratelimit(self._headers["ratelimit"])

        if "ratelimit-policy" in self._headers:
            self.ratelimit_info.add_policy(self._headers["ratelimit-policy"])

        if "content-disposition" in self._headers and (
            match := _HEADER_FILENAME_RE.search(self._headers["content-disposition"]
        )):
            self.filename = match.group("filename").strip()
        
        self._parse_file_length()
