import requests
import atexit
import os
import logging
from contextlib import ExitStack, contextmanager
from urllib.parse import urljoin, urlparse
from enum import IntEnum
from dataclasses import dataclass
import re

logger = logging.getLogger("iantirta.remote.http")


_session: requests.Session | None = None


def get_http_session() -> requests.Session:
    global _session

    if _session is None:
        _session = requests.Session()

    return _session


def close_http_session() -> None:
    global _session
    sess = _session

    _session = None

    if sess is not None:
        try:
            sess.close()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error closing client: {e}")


atexit.register(close_http_session)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=close_http_session)


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


# Regex patterns for parsing rate limit headers
# e.g.: "api";r=0;t=55 --> resource_type="api", r=0, t=55
_RATELIMIT_REGEX = re.compile(r"\"(?P<resource_type>\w+)\"\s*;\s*r\s*=\s*(?P<r>\d+)\s*;\s*t\s*=\s*(?P<t>\d+)")
# e.g.: "fixed window";"api";q=500;w=300 --> q=500, w=300
_RATELIMIT_POLICY_REGEX = re.compile(r"q\s*=\s*(?P<q>\d+).*?w\s*=\s*(?P<w>\d+)")


@dataclass(slots=True, frozen=True)
class HTTPHeader:
    pass

class HTTPMixin:
    endpoint: str
    
    def __init__(
        self,
        endpoint: str | None = None,
    ):
        if endpoint is not None:
            self.endpoint = endpoint

        self.session = get_http_session()

    def _parse_ratelimit(
        self,
        headers: dict[str, str]
    ):
        ratelimit: str | None = None
        policy: str | None = None
        for key in headers:
            lower_key = key.lower()
            if lower_key == "ratelimit":
                ratelimit = headers[key]
            elif lower_key == "ratelimit-policy":
                policy = headers[key]

        if not ratelimit:
            return None
    
        match = _RATELIMIT_REGEX.search(ratelimit)
        if not match:
            return None
    
        resource_type = match.group("resource_type")
        remaining = int(match.group("r"))
        reset_in_seconds = int(match.group("t"))
    
        limit: int | None = None
        window_seconds: int | None = None
    
        if policy:
            policy_match = _RATELIMIT_POLICY_REGEX.search(policy)
            if policy_match:
                limit = int(policy_match.group("q"))
                window_seconds = int(policy_match.group("w"))

    def parse_headers(
        self,
        headers: dict[str, str],
    ):
        return HTTPHeader(
            ratelimit=self._parse_ratelimit(headers)
        )

    def _inner_request(
        self,
        method: str,
        url: str,
        *,
        stream: bool = False,
        params: dict[str, str] | None = None,
        max_attempt: int = 5,
        **kwargs
    ):
        method = method.upper()

        attempt = 0
        while True:
            attempt += 1
            try:
                def _should_retry():
                    if res.status_code not in HTTPStatusCode.get_retryable_codes():
                        return False
                    
                    if attempt > max_attempt:
                        res.raise_for_status()
                        return False

                    header_metadata = self.parse_headers(res.headers)
                    if (
                        res.status_code == 429
                        and (
                            ratelimit_info := parse_ratelimit_headers(res.headers)
                        ) is not None
                        and ratelimit_info.remaining == 0
                    ):
                        ratelimit_reset = ratelimit_info.reset_in_seconds
                    elif (retry_after := _parse_retry_after(res.headers)) is not None:
                        ratelimit_reset = retry_after
    
                    return True  # Should retry
                    
                res = self.session.request(
                    method,
                    url,
                    params=params,
                    stream=stream,
                    **kwargs
                )
                yield res
                return
            except Exception:
                raise

    def raise_for_status(
        self,
        res: requests.Response
    ):
        pass

    @staticmethod
    def _is_same_host(url: str, target: str) -> bool:
        target_host = (
            urlparse(target).hostname or ""
        ).lower()
        return (
            target_host == (
                urlparse(url).hostname or ""
            ).lower()
        )

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, str] | None = None,
        **kwargs
    ):
        return next(self._inner_request(
            method,
            url,
            params=params,
            **kwargs,
        ))

    def request_follow_redirect(
        self,
        method: str,
        url: str,
        *,
        max_redirect: int = 20,
    ):
        for _ in range(max_redirect):
            res = self.request(
                method,
                url,
                allow_redirects=False,
            )
            self.raise_for_status(res)
            
            if not res.is_redirect:
                return res

            target = urljoin(url, res.headers["Location"])
            if not self._is_same_host(url, target):
                return res
    
            url = target

        raise requests.TooManyRedirects(f"Exceeded {max_redirect} redirects.")


    def paginate(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None
    ):
        res = self.request(
            "get",
            url,
            params=params,
        )
        yield from res.json()

        while next_page := res.links.get("next", {}).get("url"):
            res = self.request(
                "get",
                next_page,
                params=params,
            )
            yield from res.json()

    @contextmanager
    def stream(
        self,
        method: str,
        url: str,
    ):
        yield from self._inner_request(
            method,
            url,
            stream=True,
        )

    def download(
        
    ):
        pass

if __name__ == "__main__":
    from pprint import pprint
    
    http = HTTPMixin()
    url = (
        "https://huggingface.co/api/models/"
        #"Qwen/Qwen3-ASR-1.7B-hf/"
        "facebook/mms-1b-all/"
        "tree/main"
    )
    pprint(http.request("get", url).links)
    for p in http.paginate(url):
        print(p)
    pprint(http.stream("get", url))
    pprint(http.request_follow_redirect("get", url).headers)
    # with ExitStack() as stack:
    #     res = stack.enter_context(
    #         http.stream("get", url)
    #     )
    #     pprint(res.headers)
    #     for chunk in res.iter_content(chunk_size=10*1024*1024):
    #         pprint(chunk)