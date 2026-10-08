import atexit
import logging
import os
import time
from abc import abstractmethod
from collections.abc import Generator, Iterable
from contextlib import ExitStack, contextmanager
from typing import Any, BinaryIO, cast
from urllib.parse import urljoin, urlparse

import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout

from .. import _progress
from . import http_constant
from .types import _RANGE_RE, HTTPHeader, HTTPStatusCode

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


class HTTPMixin:
    endpoint: str
    
    def __init__(self, endpoint: str | None = None, local_logger=None):
        if endpoint is not None:
            self.endpoint = endpoint

        self.session = get_http_session()
        self.logger = local_logger or logger

    def _inner_request(
        self,
        method: str,
        url: str,
        *,
        max_attempt: int = 5,
        base_wait_time: int = 1,
        max_wait_time: int = 8,
        stream: bool = False,
        params: dict[str, str] | None = None,
        **kwargs
    ) -> Generator[requests.Response, None, None]:
        method = method.upper()

        sleep_time = base_wait_time
        ratelimit_reset = None
        attempt = 0
        while True:
            attempt += 1
            ratelimit_reset = None
            try:
                def _should_retry(response: requests.Response, _attempt: int):
                    nonlocal ratelimit_reset

                    if response.status_code not in HTTPStatusCode.get_retryable_codes():
                        return False
                    
                    if _attempt > max_attempt:
                        response.raise_for_status()
                        return False

                    header_metadata = HTTPHeader(response.headers)
                    if (response.status_code == 429
                        and header_metadata.ratelimit_info is not None
                        and header_metadata.ratelimit_info.remaining == 0
                    ): # RateLimit
                        ratelimit_reset = header_metadata.ratelimit_info.reset_in_seconds
                    elif header_metadata.retry_after is not None:
                        ratelimit_reset = header_metadata.retry_after
    
                    return True  # Should retry
                    
                res = self.session.request(
                    method,
                    url,
                    params=params,
                    stream=stream,
                    **kwargs
                )
                if not _should_retry(res, attempt):
                    yield res
                    return
            except Exception as e:

                if isinstance(e, (requests.exceptions.SSLError)):
                    close_http_session()

                if isinstance(e, (ConnectionError, Timeout, HTTPError)):
                    if attempt > max_attempt:
                        raise
                else:
                    raise

            actual_sleep = float(ratelimit_reset) + 1.0 if ratelimit_reset is not None else sleep_time
            time.sleep(actual_sleep)

            # Update sleep time for next retry
            sleep_time = min(max_wait_time, sleep_time * 2)  # Exponential backoff

    @abstractmethod
    def raise_for_status(self, res: requests.Response) -> None:
        """To Be Override by child"""
        res.raise_for_status()

    @staticmethod
    def _is_same_host(url: str, target: str) -> bool:
        target_host = (urlparse(target).hostname or "").lower()
        return (target_host == (urlparse(url).hostname or "").lower())

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, str] | None = None,
        **kwargs
    ) -> requests.Response:
        return next(self._inner_request(method, url, params=params, **kwargs,))

    def request_follow_redirect(
        self,
        method: str,
        url: str,
        *,
        max_redirect: int = 20,
    ) -> requests.Response:
        for _ in range(max_redirect):
            res = self.request(method, url, allow_redirects=False,)
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
    ) -> Iterable:
        res = self.request("get", url, params=params,)
        self.raise_for_status(res)
        yield from res.json()

        while next_page := res.links.get("next", {}).get("url"):
            res = self.request("get", next_page, params=params,)
            self.raise_for_status(res)
            yield from res.json()

    @contextmanager
    def stream(self, method: str, url: str, **kwargs) -> Generator[requests.Response, None, None]:
        yield from self._inner_request(method, url, stream=True, **kwargs)

    @staticmethod
    def _adjust_range_header(original_range: str | None, resume_size: int) -> str | None:
        if not original_range:
            return f"bytes={resume_size}-"

        if "," in original_range:
            raise ValueError(f"Multiple ranges detected - {original_range!r}, not supported yet.")

        match = _RANGE_RE.fullmatch(original_range)
        if not match:
            raise RuntimeError(f"Invalid range format - {original_range!r}.")
        start, end = match.groups()

        if not start:
            if not end:
                raise RuntimeError(f"Invalid range format - {original_range!r}.")

            new_suffix = int(end) - resume_size
            new_range = f"bytes=-{new_suffix}"
            if new_suffix <= 0:
                raise RuntimeError(f"Empty new range - {new_range!r}.")
            return new_range

        start = int(start)
        new_start = start + resume_size
        if end:
            end = int(end)
            new_range = f"bytes={new_start}-{end}"
            if new_start > end:
                raise RuntimeError(f"Empty new range - {new_range!r}.")
            return new_range

        return f"bytes={new_start}-"

    def download(
        self,
        url: str,
        temp_file: BinaryIO,
        *,
        resume_size: int = 0,
        expected_size: int | None = None,
        headers: dict[str, Any] | None = None,
        displayed_filename: str | None = None,
        max_retries: int = 5,
        tqdm_class: type | None = None,
        _tqdm_bar = None,
    ):
        if expected_size is not None and resume_size == expected_size:
            return # Already downloaded

        initial_headers = dict(headers or {})
        if resume_size > 0:
            headers["Range"] = self._adjust_range_header(headers.get("Range"), resume_size)
        elif expected_size and expected_size > http_constant.MAX_HTTP_DOWNLOAD_SIZE:
            raise ValueError(
                "The file is too large to be downloaded "
                "using HTTP regular download method."
            )
    
        with ExitStack() as stack:
            progress = _tqdm_bar
            downloaded = resume_size
            try:
                res: requests.Response = stack.enter_context(
                    self.stream(
                        method="get",
                        url=url,
                        headers=headers,
                    )
                )
                self.raise_for_status(res)

                if resume_size > 0 and res.status_code == 200:
                    temp_file.seek(max(temp_file.tell() - resume_size, 0))
                    temp_file.truncate()
                    if _tqdm_bar is not None:
                        _tqdm_bar.update(-resume_size)
                    resume_size = 0

                header_metadata = HTTPHeader(res.headers)
                total: int | None = header_metadata.total_file_size
                if expected_size is None:
                    expected_size = total
                elif total is None:
                    total = expected_size

                if displayed_filename is None:
                    displayed_filename = url
                    if header_metadata.filename is not None:
                        displayed_filename = header_metadata.filename

                if len(displayed_filename) > 40:
                    displayed_filename = f"(...){displayed_filename[-40:]}"

                consistency_error_message = (
                    f"Consistency check failed: file should be of size {expected_size} but has size"
                    f" {{actual_size}} ({displayed_filename}).\nThis is usually due to network issues while downloading the file."
                    " Please retry with `force_download=True`."
                )

                progress_cm = _progress.get_context_progressbar(
                    desc=displayed_filename,
                    log_level=self.logger.getEffectiveLevel(),
                    total=total,
                    initial=resume_size,
                    name="iantirta.models.http_download",
                    tqdm_class=cast(Any, tqdm_class),
                    _tqdm_bar=cast(Any, _tqdm_bar),
                )

                progress = stack.enter_context(progress_cm)
                downloaded = resume_size
                for chunk in res.iter_content(chunk_size=http_constant.DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        progress.update(len(chunk))
                        temp_file.write(chunk)
                        downloaded += len(chunk)
                        max_retries = 5 # Reset if success
            except Exception as e:
                if isinstance(e, (Timeout, ConnectionError, HTTPError)):
                    if max_retries <= 0:
                        self.logger.warning("Error while downloading from %s: %s\nMax retries exceeded.", url, str(e))
                        raise
                    time.sleep(1)
                    self.logger.warning("Error while downloading from %s: %s\nTrying to resume download...", url, str(e))
                    return self.download(
                        url=url,
                        temp_file=temp_file,
                        resume_size=downloaded,
                        headers=initial_headers,
                        expected_size=expected_size,
                        tqdm_class=tqdm_class,
                        _tqdm_bar=_tqdm_bar,
                        max_retries=max_retries - 1
                    )
                else:
                    raise
        if expected_size is not None and expected_size != downloaded:
            raise OSError(
                consistency_error_message.format(
                    actual_size=downloaded,
                )
            )

if __name__ == "__main__":
    from pprint import pprint
    
    http = HTTPMixin()
    url = (
        "https://huggingface.co/api/models/"
        #"Qwen/Qwen3-ASR-1.7B-hf/"
        "facebook/mms-1b-all/"
        "tree/main"
    )
    print("= Test Normal Request =")
    pprint(http.request("get", url).links)
    print("= Test Pagination = ")
    for p in http.paginate(url):
        pprint(p)
    print("= Test Follow Redirect =")
    pprint(http.request_follow_redirect("get", url).headers)
    print("= Test Download = ")
    with open("test.tmp", "wb") as f:
        http.download(url, f)
    # with ExitStack() as stack:
    #     res = stack.enter_context(
    #         http.stream("get", url)
    #     )
    #     pprint(res.headers)
    #     for chunk in res.iter_content(chunk_size=10*1024*1024):
    #         pprint(chunk)