

import atexit
import copy
import logging
import os
import re
import time
import uuid
from collections.abc import Generator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, BinaryIO, cast
from urllib.parse import urljoin, urlparse

import requests

from . import _config, _progress

logger = logging.getLogger(__name__)


MAX_HTTP_DOWNLOAD_SIZE = 50 * 1000 * 1000 * 1000  # 50 GB
DOWNLOAD_CHUNK_SIZE = 10 * 1024 * 1024
RETRY_STATUS_CODES = frozenset({
    408,
    429,
    500,
    502,
    503,
    504,
})


_session: requests.Session | None = None


def hf_request_event_hook(req: requests.Request):
    if "X-Amzn-Trace-Id" not in req.headers:
        req.headers["X-Amzn-Trace-Id"] = req.headers.get("x-request-id") or str(uuid.uuid4())
    request_id = req.headers.get("X-Amzn-Trace-Id")
    logger.debug(
        "Request %s: %s %s (authenticated: %s)",
        request_id,
        req.method,
        req.url,
        req.headers.get("authorization") is not None,
    )
    return request_id


def get_session() -> requests.Session:
    global _session

    if _session is None:
        _session = requests.Session()

    return _session


def close_session() -> None:
    global _session
    sess = _session

    # First, set global client to None
    _session = None

    # Then, close the clients
    if sess is not None:
        try:
            sess.close()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error closing client: {e}")


atexit.register(close_session)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=close_session)


# Main
_DEFAULT_RETRY_ON_EXCEPTIONS: tuple[type[Exception], ...] = (
    requests.exceptions.ConnectTimeout,
    requests.exceptions.ConnectionError,
)
_DEFAULT_RETRY_ON_STATUS_CODES: tuple[int, ...] = (408, 429, 500, 502, 503, 504)

# Regex patterns for parsing rate limit headers
# e.g.: "api";r=0;t=55 --> resource_type="api", r=0, t=55
_RATELIMIT_REGEX = re.compile(r"\"(?P<resource_type>\w+)\"\s*;\s*r\s*=\s*(?P<r>\d+)\s*;\s*t\s*=\s*(?P<t>\d+)")

# e.g.: "fixed window";"api";q=500;w=300 --> q=500, w=300
_RATELIMIT_POLICY_REGEX = re.compile(r"q\s*=\s*(?P<q>\d+).*?w\s*=\s*(?P<w>\d+)")
HEADER_FILENAME_PATTERN = re.compile(r'filename="(?P<filename>.*?)";')
# Regex to parse HTTP Range header
RANGE_REGEX = re.compile(r"\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*", re.IGNORECASE)


@dataclass(frozen=True)
class RateLimitInfo:
    """
    Parsed rate limit information from HTTP response headers.

    Attributes:
        resource_type (`str`): The type of resource being rate limited.
        remaining (`int`): The number of requests remaining in the current window.
        reset_in_seconds (`int`): The number of seconds until the rate limit resets.
        limit (`int`, *optional*): The maximum number of requests allowed in the current window.
        window_seconds (`int`, *optional*): The number of seconds in the current window.

    """

    resource_type: str
    remaining: int
    reset_in_seconds: int
    limit: int | None = None
    window_seconds: int | None = None


def _parse_ratelimit(headers: Mapping[str, str]) -> RateLimitInfo | None:
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

    return RateLimitInfo(
        resource_type=resource_type,
        remaining=remaining,
        reset_in_seconds=reset_in_seconds,
        limit=limit,
        window_seconds=window_seconds,
    )


def _parse_retry_after(headers: Mapping[str, str]) -> int | None:
    value: str | None = None
    for key in headers:
        if key.lower() == "retry-after":
            value = headers[key]
            break

    if value is None:
        return None
    value = value.strip()
    if not value:
        return None

    if value.isdigit():
        return int(value)  # e.g. "Retry-After: 120"
    return None  #  e.g. "Retry-After: Wed, 21 Oct 2015 07:28:00 GMT" - not supported


def _parse_total(headers: Mapping[str, str]) -> int | None:
    # If HTTP response contains compressed body (e.g. gzip), the `Content-Length` header will
    # contain the length of the compressed body, not the uncompressed file size.
    # And at the start of transmission there's no way to know the uncompressed file size for gzip,
    # thus we return None in that case.
    content_encoding = headers.get("Content-Encoding", "identity").lower()
    if content_encoding != "identity":
        # gzip/br/deflate/zstd etc
        return None

    content_range = headers.get("Content-Range")
    if content_range is not None:
        try:
            return int(content_range.rsplit("/", 1)[1])
        except (IndexError, ValueError):
            return None

    content_length = headers.get("Content-Length")
    if content_length is not None:
        try:
            return int(content_length)
        except ValueError:
            return None

    return None


def _adjust_range_header(original_range: str | None, resume_size: int) -> str | None:
    """
    Adjust HTTP Range header to account for resume position.
    """
    if not original_range:
        return f"bytes={resume_size}-"

    if "," in original_range:
        raise ValueError(f"Multiple ranges detected - {original_range!r}, not supported yet.")

    match = RANGE_REGEX.fullmatch(original_range)
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


def _is_same_or_hub_host(url: str, target: str) -> bool:
    """Whether `target` is served by the same host as `url`, or by a known Hub host."""
    target_host = (urlparse(target).hostname or "").lower()
    return target_host == (urlparse(url).hostname or "").lower() or target_host in _config.HF_URL_HOSTS


def _http_request(
    method: str,
    url: str,
    *,
    max_retries: int = 5,
    base_wait_time: float = 1,
    max_wait_time: float = 8,
    retry_on_exceptions: type[Exception] | tuple[type[Exception], ...] = _DEFAULT_RETRY_ON_EXCEPTIONS,
    retry_on_status_codes: int | tuple[int, ...] = _DEFAULT_RETRY_ON_STATUS_CODES,
    stream: bool = False,
    **kwargs,
) -> Generator:
    method = method.upper()
    
    if isinstance(retry_on_exceptions, type):  # Tuple from single exception type
        retry_on_exceptions = (retry_on_exceptions,)

    if isinstance(retry_on_status_codes, int):  # Tuple from single status code
        retry_on_status_codes = (retry_on_status_codes,)

    attempt = 0
    sleep_time = base_wait_time
    ratelimit_reset: int | None = None  # seconds to wait for rate limit reset if 429 response

    while True:
        attempt += 1
        ratelimit_reset = None
        res: requests.Response | None = None
        session = get_session()
        try:

            def _should_retry(res: requests.Response) -> bool:
                nonlocal ratelimit_reset

                if res.status_code not in retry_on_status_codes:
                    return False  # Success, don't retry

                # Wrong status code returned (HTTP 503 for instance)
                logger.warning(f"HTTP Error {res.status_code} thrown while requesting {method} {url}")
                if attempt > max_retries:
                    res.raise_for_status()  # Will raise uncaught exception
                    # Return/yield response to avoid infinite loop in the corner case where the
                    # user ask for retry on a status code that doesn't raise_for_status.
                    return False  # Don't retry, return/yield response

                # Check 'ratelimit' and `Retry-After` headers.
                if (
                    res.status_code == 429
                    and (ratelimit_info := _parse_ratelimit(res.headers)) is not None
                    and ratelimit_info.remaining == 0
                ):
                    ratelimit_reset = ratelimit_info.reset_in_seconds
                elif (retry_after := _parse_retry_after(res.headers)) is not None:
                    ratelimit_reset = retry_after

                return True  # Should retry

            res = session.request(
                method,
                url,
                stream=stream,
                # hooks=[hf_request_event_hook],
                **kwargs,
            )
            if not _should_retry(res):
                yield res
                return

        except retry_on_exceptions as err:
            logger.warning(f"'{err}' thrown while requesting {method} {url}")
            if isinstance(err, requests.exceptions.SSLError):
                close_session()
            if attempt > max_retries:
                raise

        finally:
            if res is not None and hasattr(res, "close") and getattr(res, "raw", None) is not None:
                res.close()
            
        if ratelimit_reset is not None:
            actual_sleep = float(ratelimit_reset) + 1  # +1s to avoid rounding issues
            logger.warning(f"Rate limited. Waiting {actual_sleep}s before retry [Retry {attempt}/{max_retries}].")
        else:
            actual_sleep = sleep_time
            logger.warning(f"Retrying in {actual_sleep}s [Retry {attempt}/{max_retries}].")
        
        time.sleep(actual_sleep)
        # Update sleep time for next retry
        sleep_time = min(max_wait_time, sleep_time * 2)  # Exponential backoff


def request(
    method: str,
    url: str,
    *,
    stream: bool = False,
    **kwargs,
) -> requests.Response:
    return next(
        _http_request(
            method=method,
            url=url,
            stream=stream,
            **kwargs,
        )
    )


def request_follow_redirect(
    method: str,
    url: str,
    *,
    max_redirects=20,
    **kwargs,
) -> requests.Response:
    for _ in range(max_redirects):
        res = request(
            method=method,
            url=url,
            **kwargs,
            allow_redirects=False,
        )
        res.raise_for_status()

        if not res.is_redirect:
            return res

        target = urljoin(url, res.headers["Location"])
        if not _is_same_or_hub_host(url, target):
            return res

        url = target

    raise requests.TooManyRedirects(f"Exceeded {max_redirects} redirects.")


@contextmanager
def stream(
    method: str,
    url: str,
    **kwargs,
) -> Generator[requests.Response, None, None]:
    yield from _http_request(
        method=method,
        url=url,
        stream=True,
        **kwargs,
    )


def http_download(
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
) -> None:
    """Download an HTTP resource to a file."""
    if expected_size is not None and resume_size == expected_size:
        # If the file is already fully downloaded, we don't need to download it again.
        return

    initial_headers = headers
    headers = copy.deepcopy(headers) or {}
    if resume_size > 0:
        headers["Range"] = _adjust_range_header(headers.get("Range"), resume_size)

    elif expected_size and expected_size > MAX_HTTP_DOWNLOAD_SIZE:
        # Any files over 50GB will not be available through basic http requests.
        raise ValueError(
            "The file is too large to be downloaded using the regular download method. "
            " Install `hf_xet` with `pip install hf_xet` for xet-powered downloads."
        )

    with ExitStack() as stack:
        downloaded = resume_size
        try:
            res = stack.enter_context(
                stream(
                    method="GET",
                    url=url,
                    headers=headers,
                    retry_on_exceptions=(),
                    retry_on_status_codes=(408, 429),
                )
            )
            res.raise_for_status()
            # If we requested a Range but got 200 back, the server ignored our Range header
            # (e.g. CloudFront with Accept-Encoding: gzip). Reset file to avoid corruption.
            if resume_size > 0 and res.status_code == 200:
                temp_file.seek(max(temp_file.tell() - resume_size, 0))
                temp_file.truncate()
                if _tqdm_bar is not None:
                    _tqdm_bar.update(-resume_size)
                resume_size = 0

            total: int | None = _parse_total(res.headers)
            if expected_size is None:
                expected_size = total
            elif total is None:
                # Hub serves compressible text files (e.g. vocab.json) with `Content-Encoding: gzip` and
                # `Transfer-Encoding: chunked`, so the response carries no `Content-Length`. Fall back to the caller's
                # `expected_size` (always known from the metadata HEAD on the hf_hub path) so the progress bar, and any
                # aggregating wrapper such as snapshot_download's `_AggregatedTqdm` — still sees the file size.
                total = expected_size

            if displayed_filename is None:
                displayed_filename = url
                content_disposition = res.headers.get("Content-Disposition")
                if content_disposition is not None:
                    match = HEADER_FILENAME_PATTERN.search(content_disposition)
                    if match is not None:
                        # Means file is on CDN
                        displayed_filename = match.groupdict()["filename"]

            # Truncate filename if too long to display
            if len(displayed_filename) > 40:
                displayed_filename = f"(…){displayed_filename[-40:]}"

            consistency_error_message = (
                f"Consistency check failed: file should be of size {expected_size} but has size"
                f" {{actual_size}} ({displayed_filename}).\nThis is usually due to network issues while downloading the file."
                " Please retry with `force_download=True`."
            )

            progress_cm = _progress.get_context_progressbar(
                desc=displayed_filename,
                log_level=logger.getEffectiveLevel(),
                total=total,
                initial=resume_size,
                name="iantirta.models.http_download",
                tqdm_class=cast(Any, tqdm_class),
                _tqdm_bar=cast(Any, _tqdm_bar),
            )

            progress = stack.enter_context(progress_cm)
            downloaded = resume_size

            if res.encoding is None:
                res.encoding = 'utf-8'

            for chunk in res.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    temp_file.write(chunk)
                    downloaded += len(chunk)
                    # Some data has been downloaded from the server so we reset the number of retries.
                    max_retries = 5
                
        except (requests.ConnectionError, requests.ConnectTimeout) as e:
            # Retry transient failures both when opening the stream and while reading its body.
            if max_retries <= 0:
                logger.warning("Error while downloading from %s: %s\nMax retries exceeded.", url, str(e))
                raise
            logger.warning("Error while downloading from %s: %s\nTrying to resume download...", url, str(e))
            time.sleep(1)
            return http_download(
                url=url,
                temp_file=temp_file,
                resume_size=downloaded,
                headers=initial_headers,
                expected_size=expected_size,
                max_retries=max_retries - 1,
                # Reuse the existing progress bar across retries so a custom `tqdm_class` (e.g. snapshot_download's `_AggregatedTqdm`,
                # which mutates a shared parent bar in `__init__`) is not re-instantiated and does not double-count `total`/`initial`.
                tqdm_class=tqdm_class,
                _tqdm_bar=progress,
            )

    # Compare against the bytes downloaded by this call rather than the absolute file position: the two differ
    # whenever the caller passed a file object that was not positioned at 0.
    if expected_size is not None and expected_size != downloaded:
        raise OSError(
            consistency_error_message.format(
                actual_size=downloaded,
            )
        )
