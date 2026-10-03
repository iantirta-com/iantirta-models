
import threading
import requests
from pathlib import Path
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextlib import ExitStack
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from typing import Any
import time
import os


SESS_LOCK = threading.Lock()
GLOBAL_SESS = None


class XetSessionHolder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: Any = None
        self._session_pid: int | None = None

    def get(self):
        """Return the current session, creating one if needed.

        Fork-safe: if the current process PID differs from the PID that created
        the session (i.e. we are in a forked child), the old session is discarded
        and a fresh session is created for this process.
        """
        with self._lock:
            current_pid = os.getpid()
    
            if self._session is not None and self._session_pid != current_pid:
                # Fork detected. Discard the parent's session; the Rust Drop will
                # call discard_runtime() (std::mem::forget) rather than the normal
                # shutdown path, so this returns immediately without blocking.
                self._session = None

            if self._session is None:
                from hf_xet import XetSession

                self._session = XetSession()
                self._session_pid = current_pid

            return self._session

    def sigint_abort(self):
        """Abort the current session and clear it so the next get() creates a fresh one."""
        with self._lock:
            if self._session is not None:
                try:
                    self._session.sigint_abort()
                except Exception:
                    pass
                self._session = None
                self._session_pid = None

GLOBAL_XET_SESS = XetSessionHolder()
GLOBAL_XET_LOCK = threading.Lock()
XET_LOCKS: dict[str, threading.Lock] = {}
XET_INFO_CACHE: dict[str, "XetConnectionInfo"] = {}
MAX_XET_INFO_CACHE_SIZE = 1_000

def get_sessions():
    global GLOBAL_SESS
    if GLOBAL_SESS is None:
        with SESS_LOCK:
            if GLOBAL_SESS is None:
                GLOBAL_SESS = requests.Session()
    return GLOBAL_SESS


def get_xet_session():
    """Return the global :class:`hf_xet.XetSession`, creating it on first call.

    The session is shared across all calls within a process, just as the HTTP
    client returned by :func:`~huggingface_hub.utils._http.get_session` is shared.
    It is created lazily and is fork-safe and thread-safe.
    """
    return GLOBAL_XET_SESS.get()


def http_request(
    method: str,
    url: str,
    *,
    timeout=30,
    max_redirects=20,
    stream: bool = False,
    allow_redirects=False,
    **kwargs,
) -> requests.Response:
    session = get_sessions()
    current_method = method.upper()

    for _ in range(max_redirects):
        res = session.request(
            current_method,
            url,
            allow_redirects=allow_redirects,
            timeout=timeout,
            stream=stream,
            **kwargs,
        )

        if not res.is_redirect or "Location" not in res.headers:
            return res

        target = urljoin(url, res.headers["Location"])
        if urlparse(url).hostname.lower() != urlparse(target).hostname.lower():
            return res
        res.close()
        
        if res.status_code in (301, 302, 303) and current_method != "HEAD":
            current_method = "GET"

        url = target

    raise requests.TooManyRedirects(f"Exceeded {max_redirects} redirects.")


@contextmanager
def http_stream_request(
    method: str,
    url: str,
    **kwargs
):
    """Context manager yielding a streamed response that safely closes on exit."""
    while True:
        try:
            res = http_request(method, url, stream=True, allow_redirects=True, **kwargs)
            if not _should_retry(res):
                yield res
                return
        finally:
            res.close()


@dataclass(frozen=True, slots=True)
class XetFileData:
    file_hash: str
    refresh_route: str


@dataclass(frozen=True, slots=True)
class XetConnectionInfo:
    access_token: str
    expiration_unix_epoch: int
    endpoint: str

    @property
    def expired(self) -> bool:
        return (
            self.expiration_unix_epoch
            <= (
                int(time.time())
                + 60
            )
        )


def _parse_xet_fd(res: requests.Response):
    if res is None:
        return None
    try:
        file_hash = res.headers["X-Xet-Hash"]

        if "xet-auth" in res.links:
            refresh_route = res.links["xet-auth"]["url"]
        else:
            refresh_route = res.headers["X-Xet-Refresh-Route"]
    except KeyError:
        return None
    # endpoint = endpoint if endpoint is not None else constants.ENDPOINT
    # if refresh_route.startswith(constants.HUGGINGFACE_CO_URL_HOME):
    #     refresh_route = refresh_route.replace(constants.HUGGINGFACE_CO_URL_HOME.rstrip("/"), endpoint.rstrip("/"))
    return XetFileData(
        file_hash=file_hash,
        refresh_route=refresh_route,
    )


def _parse_xet_connection_info(headers):
    try:
        endpoint = headers["X-Xet-Cas-Url"]
        access_token = headers["X-Xet-Access-Token"]
        expiration_unix_epoch = int(headers["X-Xet-Token-Expiration"])
    except (KeyError, ValueError, TypeError):
        return None

    return XetConnectionInfo(
        endpoint=endpoint,
        access_token=access_token,
        expiration_unix_epoch=expiration_unix_epoch,
    )





def _cache_key(url: str, headers: dict[str, str]) -> str:
    """Return a unique cache key for the given request parameters."""
    lower_headers = {k.lower(): v for k, v in headers.items()}  # casing is not guaranteed here
    auth_header = lower_headers.get("authorization", "")
    return f"{url}|{auth_header}"


def get_xet_connection_info(
    xet_fd: XetFileData,
    headers: dict[str, str] = {},
):
    cache_key = _cache_key(xet_fd.refresh_route, headers)
    cached_info = XET_INFO_CACHE.get(cache_key)
    if cached_info is not None and not cached_info.expired:
        return cached_info

    with GLOBAL_XET_LOCK:
        if cache_key not in XET_LOCKS:
            XET_LOCKS[cache_key] = threading.Lock()
        key_lock = XET_LOCKS[cache_key]

    with key_lock:
        cached_info = XET_INFO_CACHE.get(cache_key)
        if cached_info is not None and not cached_info.expired:
            return cached_info

        res = http_request("GET", xet_fd.refresh_route, headers=headers)
        res.raise_for_status()

        metadata = _parse_xet_connection_info(res.headers)
        if metadata is None:
            raise ValueError("Xet headers have not been correctly set by the server.")

        with GLOBAL_XET_LOCK:
            # Purge expired entries
            expired_keys = [k for k, v in XET_INFO_CACHE.items() if v.expired]
            for k in expired_keys:
                XET_INFO_CACHE.pop(k, None)

            # Evict LRU/First item if cache limit is exceeded
            if len(XET_INFO_CACHE) >= MAX_XET_INFO_CACHE_SIZE:
                XET_INFO_CACHE.pop(next(iter(XET_INFO_CACHE)))

            XET_INFO_CACHE[cache_key] = metadata

        return metadata


def _test_progress(group_report, _item_reports: dict | None = None):
    bytes_inc = max(0, group_report.total_bytes_completed)
    transfer_inc = max(0, group_report.total_transfer_bytes_completed)
    print(bytes_inc, transfer_inc)
    print(_item_reports)

def xet_download(xet_fd, headers={}):
    from hf_xet import XetFileInfo
    
    connection_info = get_xet_connection_info(xet_fd)
    session = get_xet_session()
    incomplete_path = Path("testxet.incomplete")
    try:
        with session.new_file_download_group(
            endpoint=connection_info.endpoint,
            token=connection_info.access_token,
            token_expiry_unix_secs=connection_info.expiration_unix_epoch,
            token_refresh_url=xet_fd.refresh_route,
            token_refresh_headers=headers,
            progress_callback=_test_progress
        ) as group:
            group.start_download_file(
                XetFileInfo(
                    xet_fd.file_hash,
                    #expected_size
                ),
                str(incomplete_path.absolute())
            )
    except KeyboardInterrupt:
        GLOBAL_XET_SESS.sigint_abort()
        raise


def http_download(
    url,
    headers={},
):
    incomplete_path = Path("testhttp.incomplete")
    with ExitStack() as stack:
        try:
            res = stack.enter_context(
                http_stream_request(
                    "GET",
                    url,
                    headers=headers,
                )
            )
            res.raise_for_status()
            if res.encoding is None:
                res.encoding = 'utf-8'
            for line in res.iter_lines(decode_unicode=True):
                print(">>", line)
                #outfile.write(line)
        except Exception:
            raise
        finally:
            print(res.headers)

def _should_retry(res):
    return False


if __name__ == "__main__":
    from rich import inspect
    url = "https://huggingface.co/adefossez/Demucs-mdx_extra_q/resolve/main/83fc094f.safetensors"
    metadata = get_hf_file_metadata(url)
    #print(xet_download(metadata.xet_fd))
    print(http_download(url))
    # print(
    #     http_get(
    #         "https://huggingface.co/adefossez/Demucs-mdx_extra_q/resolve/main/83fc094f.safetensors",
    #         Path("test.txt"),
    #     )
    # )
    #res = http_stream_request("HEAD", url)
    # 