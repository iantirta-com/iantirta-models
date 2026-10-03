
import logging
import os
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ._http import request
from ._types import XetFileData

logger = logging.getLogger(__name__)

@lru_cache
def available() -> bool:
    try:
        import hf_xet  # noqa: F401
    except ImportError:
        return False

    return True


def parse_file_data(res):
    if not available():
        return None

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
    
    return XetFileData(
        file_hash=file_hash,
        refresh_route=refresh_route,
    )


# Main

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


GLOBAL_XET_SESS = XetSessionHolder()
GLOBAL_XET_LOCK = threading.Lock()
XET_LOCKS: dict[str, threading.Lock] = {}
XET_INFO_CACHE: dict[str, "XetConnectionInfo"] = {}
MAX_XET_INFO_CACHE_SIZE = 1_000


def get_xet_session():
    """Return the global :class:`hf_xet.XetSession`, creating it on first call.

    The session is shared across all calls within a process, just as the HTTP
    client returned by :func:`~huggingface_hub.utils._http.get_session` is shared.
    It is created lazily and is fork-safe and thread-safe.
    """
    return GLOBAL_XET_SESS.get()

def abort_xet_session():
    return GLOBAL_XET_SESS.sigint_abort()

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

        res = request("GET", xet_fd.refresh_route, headers=headers)
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


def download(
    xet_file_data: XetFileData,
    incomplete_path: Path,
    headers: dict[str, str],
    expected_size: int | None = None,
    displayed_filename: str | None = None,
    tqdm_class: type | None = None,
    _tqdm_bar = None,
) -> None:
    try:
        from hf_xet import XetFileInfo  # type: ignore[no-redef]
    except ImportError:
        raise ValueError(
            "To use optimized download using Xet storage, you need to install the hf_xet package. "
            'Try `pip install "huggingface_hub[hf_xet]"` or `pip install hf_xet`.'
        )

    if not displayed_filename:
        displayed_filename = incomplete_path.name

    # Truncate filename if too long to display
    if len(displayed_filename) > 40:
        displayed_filename = f"{displayed_filename[:40]}(…)"

    xet_headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}

    connection_info = get_xet_connection_info(xet_file_data)
    session = get_xet_session()

    with XetDownloadProgressReporter(
        reconstruction_desc=f"{displayed_filename}: reconstructing file",
        transfer_desc=f"{displayed_filename}: downloading bytes",
        total=expected_size,
        log_level=logger.getEffectiveLevel(),
        name="iantirta.models.xet_download",
        tqdm_class=tqdm_class,
        external_reconstruction_bar=_tqdm_bar,
    ) as progress:
        try:
            with session.new_file_download_group(
                endpoint=connection_info.endpoint,
                token=connection_info.access_token,
                token_expiry_unix_secs=connection_info.expiration_unix_epoch,
                token_refresh_url=xet_file_data.refresh_route,
                token_refresh_headers=headers,
                custom_headers=xet_headers,
                progress_callback=progress.update_progress,
            ) as group:
                group.start_download_file(
                    XetFileInfo(xet_file_data.file_hash, expected_size), str(incomplete_path.absolute())
                )
        except KeyboardInterrupt:
            abort_xet_session()
            raise
