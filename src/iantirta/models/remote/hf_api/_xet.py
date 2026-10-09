from functools import lru_cache
import re
import threading
from typing import Any
from .types import XetFileData
from pathlib import Path

_XET_HASH_RE = re.compile(r"[0-9a-f]{64}")

@lru_cache
def available() -> bool:
    try:
        import hf_xet  # noqa: F401
    except ImportError:
        return False

    return True



class XetSessionHolder:
    """Holds an optional XetSession; supports safe re-creation after sigint_abort or fork.

    Thread-safe: a ``threading.Lock`` guards all state mutations, which matters
    for free-threaded Python (3.14t) where multiple threads can race on ``get()``
    or ``sigint_abort()`` without the GIL serialising them.
    """

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


_xet_session: XetSessionHolder = XetSessionHolder()

def get_xet_session():
    """Return the global :class:`hf_xet.XetSession`, creating it on first call.

    The session is shared across all calls within a process, just as the HTTP
    client returned by :func:`~huggingface_hub.utils._http.get_session` is shared.
    It is created lazily and is fork-safe and thread-safe.
    """
    return _GLOBAL_XET_HOLDER.get()


def abort_xet_session():
    """Abort the global xet session after a KeyboardInterrupt.

    Cancels any in-flight Rust operation and clears the session so the next
    call to :func:`get_xet_session` starts fresh (notebook-friendly).
    """
    _GLOBAL_XET_HOLDER.sigint_abort()


class HFXetAPI:
    def __init__(self):
        super().__init__()
        self.xet_session = get_xet_session()

    def _make_xet_headers_without_auth(self, headers: dict[str, str]) -> dict[str, str]:
        """Return a copy of headers with the authorization header removed.

        Xet storage requests use a short-lived xet access token for auth, so the
        Hub authorization header must not be forwarded to xet storage endpoints.
        """
        return {key: value for key, value in headers.items() if key.lower() != "authorization"}

    def xet_download(
        self,
        *,
        incomplete_path: Path,
        xet_file_data: XetFileData,
        headers: dict[str, str],
        expected_size: int | None = None,
        displayed_filename: str | None = None,
        tqdm_class: type | None = None,
        _tqdm_bar = None,        
    ) -> None:
        """
        Download a file using Xet storage service.
        """
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

        from .utils._xet import abort_xet_session, get_xet_session, refresh_xet_connection_info, xet_headers_without_auth
        from .utils._xet_progress_reporting import XetDownloadProgressReporter

        xet_headers = self._make_xet_headers_without_auth(headers)

        # Fetched once per repo revision and cached; otherwise each download group would request its own
        # token, i.e. one Hub API call per file (rate-limited on large snapshot downloads, see #4722).
        connection_info = refresh_xet_connection_info(file_data=xet_file_data, headers=headers)

        session = get_xet_session()

        with XetDownloadProgressReporter(
            reconstruction_desc=f"{displayed_filename}: reconstructing file",
            transfer_desc=f"{displayed_filename}: downloading bytes",
            total=expected_size,
            log_level=logger.getEffectiveLevel(),
            name="huggingface_hub.xet_get",
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