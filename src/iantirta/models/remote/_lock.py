


import contextlib
import logging
import os
import time
from collections.abc import Generator
from pathlib import Path


logger = logging.getLogger(__name__)


class FileLockTimeout(TimeoutError):
    pass


@contextlib.contextmanager
def file_lock(
    path: str | Path,
    *,
    timeout: float | None = None,
    poll_interval: float = 0.1,
) -> Generator[None, None, None]:
    """Acquire a cross-process lock using a lock-file.

    The lock is intentionally simple:
    creating the file atomically means only one process wins.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    fd: int | None = None
    flags = os.O_CREAT | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    while True:
        try:
            fd = os.open(path, flags, 0o666)
            break
        except FileExistsError:
            if timeout is not None and time.monotonic() - start >= timeout:
                raise FileLockTimeout(f"Timed out waiting for lock: {path}")

            time.sleep(poll_interval)

    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)

        try:
            path.unlink()
        except FileNotFoundError:
            pass