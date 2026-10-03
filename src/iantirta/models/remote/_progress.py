
import logging
import os
import sys
from contextlib import AbstractContextManager, nullcontext


class _FallbackTqdm:
    def __init__(self, *args, **kwargs):
        self.n = 0

    def update(self, n: int, *args, **kwargs):
        self.n += n
        sys.stderr.write(f"\r[ ]: {self.n}")
        sys.stderr.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return

try:
    from tqdm import tqdm
except ImportError:
    tqdm = _FallbackTqdm


def is_tqdm_disabled(log_level: int) -> bool | None:
    """
    Determine if tqdm progress bars should be disabled based on logging level and environment settings.

    see https://github.com/huggingface/huggingface_hub/pull/2000 and https://github.com/huggingface/huggingface_hub/pull/2698.
    """
    if log_level == logging.NOTSET:
        return True
    if os.getenv("TQDM_POSITION") == "-1":
        return False
    return None


def _create_progress_bar(
    *,
    cls: type[tqdm],
    log_level: int,
    name: str | None = None,
    **kwargs
) -> tqdm:
    """Create a progress bar.
    """
    # issubclass() crashes on non-class callables (e.g. functools.partial), guard with isinstance.
    if not (isinstance(cls, type) and issubclass(cls, tqdm)):
        return cls(**kwargs)  # type: ignore[return-value]

    # HF subclass: keep the historical log-level / TTY behavior. Group-based
    # disabling is already handled in `tqdm.__init__`.
    disable = is_tqdm_disabled(log_level)
    return cls(disable=disable, **kwargs)  # type: ignore[return-value]


def get_context_progressbar(
    *,
    desc: str,
    log_level: int,
    total: int | None = None,
    initial: int = 0,
    unit: str = "B",
    unit_scale: bool = True,
    name: str | None = None,
    tqdm_class: type[tqdm] | None = None,
    _tqdm_bar: tqdm | None = None,
) -> AbstractContextManager[tqdm]:
    if _tqdm_bar is not None:
        return nullcontext(_tqdm_bar)
        # ^ `contextlib.nullcontext` mimics a context manager that does nothing
        #   Makes it easier to use the same code path for both cases but in the later
        #   case, the progress bar is not closed when exiting the context manager.
    
    return _create_progress_bar(  # type: ignore
        cls=tqdm_class or tqdm,
        log_level=log_level,
        name=name,
        unit=unit,
        unit_scale=unit_scale,
        total=total,
        initial=initial,
        desc=desc,
    )
