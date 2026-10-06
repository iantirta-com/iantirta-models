
import logging
import os
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from typing_extensions import Self

from iantirta.models.tools._tqdm import tqdm


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


# Transfer byte count is hard to predict (dedup/compression), so we omit a total and show bytes only.
XET_TRANSFER_BAR_FORMAT = "{desc}: {bar}| {n_fmt:>5}B{postfix:>12}"
XET_BYTES_BAR_FORMAT = "{l_bar}{bar}| {n_fmt:>5}B / {total_fmt:>5}B{postfix:>12}"


def _format_speed_postfix(speed: float | None) -> str:
    s = tqdm.format_sizeof(speed) if speed is not None else "???"
    return f"{s}B/s  ".rjust(10, " ")


def _set_monotonic_total(bar, total: int | None) -> None:
    if total is None or not hasattr(bar, "total"):
        return
    bar.total = max(bar.total or 0, total)


def _update_transfer_bar(bar, inc: int) -> None:
    """Update the transfer bar and grow its hidden total so the bar graphic advances.

    Network bytes are hard to predict (dedup/compression), so the display omits a denominator.
    tqdm still needs an internal total for the bar width — seeded from file size when known,
    then expanded here if bytes received exceed that estimate.
    """
    n_after = getattr(bar, "n", 0) + inc
    current_total = getattr(bar, "total", 0) or 0
    if n_after > 0 and current_total < n_after:
        bar.total = max(current_total, int(n_after * 1.25) + 1)
    bar.update(inc)


class XetDownloadProgressReporter:
    """Dual progress bars for Xet downloads: network transfer and file reconstruction.

    ``total_transfer_bytes_completed`` tracks bytes received from the network (updated continuously).
    ``total_bytes_completed`` tracks bytes written to disk (updated after buffered chunks are flushed).
    Showing both bars gives responsive feedback on slow connections where reconstruction lags behind transfer.
    """
    def __init__(
        self,
        *,
        reconstruction_desc: str,
        transfer_desc: str = "Downloading bytes",
        total: int | None = None,
        log_level: int,
        name: str | None = None,
        tqdm_class: type | None = None,
        external_reconstruction_bar: Any | None = None,
        position: int = 0,
    ):
        self._prev_bytes_completed = 0
        self._prev_transfer_bytes_completed = 0

        cls = tqdm_class or tqdm
        routes_transfer_via_reconstruction = (
            external_reconstruction_bar is not None
            and callable(
                getattr(
                    external_reconstruction_bar,
                    "update_transfer",
                    None
                )
            )
        )
        uses_aggregated_tqdm_class = (
            external_reconstruction_bar is None
            and callable(
                getattr(cls, "update_transfer", None)
            )
        )

        if external_reconstruction_bar is not None:
            self.reconstruction_bar = external_reconstruction_bar
            self._owns_reconstruction_bar = False
        else:
            self.reconstruction_bar = _create_progress_bar(
                cls=cls,  # ty: ignore[invalid-argument-type]
                log_level=log_level,
                name=name,
                desc=reconstruction_desc,
                total=total,
                unit="B",
                unit_scale=True,
                position=position + 1,
                bar_format=XET_BYTES_BAR_FORMAT,
                leave=True,
            )
            self._owns_reconstruction_bar = True

        if routes_transfer_via_reconstruction or uses_aggregated_tqdm_class:
            self.transfer_bar = self.reconstruction_bar
            self._owns_transfer_bar = False
        elif external_reconstruction_bar is not None:
            self.transfer_bar = None
            self._owns_transfer_bar = False
        else:
            self.transfer_bar = _create_progress_bar(
                cls=cls,  # ty: ignore[invalid-argument-type]
                log_level=log_level,
                name=f"{name}.transfer" if name else None,
                desc=transfer_desc,
                total=total,
                unit="B",
                unit_scale=True,
                position=position,
                bar_format=XET_TRANSFER_BAR_FORMAT,
                leave=True,
            )
            self._owns_transfer_bar = True

    @property
    def _aggregated(self) -> bool:
        return self.transfer_bar is not None and self.transfer_bar is self.reconstruction_bar

    def update_progress(
        self,
        group_report,
        _item_reports: dict | None = None
    ) -> None:
        bytes_inc = max(0, group_report.total_bytes_completed - self._prev_bytes_completed)
        transfer_inc = max(0, group_report.total_transfer_bytes_completed - self._prev_transfer_bytes_completed)
        self._prev_bytes_completed = group_report.total_bytes_completed
        self._prev_transfer_bytes_completed = group_report.total_transfer_bytes_completed

        if bytes_inc > 0:
            self.reconstruction_bar.update(bytes_inc)
            self.reconstruction_bar.set_postfix_str(
                _format_speed_postfix(group_report.total_bytes_completion_rate), refresh=False
            )

        if transfer_inc > 0 and self.transfer_bar is not None:
            if self._aggregated:
                self.reconstruction_bar.update_transfer(transfer_inc)
                self.reconstruction_bar.set_transfer_postfix_str(
                    _format_speed_postfix(group_report.total_transfer_bytes_completion_rate), refresh=False
                )
            else:
                _update_transfer_bar(self.transfer_bar, transfer_inc)
                self.transfer_bar.set_postfix_str(
                    _format_speed_postfix(group_report.total_transfer_bytes_completion_rate), refresh=False
                )

        if group_report.total_bytes:
            _set_monotonic_total(self.reconstruction_bar, group_report.total_bytes)

    def close(self) -> None:
        if self.transfer_bar is not None and self._owns_transfer_bar:
            
            n = getattr(self.transfer_bar, "n", 0)
            if n > 0 and hasattr(self.transfer_bar, "total") and self.transfer_bar.total != n:
                self.transfer_bar.total = n
                self.transfer_bar.refresh()
            
            if hasattr(self.transfer_bar, "close"):
                self.transfer_bar.close()
        if self._owns_reconstruction_bar and hasattr(self.reconstruction_bar, "close"):
            self.reconstruction_bar.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        self.close()
