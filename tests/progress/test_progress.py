import logging

import pytest

from iantirta.models.remote._progress import (
    _FallbackTqdm,
    _create_progress_bar,
    get_context_progressbar,
    is_tqdm_disabled,
)


def test_fallback_tqdm_initial_value():
    progress = _FallbackTqdm()

    assert progress.n == 0


def test_fallback_tqdm_update(capsys):
    progress = _FallbackTqdm()

    progress.update(10)

    assert progress.n == 10


def test_fallback_tqdm_accumulates_updates(capsys):
    progress = _FallbackTqdm()

    progress.update(10)
    progress.update(20)
    progress.update(5)

    assert progress.n == 35


def test_fallback_tqdm_accepts_extra_arguments():
    progress = _FallbackTqdm(
        total=100,
        initial=20,
        desc="Downloading",
        unit="B",
        unit_scale=True,
    )

    progress.update(10)

    assert progress.n == 10


def test_fallback_tqdm_context_manager():
    progress = _FallbackTqdm()

    with progress as result:
        assert result is progress


def test_fallback_tqdm_context_manager_does_not_suppress_exception():
    progress = _FallbackTqdm()

    with pytest.raises(RuntimeError, match="test"), progress:
        raise RuntimeError("test")


def test_is_tqdm_disabled_notset():
    assert is_tqdm_disabled(logging.NOTSET) is True


def test_is_tqdm_disabled_tqdm_position(monkeypatch):
    monkeypatch.setenv("TQDM_POSITION", "-1")

    assert is_tqdm_disabled(logging.INFO) is False


def test_is_tqdm_disabled_default(monkeypatch):
    monkeypatch.delenv("TQDM_POSITION", raising=False)

    assert is_tqdm_disabled(logging.INFO) is None


def test_is_tqdm_disabled_default_for_debug(monkeypatch):
    monkeypatch.delenv("TQDM_POSITION", raising=False)

    assert is_tqdm_disabled(logging.DEBUG) is None


def test_is_tqdm_disabled_notset_overrides_environment(monkeypatch):
    monkeypatch.setenv("TQDM_POSITION", "-1")

    assert is_tqdm_disabled(logging.NOTSET) is True


def test_create_progress_bar_fallback():
    progress = _create_progress_bar(
        cls=_FallbackTqdm,
        log_level=logging.INFO,
        total=100,
        initial=20,
        desc="Downloading",
    )

    assert isinstance(progress, _FallbackTqdm)
    assert progress.n == 0


def test_create_progress_bar_non_tqdm_class():
    class DummyProgress:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    progress = _create_progress_bar(
        cls=DummyProgress,
        log_level=logging.INFO,
        total=100,
    )

    assert progress.kwargs == {
        "total": 100,
    }


def test_get_context_progressbar_uses_supplied_bar():
    progress = _FallbackTqdm()

    context = get_context_progressbar(
        desc="Downloading",
        log_level=logging.INFO,
        _tqdm_bar=progress,
    )

    with context as result:
        assert result is progress


def test_get_context_progressbar_does_not_close_supplied_bar():
    progress = _FallbackTqdm()
    progress.update(10)

    with get_context_progressbar(
        desc="Downloading",
        log_level=logging.INFO,
        _tqdm_bar=progress,
    ):
        pass

    assert progress.n == 10


def test_get_context_progressbar_uses_custom_class():
    class DummyProgress:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return None

    with get_context_progressbar(
        desc="Downloading",
        log_level=logging.INFO,
        total=100,
        initial=20,
        unit="B",
        unit_scale=True,
        tqdm_class=DummyProgress,
    ) as progress:
        assert progress.kwargs == {
            "unit": "B",
            "unit_scale": True,
            "total": 100,
            "initial": 20,
            "desc": "Downloading",
        }

def test_get_context_progressbar_accepts_name():
    progress = _FallbackTqdm()

    with get_context_progressbar(
        desc="Downloading",
        log_level=logging.INFO,
        name="download",
        _tqdm_bar=progress,
    ) as result:
        assert result is progress


def test_progress_bar_increasing():
    progress = _FallbackTqdm()

    for _ in range(100):
        progress.update(1)

    assert progress.n == 100
