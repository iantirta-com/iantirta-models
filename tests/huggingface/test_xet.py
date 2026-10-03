import sys
from unittest.mock import MagicMock, patch
from pathlib import Path
import os

import pytest

from iantirta.models.remote import _xet as remote_xet
from iantirta.models.remote._types import XetFileData


def test_available_when_hf_xet_is_installed():
    remote_xet.available.cache_clear()

    fake_hf_xet = MagicMock()

    with patch.dict(sys.modules, {"hf_xet": fake_hf_xet}):
        assert remote_xet.available() is True

    remote_xet.available.cache_clear()


def test_available_when_hf_xet_is_not_installed():
    remote_xet.available.cache_clear()

    original = sys.modules.pop("hf_xet", None)

    try:
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args, **kwargs: (
                (_ for _ in ()).throw(ImportError)
                if name == "hf_xet"
                else __import__(name, *args, **kwargs)
            ),
        ):
            assert remote_xet.available() is False
    finally:
        if original is not None:
            sys.modules["hf_xet"] = original

    remote_xet.available.cache_clear()


def test_parse_file_data_returns_none_when_xet_unavailable():
    response = MagicMock()

    with patch.object(remote_xet, "available", return_value=False):
        assert remote_xet.parse_file_data(response) is None


def test_parse_file_data_returns_none_for_none():
    with patch.object(remote_xet, "available", return_value=True):
        assert remote_xet.parse_file_data(None) is None


def test_parse_file_data_from_refresh_route_header():
    response = MagicMock()
    response.headers = {
        "X-Xet-Hash": "abc123",
        "X-Xet-Refresh-Route": "https://example.com/xet/refresh",
    }
    response.links = {}

    with patch.object(remote_xet, "available", return_value=True):
        result = remote_xet.parse_file_data(response)

    assert result == XetFileData(
        file_hash="abc123",
        refresh_route="https://example.com/xet/refresh",
    )


def test_parse_file_data_uses_xet_auth_link():
    response = MagicMock()
    response.headers = {
        "X-Xet-Hash": "abc123",
        "X-Xet-Refresh-Route": "https://example.com/header",
    }
    response.links = {
        "xet-auth": {
            "url": "https://example.com/link",
        }
    }

    with patch.object(remote_xet, "available", return_value=True):
        result = remote_xet.parse_file_data(response)

    assert result == XetFileData(
        file_hash="abc123",
        refresh_route="https://example.com/link",
    )


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-Xet-Hash": "abc123"},
        {"X-Xet-Refresh-Route": "https://example.com/refresh"},
    ],
)
def test_parse_file_data_returns_none_for_missing_headers(headers):
    response = MagicMock()
    response.headers = headers
    response.links = {}

    with patch.object(remote_xet, "available", return_value=True):
        assert remote_xet.parse_file_data(response) is None


def test_xet_connection_info_not_expired(monkeypatch):
    monkeypatch.setattr(remote_xet.time, "time", lambda: 1000)

    info = remote_xet.XetConnectionInfo(
        access_token="token",
        expiration_unix_epoch=1061,
        endpoint="https://cas.example.com",
    )

    assert info.expired is False


def test_xet_connection_info_expired_within_60_seconds(monkeypatch):
    monkeypatch.setattr(remote_xet.time, "time", lambda: 1000)

    info = remote_xet.XetConnectionInfo(
        access_token="token",
        expiration_unix_epoch=1060,
        endpoint="https://cas.example.com",
    )

    assert info.expired is True


def test_xet_connection_info_expired():
    info = remote_xet.XetConnectionInfo(
        access_token="token",
        expiration_unix_epoch=0,
        endpoint="https://cas.example.com",
    )

    assert info.expired is True


def test_parse_xet_connection_info():
    headers = {
        "X-Xet-Cas-Url": "https://cas.example.com",
        "X-Xet-Access-Token": "secret",
        "X-Xet-Token-Expiration": "1234567890",
    }

    result = remote_xet._parse_xet_connection_info(headers)

    assert result == remote_xet.XetConnectionInfo(
        endpoint="https://cas.example.com",
        access_token="secret",
        expiration_unix_epoch=1234567890,
    )


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-Xet-Cas-Url": "https://cas.example.com"},
        {
            "X-Xet-Cas-Url": "https://cas.example.com",
            "X-Xet-Access-Token": "secret",
        },
    ],
)
def test_parse_xet_connection_info_missing_headers(headers):
    assert remote_xet._parse_xet_connection_info(headers) is None


@pytest.mark.parametrize(
    "expiration",
    ["", "abc", None],
)
def test_parse_xet_connection_info_invalid_expiration(expiration):
    headers = {
        "X-Xet-Cas-Url": "https://cas.example.com",
        "X-Xet-Access-Token": "secret",
        "X-Xet-Token-Expiration": expiration,
    }

    assert remote_xet._parse_xet_connection_info(headers) is None


def test_cache_key_includes_url_and_authorization():
    result = remote_xet._cache_key(
        "https://example.com/refresh",
        {"Authorization": "Bearer abc"},
    )

    assert result == (
        "https://example.com/refresh|Bearer abc"
    )


def test_cache_key_authorization_is_case_insensitive():
    first = remote_xet._cache_key(
        "https://example.com/refresh",
        {"Authorization": "Bearer abc"},
    )

    second = remote_xet._cache_key(
        "https://example.com/refresh",
        {"authorization": "Bearer abc"},
    )

    assert first == second


def test_cache_key_differs_for_different_authorization():
    first = remote_xet._cache_key(
        "https://example.com/refresh",
        {"Authorization": "Bearer abc"},
    )

    second = remote_xet._cache_key(
        "https://example.com/refresh",
        {"Authorization": "Bearer xyz"},
    )

    assert first != second


@pytest.fixture(autouse=True)
def clear_xet_cache():
    remote_xet.XET_INFO_CACHE.clear()
    remote_xet.XET_LOCKS.clear()
    yield
    remote_xet.XET_INFO_CACHE.clear()
    remote_xet.XET_LOCKS.clear()


def test_get_xet_connection_info_fetches_token():
    xet_file = XetFileData(
        file_hash="abc",
        refresh_route="https://example.com/refresh",
    )

    response = MagicMock()
    response.headers = {
        "X-Xet-Cas-Url": "https://cas.example.com",
        "X-Xet-Access-Token": "token",
        "X-Xet-Token-Expiration": "9999999999",
    }

    with patch.object(
        remote_xet,
        "request",
        return_value=response,
    ) as request:
        result = remote_xet.get_xet_connection_info(xet_file)

    assert result.endpoint == "https://cas.example.com"
    assert result.access_token == "token"
    assert result.expiration_unix_epoch == 9999999999

    request.assert_called_once_with(
        "GET",
        "https://example.com/refresh",
        headers={},
    )

    response.raise_for_status.assert_called_once()


def test_get_xet_connection_info_uses_cache():
    xet_file = XetFileData(
        file_hash="abc",
        refresh_route="https://example.com/refresh",
    )

    info = remote_xet.XetConnectionInfo(
        endpoint="https://cas.example.com",
        access_token="token",
        expiration_unix_epoch=9999999999,
    )

    key = remote_xet._cache_key(
        xet_file.refresh_route,
        {},
    )

    remote_xet.XET_INFO_CACHE[key] = info

    with patch.object(remote_xet, "request") as request:
        result = remote_xet.get_xet_connection_info(xet_file)

    assert result is info
    request.assert_not_called()


def test_get_xet_connection_info_refreshes_expired_cache(monkeypatch):
    xet_file = XetFileData(
        file_hash="abc",
        refresh_route="https://example.com/refresh",
    )

    key = remote_xet._cache_key(
        xet_file.refresh_route,
        {},
    )

    remote_xet.XET_INFO_CACHE[key] = remote_xet.XetConnectionInfo(
        endpoint="https://old.example.com",
        access_token="old",
        expiration_unix_epoch=0,
    )

    response = MagicMock()
    response.headers = {
        "X-Xet-Cas-Url": "https://new.example.com",
        "X-Xet-Access-Token": "new",
        "X-Xet-Token-Expiration": "9999999999",
    }

    with patch.object(
        remote_xet,
        "request",
        return_value=response,
    ) as request:
        result = remote_xet.get_xet_connection_info(xet_file)

    assert result.endpoint == "https://new.example.com"
    assert result.access_token == "new"
    request.assert_called_once()


def test_get_xet_connection_info_rejects_invalid_response():
    xet_file = XetFileData(
        file_hash="abc",
        refresh_route="https://example.com/refresh",
    )

    response = MagicMock()
    response.headers = {}

    with (
        patch.object(remote_xet, "request", return_value=response),
        pytest.raises(
            ValueError,
            match="Xet headers have not been correctly set",
        ),
    ):
        remote_xet.get_xet_connection_info(xet_file)


def test_xet_session_holder_creates_session():
    holder = remote_xet.XetSessionHolder()
    fake_session = MagicMock()

    fake_module = MagicMock()
    fake_module.XetSession.return_value = fake_session

    with patch.dict(sys.modules, {"hf_xet": fake_module}):
        result = holder.get()

    assert result is fake_session
    fake_module.XetSession.assert_called_once()


def test_xet_session_holder_reuses_session():
    holder = remote_xet.XetSessionHolder()
    fake_session = MagicMock()

    fake_module = MagicMock()
    fake_module.XetSession.return_value = fake_session

    with patch.dict(sys.modules, {"hf_xet": fake_module}):
        first = holder.get()
        second = holder.get()

    assert first is second
    fake_module.XetSession.assert_called_once()


def test_xet_session_holder_sigint_abort():
    holder = remote_xet.XetSessionHolder()
    fake_session = MagicMock()

    fake_module = MagicMock()
    fake_module.XetSession.return_value = fake_session

    with patch.dict(sys.modules, {"hf_xet": fake_module}):
        holder.get()
        holder.sigint_abort()

        assert holder._session is None
        assert holder._session_pid is None

    fake_session.sigint_abort.assert_called_once()


def test_xet_session_holder_sigint_abort_ignores_error():
    holder = remote_xet.XetSessionHolder()
    fake_session = MagicMock()
    fake_session.sigint_abort.side_effect = RuntimeError("boom")

    holder._session = fake_session
    holder._session_pid = os.getpid()

    holder.sigint_abort()

    assert holder._session is None
    assert holder._session_pid is None


def test_xet_session_holder_recreates_session_after_fork():
    holder = remote_xet.XetSessionHolder()

    first_session = MagicMock()
    second_session = MagicMock()

    fake_module = MagicMock()
    fake_module.XetSession.side_effect = [
        first_session,
        second_session,
    ]

    with (
        patch.dict(sys.modules, {"hf_xet": fake_module}),
        patch.object(remote_xet.os, "getpid", side_effect=[100, 200]),
    ):
        assert holder.get() is first_session
        assert holder.get() is second_session

    assert fake_module.XetSession.call_count == 2


def test_download():
    xet_file = XetFileData(
        file_hash="abc123",
        refresh_route="https://example.com/refresh",
    )

    connection_info = remote_xet.XetConnectionInfo(
        endpoint="https://cas.example.com",
        access_token="token",
        expiration_unix_epoch=9999999999,
    )

    session = MagicMock()
    group = MagicMock()
    session.new_file_download_group.return_value.__enter__.return_value = group

    fake_xet = MagicMock()
    fake_xet.XetFileInfo.return_value = "file-info"

    progress = MagicMock()

    incomplete_path = Path("/tmp/test.incomplete")

    with (
        patch.dict(sys.modules, {"hf_xet": fake_xet}),
        patch.object(
            remote_xet,
            "get_xet_connection_info",
            return_value=connection_info,
        ),
        patch.object(
            remote_xet,
            "get_xet_session",
            return_value=session,
        ),
        patch.object(
            remote_xet,
            "XetDownloadProgressReporter",
            return_value=progress,
        ),
    ):
        progress.__enter__.return_value = progress

        remote_xet.download(
            xet_file,
            incomplete_path,
            headers={"Authorization": "Bearer abc"},
            expected_size=1234,
            displayed_filename="test.bin",
        )

    session.new_file_download_group.assert_called_once()
    group.start_download_file.assert_called_once_with(
        fake_xet.XetFileInfo.return_value,
        str(incomplete_path.absolute()),
    )


def test_download_requires_hf_xet():
    xet_file = XetFileData(
        file_hash="abc",
        refresh_route="https://example.com/refresh",
    )

    with (
        patch.dict(sys.modules, {"hf_xet": None}),
        pytest.raises(ValueError, match="hf_xet"),
    ):
        remote_xet.download(
            xet_file,
            Path("/tmp/file"),
            headers={},
        )
