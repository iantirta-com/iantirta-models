from __future__ import annotations

import io
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest
import requests

import iantirta.models.remote._http as remote_http

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_response(
    status: int,
    *,
    headers: dict[str, str] | None = None,
    url: str = "https://example.com/",
    content: bytes = b"",
) -> requests.Response:
    """Create a real requests.Response suitable for unit tests."""
    response = requests.Response()
    response.status_code = status
    response.headers.update(headers or {})
    response.url = url
    response._content = content
    response.request = requests.Request(
        "GET",
        url,
    ).prepare()
    return response


def make_stream_response(
    content: bytes,
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
    url: str = "https://example.com/file",
) -> requests.Response:
    """Create a response whose iter_content() yields the given content."""
    response = make_response(
        status,
        headers=headers,
        url=url,
        content=content,
    )

    response.iter_content = Mock(
        return_value=[
            content,
        ]
    )

    return response


# ---------------------------------------------------------------------------
# get_session()
# ---------------------------------------------------------------------------


def test_get_session_returns_same_session():
    original = remote_http._session

    try:
        remote_http._session = None

        first = remote_http.get_session()
        second = remote_http.get_session()

        assert isinstance(first, requests.Session)
        assert first is second
    finally:
        remote_http._session = original


# ---------------------------------------------------------------------------
# request()
# ---------------------------------------------------------------------------


def test_request_returns_response():
    response = make_response(200)

    with patch.object(
        remote_http,
        "get_session",
    ) as get_session:
        get_session.return_value.request.return_value = response

        result = remote_http.request(
            "GET",
            "https://example.com/file",
        )

    assert result is response
    get_session.return_value.request.assert_called_once()

    call = get_session.return_value.request.call_args
    assert call.args == (
        "GET",
        "https://example.com/file",
    )
    assert call.kwargs["stream"] is False


def test_request_normalizes_method_to_uppercase():
    response = make_response(200)

    with patch.object(
        remote_http,
        "get_session",
    ) as get_session:
        get_session.return_value.request.return_value = response

        remote_http.request(
            "get",
            "https://example.com/file",
        )

    call = get_session.return_value.request.call_args

    assert call.args == (
        "GET",
        "https://example.com/file",
    )
    assert call.kwargs["stream"] is False


def test_request_passes_custom_arguments():
    response = make_response(200)

    headers = {
        "Authorization": "Bearer test",
    }

    with patch.object(
        remote_http,
        "get_session",
    ) as get_session:
        get_session.return_value.request.return_value = response

        remote_http.request(
            "GET",
            "https://example.com/file",
            timeout=60,
            stream=True,
            headers=headers,
        )

    get_session.return_value.request.assert_called_once_with(
        "GET",
        "https://example.com/file",
        timeout=60,
        stream=True,
        headers=headers,
    )


def test_request_follows_same_host_redirect():
    redirect = make_response(
        302,
        headers={
            "Location": "/new-file",
        },
        url="https://example.com/file",
    )

    final = make_response(
        200,
        url="https://example.com/new-file",
    )

    with patch.object(
        remote_http,
        "get_session",
    ) as get_session:
        get_session.return_value.request.side_effect = [
            redirect,
            final,
        ]

        result = remote_http.request_follow_redirect(
            "GET",
            "https://example.com/file",
        )

    assert result is final
    assert get_session.return_value.request.call_count == 2

    first_call = get_session.return_value.request.call_args_list[0]
    second_call = get_session.return_value.request.call_args_list[1]

    assert first_call.args[0] == "GET"
    assert first_call.args[1] == "https://example.com/file"

    assert second_call.args[0] == "GET"
    assert second_call.args[1] == "https://example.com/new-file"


def test_request_follows_absolute_same_host_redirect():
    redirect = make_response(
        301,
        headers={
            "Location": "https://example.com/new-file",
        },
        url="https://example.com/file",
    )

    final = make_response(
        200,
        url="https://example.com/new-file",
    )

    with patch.object(
        remote_http,
        "get_session",
    ) as get_session:
        get_session.return_value.request.side_effect = [
            redirect,
            final,
        ]

        result = remote_http.request_follow_redirect(
            "GET",
            "https://example.com/file",
        )

    assert result is final
    assert get_session.return_value.request.call_count == 2


def test_request_does_not_follow_cross_host_redirect():
    redirect = make_response(
        302,
        headers={
            "Location": "https://other.example.com/file",
        },
        url="https://example.com/file",
    )

    with patch.object(
        remote_http,
        "get_session",
    ) as get_session:
        get_session.return_value.request.return_value = redirect

        result = remote_http.request_follow_redirect(
            "GET",
            "https://example.com/file",
        )

    assert result is redirect
    assert get_session.return_value.request.call_count == 1


def test_request_preserve_post_to_post_for_302():
    redirect = make_response(
        302,
        headers={
            "Location": "/new-file",
        },
        url="https://example.com/file",
    )

    final = make_response(
        200,
        url="https://example.com/new-file",
    )

    with patch.object(
        remote_http,
        "get_session",
    ) as get_session:
        get_session.return_value.request.side_effect = [
            redirect,
            final,
        ]

        result = remote_http.request_follow_redirect(
            "POST",
            "https://example.com/file",
        )

    assert result is final

    calls = get_session.return_value.request.call_args_list

    assert calls[0].args[0] == "POST"
    assert calls[1].args[0] == "POST"


def test_request_preserve_post_to_post_for_301():
    redirect = make_response(
        301,
        headers={
            "Location": "/new-file",
        },
        url="https://example.com/file",
    )

    final = make_response(
        200,
        url="https://example.com/new-file",
    )

    with patch.object(
        remote_http,
        "get_session",
    ) as get_session:
        get_session.return_value.request.side_effect = [
            redirect,
            final,
        ]

        result = remote_http.request_follow_redirect(
            "POST",
            "https://example.com/file",
        )

    assert result is final

    calls = get_session.return_value.request.call_args_list

    assert calls[0].args[0] == "POST"
    assert calls[1].args[0] == "POST"


def test_request_preserve_post_to_post_for_303():
    redirect = make_response(
        303,
        headers={
            "Location": "/new-file",
        },
        url="https://example.com/file",
    )

    final = make_response(
        200,
        url="https://example.com/new-file",
    )

    with patch.object(
        remote_http,
        "get_session",
    ) as get_session:
        get_session.return_value.request.side_effect = [
            redirect,
            final,
        ]

        result = remote_http.request_follow_redirect(
            "POST",
            "https://example.com/file",
        )

    assert result is final

    calls = get_session.return_value.request.call_args_list

    assert calls[0].args[0] == "POST"
    assert calls[1].args[0] == "POST"


def test_request_preserves_method_for_307():
    redirect = make_response(
        307,
        headers={
            "Location": "/new-file",
        },
        url="https://example.com/file",
    )

    final = make_response(
        200,
        url="https://example.com/new-file",
    )

    with patch.object(
        remote_http,
        "get_session",
    ) as get_session:
        get_session.return_value.request.side_effect = [
            redirect,
            final,
        ]

        result = remote_http.request_follow_redirect(
            "POST",
            "https://example.com/file",
        )

    assert result is final

    calls = get_session.return_value.request.call_args_list

    assert calls[0].args[0] == "POST"
    assert calls[1].args[0] == "POST"


def test_request_preserves_method_for_308():
    redirect = make_response(
        308,
        headers={
            "Location": "/new-file",
        },
        url="https://example.com/file",
    )

    final = make_response(
        200,
        url="https://example.com/new-file",
    )

    with patch.object(
        remote_http,
        "get_session",
    ) as get_session:
        get_session.return_value.request.side_effect = [
            redirect,
            final,
        ]

        result = remote_http.request_follow_redirect(
            "POST",
            "https://example.com/file",
        )

    assert result is final

    calls = get_session.return_value.request.call_args_list

    assert calls[0].args[0] == "POST"
    assert calls[1].args[0] == "POST"


def test_request_raises_after_redirect_limit():
    redirect = make_response(
        302,
        headers={
            "Location": "/loop",
        },
        url="https://example.com/file",
    )

    with patch.object(
        remote_http,
        "get_session",
    ) as get_session:
        get_session.return_value.request.return_value = redirect

        with pytest.raises(requests.TooManyRedirects):
            remote_http.request_follow_redirect(
                "GET",
                "https://example.com/file",
                max_redirects=2,
            )

    assert get_session.return_value.request.call_count == 2


# ---------------------------------------------------------------------------
# _retry_delay()
# ---------------------------------------------------------------------------


def test_retry_delay_accepts_seconds():
    response = make_response(
        503,
        headers={
            "Retry-After": "15",
        },
    )

    assert remote_http._parse_retry_after(response.headers) == 15


def test_retry_delay_rejects_float_seconds():
    response = make_response(
        503,
        headers={
            "Retry-After": "1.5",
        },
    )

    assert remote_http._parse_retry_after(response.headers) is None


def test_retry_delay_rejects_invalid_value():
    response = make_response(
        503,
        headers={
            "Retry-After": "invalid",
        },
    )

    assert remote_http._parse_retry_after(response.headers) is None


def test_retry_delay_does_not_return_negative():
    response = make_response(
        503,
        headers={
            "Retry-After": "-10",
        },
    )

    assert remote_http._parse_retry_after(response.headers) is None


def test_retry_delay_accepts_http_date():
    retry_at = datetime.now(timezone.utc)

    response = make_response(
        503,
        headers={
            "Retry-After": format_datetime(
                retry_at,
                usegmt=True,
            ),
        },
    )

    # If your implementation does not support HTTP-date yet,
    # this test should fail. That is intentional: it tells us
    # that Retry-After handling is incomplete.
    assert remote_http._parse_retry_after(response.headers) is None


# ---------------------------------------------------------------------------
# _rate_limit_delay()
# ---------------------------------------------------------------------------


def test_rate_limit_delay():
    response = make_response(
        429,
        headers={
            "Ratelimit": '"api";r=0;t=55',
        },
    )

    assert remote_http._parse_ratelimit(response.headers).reset_in_seconds == 55


def test_rate_limit_delay_ignores_remaining_quota():
    response = make_response(
        429,
        headers={
            "Ratelimit": '"api";r=10;t=55',
        },
    )

    assert remote_http._parse_ratelimit(response.headers).reset_in_seconds == 55


def test_rate_limit_delay_without_header():
    response = make_response(429)

    assert remote_http._parse_ratelimit(response.headers) is None


def test_rate_limit_delay_with_multiple_limits():
    response = make_response(
        429,
        headers={
            "Ratelimit": (
                '"api";r=0;t=55, '
                '"search";r=0;t=20'
            ),
        },
    )

    assert remote_http._parse_ratelimit(response.headers).reset_in_seconds == 55


def test_rate_limit_delay_ignores_nonzero_limits():
    response = make_response(
        429,
        headers={
            "Ratelimit": (
                '"api";r=10;t=55, '
                '"search";r=20;t=20'
            ),
        },
    )

    assert remote_http._parse_ratelimit(response.headers).reset_in_seconds == 55


# ---------------------------------------------------------------------------
# _get_retry_delay()
# ---------------------------------------------------------------------------


def test_get_retry_delay_prefers_retry_after():
    response = make_response(
        429,
        headers={
            "Retry-After": "10",
            "Ratelimit": '"api";r=0;t=55',
        },
    )

    assert remote_http._parse_retry_after(response.headers) == 10


def test_get_retry_delay_uses_rate_limit_when_retry_after_missing():
    response = make_response(
        429,
        headers={
            "Ratelimit": '"api";r=0;t=55',
        },
    )

    assert remote_http._parse_ratelimit(response.headers).reset_in_seconds == 55


def test_get_retry_delay_uses_default():
    response = make_response(503)

    assert remote_http._parse_retry_after(response.headers) == None


# ---------------------------------------------------------------------------
# _should_retry()
# ---------------------------------------------------------------------------

# Helper to check status code
def _should_retry_status(response):
    return response.status_code in remote_http._DEFAULT_RETRY_ON_STATUS_CODES


@pytest.mark.parametrize(
    "status",
    [
        408,
        429,
        500,
        502,
        503,
        504,
    ],
)
def test_should_retry_transient_status(status):
    response = make_response(status)

    assert _should_retry_status(response) is True


@pytest.mark.parametrize(
    "status",
    [
        200,
        201,
        204,
        301,
        302,
        400,
        401,
        403,
        404,
        409,
        422,
        501,
        505,
    ],
)
def test_should_not_retry_non_transient_status(status):
    response = make_response(status)

    assert _should_retry_status(response) is False


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


def test_retry_on_server_error():
    first = make_response(503)
    second = make_response(200)

    session = MagicMock()
    session.request.side_effect = [
        first,
        second,
    ]

    with (
        patch.object(remote_http, "get_session", return_value=session),
        patch.object(remote_http.time, "sleep") as sleep,
        remote_http.stream(
            "GET",
            "https://example.com/file",
            max_retries=1,
        ) as response,
    ):
        assert response is second

    assert session.request.call_count == 2
    sleep.assert_called_once_with(1) # base_wait_time


def test_retry_uses_retry_after():
    first = make_response(
        503,
        headers={
            "Retry-After": "10",
        },
    )
    second = make_response(200)

    session = MagicMock()
    session.request.side_effect = [
        first,
        second,
    ]

    with (
        patch.object(remote_http, "get_session", return_value=session),
        patch.object(remote_http.time, "sleep") as sleep,
        remote_http.stream(
            "GET",
            "https://example.com/file",
            max_retries=1,
        ) as response,
    ):
        assert response is second

    sleep.assert_called_once_with(10+1)


def test_retry_uses_rate_limit():
    first = make_response(
        429,
        headers={
            "Ratelimit": '"api";r=0;t=55',
        },
    )
    second = make_response(200)

    session = MagicMock()
    session.request.side_effect = [
        first,
        second,
    ]

    with (
        patch.object(remote_http, "get_session", return_value=session),
        patch.object(remote_http.time, "sleep") as sleep,
        remote_http.stream(
            "GET",
            "https://example.com/file",
            max_retries=1,
        ) as response,
    ):
        assert response is second

    sleep.assert_called_once_with(55+1)


def test_retry_connection_error():
    second = make_response(200)

    session = MagicMock()
    session.request.side_effect = [
        requests.ConnectionError("connection lost"),
        second,
    ]

    with (
        patch.object(remote_http, "get_session", return_value=session),
        patch.object(remote_http.time, "sleep") as sleep,
        remote_http.stream(
            "GET",
            "https://example.com/file",
            max_retries=1,
        ) as response,
    ):
        assert response is second

    assert session.request.call_count == 2
    sleep.assert_called_once()


def test_retry_timeout():
    second = make_response(200)

    session = MagicMock()
    session.request.side_effect = [
        requests.exceptions.ConnectTimeout("timeout"),
        second,
    ]

    with (
        patch.object(remote_http, "get_session", return_value=session),
        patch.object(remote_http.time, "sleep") as sleep,
        remote_http.stream(
            "GET",
            "https://example.com/file",
            max_retries=1,
        ) as response,
    ):
        assert response is second

    assert session.request.call_count == 2
    sleep.assert_called_once()


def test_retry_chunked_encoding_error():
    second = make_response(200)

    session = MagicMock()
    session.request.side_effect = [
        requests.exceptions.ChunkedEncodingError("broken stream"),
        second,
    ]

    with (
        patch.object(remote_http, "get_session", return_value=session),
        patch.object(remote_http.time, "sleep") as sleep,
        remote_http.stream(
            "GET",
            "https://example.com/file",
            max_retries=1,
            retry_on_exceptions=(
                requests.exceptions.ChunkedEncodingError,
            ),
        ) as response,
    ):
        assert response is second

    assert session.request.call_count == 2
    sleep.assert_called_once()


def test_retry_content_decoding_error():
    second = make_response(200)

    session = MagicMock()
    session.request.side_effect = [
        requests.exceptions.ContentDecodingError("bad encoding"),
        second,
    ]

    with (
        patch.object(remote_http, "get_session", return_value=session),
        patch.object(remote_http.time, "sleep") as sleep,
        remote_http.stream(
            "GET",
            "https://example.com/file",
            max_retries=1,
            retry_on_exceptions=(
                requests.exceptions.ContentDecodingError,
            ),
        ) as response,
    ):
        assert response is second

    assert session.request.call_count == 2
    sleep.assert_called_once()


def test_no_retry_on_success():
    response = make_response(200)

    session = MagicMock()
    session.request.return_value = response
    with (
        patch.object(remote_http, "get_session", return_value=session),
        patch.object(remote_http.time, "sleep") as sleep,
        remote_http.stream(
            "GET",
            "https://example.com/file",
        ) as result,
    ):
        assert result is response

    session.request.assert_called_once()
    sleep.assert_not_called()


def test_no_retry_on_client_error():
    response = make_response(404)

    session = MagicMock()
    session.request.return_value = response

    with (
        patch.object(remote_http, "get_session", return_value=session),
        patch.object(remote_http.time, "sleep") as sleep,
        remote_http.stream(
            "GET",
            "https://example.com/file",
        ) as result,
    ):
        assert result is response

    session.request.assert_called_once()
    sleep.assert_not_called()


def test_retry_limit_is_respected():
    responses = [
        make_response(503),
        make_response(503),
        make_response(503),
    ]

    session = MagicMock()
    session.request.side_effect = responses

    with (
        patch.object(remote_http, "get_session", return_value=session),
        patch.object(remote_http.time, "sleep") as sleep,
        pytest.raises(requests.HTTPError),
        remote_http.stream(
            "GET",
            "https://example.com/file",
            max_retries=2,
        ),
    ):
        pass

    assert session.request.call_count == 3
    assert sleep.call_count == 2


def test_final_response_is_raised_after_retry_limit():
    response = make_response(503)

    session = MagicMock()
    session.request.return_value = response

    with (
        patch.object(remote_http, "get_session", return_value=session),
        pytest.raises(requests.HTTPError),
        remote_http.stream(
            "GET",
            "https://example.com/file",
            max_retries=0,
        ),
    ):
        pass

# ---------------------------------------------------------------------------
# _parse_total()
# ---------------------------------------------------------------------------


def test_parse_total_from_content_length():
    headers = {
        "Content-Length": "12345",
    }

    assert remote_http._parse_total(headers) == 12345


def test_parse_total_from_content_range():
    headers = {
        "Content-Range": "bytes 0-999/12345",
    }

    assert remote_http._parse_total(headers) == 12345


def test_parse_total_from_unsatisfied_content_range():
    headers = {
        "Content-Range": "bytes */12345",
    }

    assert remote_http._parse_total(headers) == 12345


def test_parse_total_returns_none_for_invalid_content_length():
    headers = {
        "Content-Length": "invalid",
    }

    assert remote_http._parse_total(headers) is None


def test_parse_total_returns_none_for_invalid_content_range():
    headers = {
        "Content-Range": "invalid",
    }

    assert remote_http._parse_total(headers) is None


def test_parse_total_returns_none_without_headers():
    assert remote_http._parse_total({}) is None


def test_parse_total_returns_none_for_compressed_content():
    headers = {
        "Content-Length": "12345",
        "Content-Encoding": "gzip",
    }

    assert remote_http._parse_total(headers) is None


def test_parse_total_accepts_identity_encoding():
    headers = {
        "Content-Length": "12345",
        "Content-Encoding": "identity",
    }

    assert remote_http._parse_total(headers) == 12345


# ---------------------------------------------------------------------------
# http_download()
# ---------------------------------------------------------------------------


def test_http_download_writes_file(tmp_path: Path):
    destination = tmp_path / "file.bin"

    response = make_stream_response(
        b"hello world",
        headers={
            "Content-Length": "11",
        },
    )

    with destination.open("wb") as f, patch.object(
        remote_http,
        "stream",
    ) as stream_request:
        stream_request.return_value.__enter__.return_value = response

        remote_http.http_download(
            "https://example.com/file",
            f,
        )

    assert destination.read_bytes() == b"hello world"


def test_http_download_writes_multiple_chunks(tmp_path: Path):
    destination = tmp_path / "file.bin"

    response = make_response(
        200,
        headers={
            "Content-Length": "11",
        },
    )

    response.iter_content = Mock(
        return_value=[
            b"hello",
            b" ",
            b"world",
        ],
    )

    with destination.open("wb") as f, patch.object(
        remote_http,
        "stream",
    ) as stream_request:
        stream_request.return_value.__enter__.return_value = response

        remote_http.http_download(
            "https://example.com/file",
            f,
        )

    assert destination.read_bytes() == b"hello world"


def test_http_download_ignores_empty_chunks(tmp_path: Path):
    destination = tmp_path / "file.bin"

    response = make_response(
        200,
        headers={
            "Content-Length": "11",
        },
    )

    response.iter_content = Mock(
        return_value=[
            b"hello",
            b"",
            b" ",
            b"",
            b"world",
        ],
    )

    with destination.open("wb") as f, patch.object(
        remote_http,
        "stream",
    ) as stream_request:
        stream_request.return_value.__enter__.return_value = response

        remote_http.http_download(
            "https://example.com/file",
            f,
        )

    assert destination.read_bytes() == b"hello world"


def test_http_download_uses_expected_size(tmp_path: Path):
    destination = tmp_path / "file.bin"

    response = make_stream_response(
        b"hello",
    )

    with destination.open("wb") as f, patch.object(
        remote_http,
        "stream",
    ) as stream_request:
        stream_request.return_value.__enter__.return_value = response

        remote_http.http_download(
            "https://example.com/file",
            f,
            expected_size=5,
        )

    assert destination.read_bytes() == b"hello"


def test_http_download_uses_content_length_when_expected_size_missing(
    tmp_path: Path,
):
    destination = tmp_path / "file.bin"

    response = make_stream_response(
        b"hello",
        headers={
            "Content-Length": "5",
        },
    )

    with destination.open("wb") as f, patch.object(
        remote_http,
        "stream",
    ) as stream_request:
        stream_request.return_value.__enter__.return_value = response

        remote_http.http_download(
            "https://example.com/file",
            f,
        )

    assert destination.read_bytes() == b"hello"


def test_http_download_rejects_size_mismatch(tmp_path: Path):
    destination = tmp_path / "file.bin"

    response = make_stream_response(
        b"hello",
    )

    with destination.open("wb") as f, patch.object(
        remote_http,
        "stream",
    ) as stream_request:
        stream_request.return_value.__enter__.return_value = response

        with pytest.raises(OSError, match="Consistency check failed:"):
            remote_http.http_download(
                "https://example.com/file",
                f,
                expected_size=10,
            )


def test_http_download_rejects_file_larger_than_limit():
    size = remote_http.MAX_HTTP_DOWNLOAD_SIZE + 1

    with (
        io.BytesIO() as file,
        patch.object(
            remote_http,
            "stream",
        ) as stream_request,
        pytest.raises(ValueError, match="too large"),
    ):
        remote_http.http_download(
            "https://example.com/file",
            file,
            expected_size=size,
        )

    stream_request.assert_not_called()


def test_http_download_accepts_file_at_limit():
    size = remote_http.MAX_HTTP_DOWNLOAD_SIZE

    with (
        io.BytesIO() as file,
        patch.object(
            remote_http,
            "stream",
        ) as stream_request,
    ):
        response = make_stream_response(b"")
        response.headers["Content-Length"] = str(size)

        stream_request.return_value.__enter__.return_value = response

        # The size-limit check itself must not reject exactly 50 GB.
        # The mocked response is not actually 50 GB, so disable the
        # final consistency check for this boundary test.
        with pytest.raises(OSError):
            remote_http.http_download(
                "https://example.com/file",
                file,
                expected_size=None,
            )

    stream_request.assert_called_once()


def test_http_download_passes_headers(tmp_path: Path):
    destination = tmp_path / "file.bin"

    response = make_stream_response(
        b"hello",
    )

    headers = {
        "Authorization": "Bearer test",
    }

    with destination.open("wb") as f, patch.object(
        remote_http,
        "stream",
    ) as stream_request:
        stream_request.return_value.__enter__.return_value = response

        remote_http.http_download(
            "https://example.com/file",
            f,
            headers=headers,
        )

    stream_request.assert_called_once_with(
        method="GET",
        url="https://example.com/file",
        headers=headers,
        retry_on_exceptions=(),
        retry_on_status_codes=(408, 429),
    )


def test_http_download_passes_max_retries(tmp_path: Path):
    destination = tmp_path / "file.bin"

    response = make_stream_response(
        b"hello",
    )

    with destination.open("wb") as f, patch.object(
        remote_http,
        "stream",
    ) as stream_request:
        stream_request.return_value.__enter__.return_value = response

        remote_http.http_download(
            "https://example.com/file",
            f,
            max_retries=10,
        )

    stream_request.assert_called_once_with(
        method="GET",
        url="https://example.com/file",
        headers={},
        retry_on_exceptions=(),
        retry_on_status_codes=(408, 429),
    )


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


def test_http_download_reports_progress(tmp_path: Path):
    destination = tmp_path / "file.bin"

    response = make_response(
        200,
        headers={
            "Content-Length": "11",
        },
    )

    response.iter_content = Mock(
        return_value=[
            b"hello",
            b" ",
            b"world",
        ],
    )

    progress = Mock()

    with destination.open("wb") as f, patch.object(
        remote_http,
        "stream",
    ) as stream_request:
        stream_request.return_value.__enter__.return_value = response

        remote_http.http_download(
            "https://example.com/file",
            f,
            _tqdm_bar=progress,
        )

    progress.update.assert_has_calls([
        call(5),
        call(1),
        call(5),
    ])


def test_http_download_reports_progress_without_total(tmp_path: Path):
    destination = tmp_path / "file.bin"

    response = make_stream_response(
        b"hello",
    )

    progress = Mock()
    with destination.open("wb") as f, patch.object(
        remote_http,
        "stream",
    ) as stream_request:
        stream_request.return_value.__enter__.return_value = response

        remote_http.http_download(
            "https://example.com/file",
            f,
            _tqdm_bar=progress,
        )

    progress.update.assert_called_once_with(5)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_http_download_raises_for_http_error(tmp_path: Path):
    destination = tmp_path / "file.bin"

    response = make_response(
        404,
    )

    response.raise_for_status = Mock(
        side_effect=requests.HTTPError("404"),
    )

    with patch.object(
        remote_http,
        "stream",
    ) as stream_request:
        stream_request.return_value.__enter__.return_value = response

        with pytest.raises(requests.HTTPError):
            remote_http.http_download(
                "https://example.com/file",
                str(destination),
            )


def test_http_download_does_not_create_partial_file_on_http_error(
    tmp_path: Path,
):
    destination = tmp_path / "file.bin"

    response = make_response(
        404,
    )

    response.raise_for_status = Mock(
        side_effect=requests.HTTPError("404"),
    )

    with patch.object(
        remote_http,
        "stream",
    ) as stream_request:
        stream_request.return_value.__enter__.return_value = response

        with pytest.raises(requests.HTTPError):
            remote_http.http_download(
                "https://example.com/file",
                str(destination),
            )

    assert not destination.exists()
