import pytest
from unittest.mock import MagicMock, patch
from requests.exceptions import HTTPError


@pytest.fixture
def mms_repo():
    return "facebook/mms-1b-all"

@pytest.fixture
def qwen_repo():
    return "Qwen/Qwen3-ASR-1.7B-hf"

@pytest.fixture
def mock_repo():
    return "pytest/model_name"

@pytest.fixture
def xet_repo():
    return "hf-internal-testing/tiny-processor-phi4_multimodal"

@pytest.fixture
def repo_id(request):
    """Dynamically returns real vs mock repo_id based on test markers."""
    marker = request.node.get_closest_marker("manual")
    if marker:
        return "facebook/mms-1b-all"  # Real repository for manual tests

    return "pytest/model_name"


@pytest.fixture(scope="session")
def mock_response():
    """Generates optimized MagicMock Response objects on demand."""
    def _create_response(
        url: str = "www.example.com",
        status_code=200,
        headers: dict[str, str] | None = None,
        json_payload=None,
        custom_error=None,
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_payload or {}
        mock_resp.request.url.return_value = url

        mock_resp.headers = {}

        if headers:
            for key, value in headers.items():
                if isinstance(value, list):
                    value = ", ".join(value)
                if key in mock_resp.headers:
                    mock_resp.headers[key] += f", {value}"
                else:
                    mock_resp.headers[key] = value

        mock_raw = MagicMock()
        mock_resp.raw = mock_raw
        mock_raw.headers.getlist.side_effect = lambda k: [mock_resp.headers.get(k)]

        # Configure raise_for_status() behavior
        if custom_error:
            mock_resp.raise_for_status.side_effect = custom_error
        elif status_code >= 400:
            # Automatically raise standard HTTPError for 4xx/5xx status codes
            mock_resp.raise_for_status.side_effect = HTTPError(f"Mocked {status_code} Error")
        else:
            # Do nothing for successful status codes (2xx)
            mock_resp.raise_for_status.return_value = None
            
        return mock_resp
    return _create_response
