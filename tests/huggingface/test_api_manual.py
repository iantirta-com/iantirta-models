# tests/huggingface/test_api_manual.py

from __future__ import annotations

from pathlib import Path

import pytest

from iantirta.models.remote import _xet
from iantirta.models.remote import api
from iantirta.models.remote._http import http_download


pytestmark = pytest.mark.manual


# Small public repository/file intended for testing.
REPO_ID = "hf-internal-testing/tiny-processor-phi4_multimodal"
FILENAME = "config.json"

XET_REPO_ID = REPO_ID
XET_FILENAME = "tokenizer.json"


def test_real_hf_metadata():
    """Fetch real metadata from the Hugging Face Hub."""

    url = api.hf_hub_url(
        REPO_ID,
        FILENAME,
    )

    metadata = api.get_hf_file_metadata(url)

    assert metadata.commit_hash
    assert metadata.etag
    assert metadata.location
    assert metadata.size is not None
    assert metadata.size > 0


def test_real_hf_download(tmp_path: Path):
    """Download a real file from Hugging Face."""

    cache_dir = tmp_path / "cache"

    result = api.hf_hub_download(
        REPO_ID,
        FILENAME,
        outdir=cache_dir,
    )

    assert result.is_file()
    assert result.stat().st_size > 0


def test_real_hf_download_uses_cache(tmp_path: Path):
    """A second download should use the local cache."""

    cache_dir = tmp_path / "cache"

    first = api.hf_hub_download(
        REPO_ID,
        FILENAME,
        outdir=cache_dir,
    )

    first_data = first.read_bytes()

    second = api.hf_hub_download(
        REPO_ID,
        FILENAME,
        outdir=cache_dir,
    )

    assert second == first
    assert second.read_bytes() == first_data


def test_real_hf_download_with_subfolder(tmp_path: Path):
    """Verify the real Hub request works with a normalized subfolder."""

    cache_dir = tmp_path / "cache"

    result = api.hf_hub_download(
        REPO_ID,
        FILENAME,
        subfolder="/",
        outdir=cache_dir,
    )

    assert result.is_file()
    assert result.stat().st_size > 0


def test_real_hf_http_range_request():
    """Verify that the real Hugging Face endpoint supports HTTP Range."""

    url = api.hf_hub_url(
        REPO_ID,
        FILENAME,
    )

    metadata = api.get_hf_file_metadata(url)

    assert metadata.size is not None
    assert metadata.size > 1

    resume_size = 1

    output = bytearray()

    class Buffer:
        def write(self, data: bytes) -> int:
            output.extend(data)
            return len(data)

    http_download(
        url,
        Buffer(),
        resume_size=resume_size,
        expected_size=metadata.size,
    )

    assert len(output) == metadata.size - resume_size


def test_real_hf_http_download_resume(tmp_path: Path):
    """
    Download a real Hugging Face file in two stages.

    The first stage downloads only the beginning of the file.
    The second stage resumes from that position using HTTP Range.
    """

    url = api.hf_hub_url(
        REPO_ID,
        FILENAME,
    )

    metadata = api.get_hf_file_metadata(url)

    assert metadata.size is not None
    assert metadata.size > 2

    output = tmp_path / "resumed.bin"

    # First download: deliberately download only a prefix.
    #
    # We use a separate HTTP Range request directly through the package's
    # HTTP implementation. This is NOT requests.get().
    partial_size = max(1, metadata.size // 2)

    with output.open("wb") as f:
        http_download(
            url,
            f,
            expected_size=partial_size,
            headers={
                "Range": f"bytes=0-{partial_size - 1}",
            },
        )

    assert output.stat().st_size == partial_size

    # Second download: resume from the existing prefix.
    with output.open("ab") as f:
        http_download(
            url,
            f,
            resume_size=partial_size,
            expected_size=metadata.size,
        )

    assert output.stat().st_size == metadata.size


def test_real_hf_http_download_full_file_matches_resumed_file(
    tmp_path: Path,
):
    """
    Verify that a resumed download produces exactly the same bytes
    as a normal complete download.
    """

    url = api.hf_hub_url(
        REPO_ID,
        FILENAME,
    )

    metadata = api.get_hf_file_metadata(url)

    assert metadata.size is not None
    assert metadata.size > 2

    normal = tmp_path / "normal.bin"
    resumed = tmp_path / "resumed.bin"

    # Normal download.
    with normal.open("wb") as f:
        http_download(
            url,
            f,
            expected_size=metadata.size,
        )

    assert normal.stat().st_size == metadata.size

    # Download the first half.
    partial_size = max(1, metadata.size // 2)

    with resumed.open("wb") as f:
        http_download(
            url,
            f,
            expected_size=partial_size,
            headers={
                "Range": f"bytes=0-{partial_size - 1}",
            },
        )

    assert resumed.stat().st_size == partial_size

    # Resume.
    with resumed.open("ab") as f:
        http_download(
            url,
            f,
            resume_size=partial_size,
            expected_size=metadata.size,
        )

    assert resumed.stat().st_size == metadata.size
    assert resumed.read_bytes() == normal.read_bytes()


# ---------------------------------------------------------------------------
# Xet
# ---------------------------------------------------------------------------

def _get_real_xet_metadata():
    """Get metadata for a real Xet-backed Hub file."""
    url = api.hf_hub_url(
        XET_REPO_ID,
        XET_FILENAME,
    )
    metadata = api.get_hf_file_metadata(url)

    assert metadata.commit_hash
    assert metadata.etag
    assert metadata.location
    assert metadata.size is not None
    assert metadata.xet is not None
    assert metadata.xet.file_hash
    assert metadata.xet.refresh_route

    return url, metadata


@pytest.mark.skipif(
    not _xet.available(),
    reason="hf_xet is not installed",
)
def test_real_hf_xet_metadata():
    """Verify the real Hub returns Xet metadata."""
    _, metadata = _get_real_xet_metadata()

    assert metadata.xet is not None
    assert metadata.xet.file_hash
    assert metadata.xet.refresh_route


@pytest.mark.skipif(
    not _xet.available(),
    reason="hf_xet is not installed",
)
def test_real_hf_xet_download(tmp_path: Path):
    """Verify a real Hub file can be downloaded through Xet."""
    _, metadata = _get_real_xet_metadata()

    cache_dir = tmp_path / "cache"

    result = api.hf_hub_download(
        XET_REPO_ID,
        XET_FILENAME,
        outdir=cache_dir,
    )

    assert result.is_file()
    assert metadata.size is not None
    assert result.stat().st_size == metadata.size


@pytest.mark.skipif(
    not _xet.available(),
    reason="hf_xet is not installed",
)
def test_real_hf_xet_download_reuses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify a cached Xet download is not downloaded again."""
    cache_dir = tmp_path / "cache"

    first = api.hf_hub_download(
        XET_REPO_ID,
        XET_FILENAME,
        outdir=cache_dir,
    )

    first_data = first.read_bytes()

    def fail_xet_download(*args, **kwargs):
        raise AssertionError(
            "Xet download was called even though the file was cached."
        )

    monkeypatch.setattr(
        api._xet,
        "download",
        fail_xet_download,
    )

    second = api.hf_hub_download(
        XET_REPO_ID,
        XET_FILENAME,
        outdir=cache_dir,
    )

    assert second == first
    assert second.read_bytes() == first_data


@pytest.mark.skipif(
    not _xet.available(),
    reason="hf_xet is not installed",
)
def test_real_hf_xet_download_matches_http(
    tmp_path: Path,
):
    """
    Download the same real file through Xet and HTTP and verify that the
    reconstructed bytes are identical.
    """
    url, metadata = _get_real_xet_metadata()

    assert metadata.size is not None

    xet_cache = tmp_path / "xet-cache"
    http_file = tmp_path / "http.bin"

    xet_file = api.hf_hub_download(
        XET_REPO_ID,
        XET_FILENAME,
        outdir=xet_cache,
    )

    with http_file.open("wb") as f:
        http_download(
            url,
            f,
            expected_size=metadata.size,
            headers={
                "Accept-Encoding": "identity",
            },
        )

    assert xet_file.stat().st_size == metadata.size
    assert http_file.stat().st_size == metadata.size
    assert xet_file.read_bytes() == http_file.read_bytes()


@pytest.mark.skipif(
    not _xet.available(),
    reason="hf_xet is not installed",
)
def test_real_hf_xet_download_from_existing_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Verify that an existing incomplete file does not corrupt the resulting
    Xet download.

    This is intentionally NOT called a resume test: the current Xet wrapper
    does not expose a resume offset. It verifies recovery when an incomplete
    destination already exists.
    """
    _, metadata = _get_real_xet_metadata()

    assert metadata.size is not None

    cache_dir = tmp_path / "cache"

    original_download = api._xet.download

    def wrapped_download(*args, **kwargs):
        incomplete_path = args[1]

        # Create garbage representing an interrupted previous download.
        incomplete_path.write_bytes(
            b"interrupted-download"
        )

        return original_download(*args, **kwargs)

    monkeypatch.setattr(
        api._xet,
        "download",
        wrapped_download,
    )

    result = api.hf_hub_download(
        XET_REPO_ID,
        XET_FILENAME,
        outdir=cache_dir,
    )

    assert result.is_file()
    assert result.stat().st_size == metadata.size
