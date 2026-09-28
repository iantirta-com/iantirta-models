# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from iantirta.models.files import (
    CACHEDIR_TAG_CONTENT,
    _get_target_path,
    ensure_file,
    http_download,
)


class _TestHandler(BaseHTTPRequestHandler):
    content = b"hello from server"

    def do_GET(self):
        if self.path == "/file":
            self._send_content()
            return

        if self.path == "/empty":
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        # Simulate:
        # /<repo>/resolve/<revision>/<filename>
        parts = self.path.strip("/").split("/")
        if (
            len(parts) >= 5
            and parts[2] == "resolve"
        ):
            self._send_content()
            return

        # Simulate a redirect from the HF resolve endpoint.
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/file")
            self.end_headers()
            return

        self.send_response(404)
        self.end_headers()

    def _send_content(self):
        self.send_response(200)
        self.send_header(
            "Content-Length",
            str(len(self.content)),
        )
        self.end_headers()
        self.wfile.write(self.content)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def http_server():
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _TestHandler,
    )

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )
    thread.start()

    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.fixture
def cache_home(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "iantirta.models.files.CACHE_HOME",
        tmp_path / "cache",
    )

    return tmp_path / "cache"


def test_http_download(http_server, tmp_path):
    destination = tmp_path / "output.bin"

    result = http_download(
        f"{http_server}/file",
        destination,
        progress=False,
    )

    assert result == destination.resolve()
    assert result.read_bytes() == b"hello from server"


def test_http_download_redirect(
    http_server,
    tmp_path,
):
    destination = tmp_path / "output.bin"

    result = http_download(
        f"{http_server}/redirect",
        destination,
        progress=False,
    )

    assert result == destination.resolve()
    assert result.read_bytes() == b"hello from server"


def test_http_download_creates_parent_directories(
    http_server,
    tmp_path,
):
    destination = (
        tmp_path
        / "nested"
        / "directory"
        / "output.bin"
    )

    result = http_download(
        f"{http_server}/file",
        destination,
        progress=False,
    )

    assert result.exists()
    assert result.read_bytes() == b"hello from server"


def test_http_download_hash(http_server, tmp_path):
    content = b"hello from server"
    digest = hashlib.sha256(content).hexdigest()

    destination = tmp_path / "output.bin"

    http_download(
        f"{http_server}/file",
        destination,
        hash_prefix=digest,
        progress=False,
    )

    assert destination.read_bytes() == content


def test_http_download_hash_prefix(
    http_server,
    tmp_path,
):
    content = b"hello from server"
    digest = hashlib.sha256(content).hexdigest()

    destination = tmp_path / "output.bin"

    http_download(
        f"{http_server}/file",
        destination,
        hash_prefix=digest[:12],
        progress=False,
    )

    assert destination.read_bytes() == content


def test_http_download_hash_mismatch(
    http_server,
    tmp_path,
):
    destination = tmp_path / "output.bin"

    with pytest.raises(ValueError, match="invalid SHA-256 hash"):
        http_download(
            f"{http_server}/file",
            destination,
            hash_prefix="00000000",
            progress=False,
        )

    assert not destination.exists()

    partial_files = list(
        tmp_path.glob("*.partial")
    )
    assert partial_files == []


def test_http_download_empty_hash_prefix(
    http_server,
    tmp_path,
):
    with pytest.raises(
        ValueError,
        match="hash_prefix must not be empty",
    ):
        http_download(
            f"{http_server}/file",
            tmp_path / "output.bin",
            hash_prefix="",
            progress=False,
        )


def test_http_download_invalid_hash_prefix(
    http_server,
    tmp_path,
):
    with pytest.raises(
        ValueError,
        match="hexadecimal",
    ):
        http_download(
            f"{http_server}/file",
            tmp_path / "output.bin",
            hash_prefix="not-hex",
            progress=False,
        )


def test_http_download_http_error(
    http_server,
    tmp_path,
):
    with pytest.raises(Exception):
        http_download(
            f"{http_server}/missing",
            tmp_path / "output.bin",
            progress=False,
        )

    assert not (tmp_path / "output.bin").exists()


def test_ensure_file_nested_filename(
    http_server,
    cache_home,
):
    result = ensure_file(
        "example/model",
        "subdir/config.json",
        endpoint=http_server,
        progress=False,
    )

    # This test requires the HTTP server to accept /subdir/config.json.

    
def test_get_target_path(cache_home):
    result = _get_target_path(
        "config.json",
        "facebook/mms-1b-all",
    )

    assert result == (
        cache_home
        / "models--facebook--mms-1b-all"
        / "main"
        / "config.json"
    )


def test_get_target_path_revision(cache_home):
    result = _get_target_path(
        "config.json",
        "facebook/mms-1b-all",
        revision="refs/tags/v1",
    )

    assert result == (
        cache_home
        / "models--facebook--mms-1b-all"
        / "refs--tags--v1"
        / "config.json"
    )


def test_get_target_path_creates_cache_tag(cache_home):
    _get_target_path(
        "config.json",
        "facebook/test",
    )

    tag = cache_home / "CACHEDIR.TAG"

    assert tag.exists()
    assert tag.read_text(encoding="utf-8") == (
        CACHEDIR_TAG_CONTENT
    )


def test_ensure_file_local_directory(tmp_path):
    file = tmp_path / "config.json"
    file.write_text("{}", encoding="utf-8")

    result = ensure_file(
        tmp_path,
        "config.json",
    )

    assert result == file
    assert result.read_text(encoding="utf-8") == "{}"


def test_ensure_file_local_missing_file(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match="File not found",
    ):
        ensure_file(
            tmp_path,
            "config.json",
        )


def test_ensure_file_local_file_instead_of_directory(
    tmp_path,
):
    file = tmp_path / "model.safetensors"
    file.write_bytes(b"test")

    with pytest.raises(NotADirectoryError):
        ensure_file(
            file,
            "config.json",
        )


def test_ensure_file_requires_filename(tmp_path):
    with pytest.raises(
        ValueError,
        match="filename is required",
    ):
        ensure_file(tmp_path)


def test_ensure_file_remote(
    http_server,
    cache_home,
):
    result = ensure_file(
        "example/model",
        "file",
        endpoint=http_server,
        progress=False,
    )

    assert result.exists()
    assert result.read_bytes() == b"hello from server"


def test_ensure_file_uses_cache(
    http_server,
    cache_home,
):
    first = ensure_file(
        "example/model",
        "file",
        endpoint=http_server,
        progress=False,
    )

    # Remove the server dependency after the first download.
    # The second call must use the cached file.
    cached_content = first.read_bytes()

    first.write_bytes(b"cached")

    second = ensure_file(
        "example/model",
        "file",
        endpoint="http://127.0.0.1:1",
        progress=False,
    )

    assert second == first
    assert second.read_bytes() == b"cached"

    assert cached_content == b"hello from server"


def test_ensure_file_revision_creates_separate_cache(
    http_server,
    cache_home,
):
    first = ensure_file(
        "example/model",
        "file",
        endpoint=http_server,
        revision="main",
        progress=False,
    )

    second = ensure_file(
        "example/model",
        "file",
        endpoint=http_server,
        revision="other",
        progress=False,
    )

    assert first != second
    assert first.exists()
    assert second.exists()


def test_ensure_file_custom_endpoint(
    http_server,
    cache_home,
):
    result = ensure_file(
        "example/model",
        "file",
        endpoint=http_server,
        revision="main",
        progress=False,
    )

    assert result.read_bytes() == b"hello from server"
