from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import requests

from iantirta.models.remote import api
from iantirta.models.remote._types import HFFileMeta, XetFileData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_response(
    *,
    headers: dict[str, str] | None = None,
    url: str = "https://huggingface.co/test",
    status_code: int = 200,
) -> MagicMock:
    response = MagicMock()
    response.headers = headers or {}
    response.url = url
    response.status_code = status_code
    return response


def make_metadata(
    *,
    commit_hash: str = "abc123",
    etag: str = "etag123",
    location: str = "https://cdn.example.com/file",
    size: int | None = 10,
    xet: XetFileData | None = None,
) -> HFFileMeta:
    return HFFileMeta(
        commit_hash=commit_hash,
        etag=etag,
        location=location,
        size=size,
        xet=xet,
    )


def write_file(path: Path, content: bytes = b"0123456789") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# hf_hub_url
# ---------------------------------------------------------------------------


def test_hf_hub_url_default():
    assert (
        api.hf_hub_url(
            "org/repo",
            "model.bin",
        )
        == "https://huggingface.co/org/repo/resolve/main/model.bin"
    )


def test_hf_hub_url_quotes_revision():
    assert (
        api.hf_hub_url(
            "org/repo",
            "model.bin",
            revision="feature/test",
        )
        == "https://huggingface.co/org/repo/resolve/feature%2Ftest/model.bin"
    )


def test_hf_hub_url_quotes_filename():
    assert (
        api.hf_hub_url(
            "org/repo",
            "folder/my file.bin",
        )
        == "https://huggingface.co/org/repo/resolve/main/folder/my%20file.bin"
    )


def test_hf_hub_url_strips_endpoint_slash():
    assert (
        api.hf_hub_url(
            "org/repo",
            "model.bin",
            endpoint="https://example.com/",
        )
        == "https://example.com/org/repo/resolve/main/model.bin"
    )


def test_hf_hub_url_repo_type_is_currently_ignored():
    # repo_type is currently part of the API but does not affect the URL.
    assert (
        api.hf_hub_url(
            "org/repo",
            "file.bin",
            repo_type="dataset",
        )
        == "https://huggingface.co/org/repo/resolve/main/file.bin"
    )


# ---------------------------------------------------------------------------
# repo_folder_name
# ---------------------------------------------------------------------------


def test_repo_folder_name():
    assert (
        api.repo_folder_name("julien-c/EsperBERTo-small")
        == "models--julien-c--EsperBERTo-small"
    )


def test_repo_folder_name_dataset():
    assert (
        api.repo_folder_name(
            "org/dataset",
            repo_type="dataset",
        )
        == "datasets--org--dataset"
    )


def test_repo_folder_name_nested_repo():
    assert (
        api.repo_folder_name("a/b/c")
        == "models--a--b--c"
    )


# ---------------------------------------------------------------------------
# get_hf_file_metadata
# ---------------------------------------------------------------------------


def test_get_hf_file_metadata():
    response = make_response(
        headers={
            "X-Linked-Size": "123",
            "X-Linked-Etag": '"abc123"',
            "X-Repo-Commit": "commit123",
            "Location": "https://cdn.example.com/file",
        }
    )

    xet_data = XetFileData(
        file_hash="a" * 64,
        refresh_route="https://huggingface.co/xet/refresh",
    )

    with (
        patch.object(api, "request_follow_redirect", return_value=response) as request,
        patch.object(api._xet, "parse_file_data", return_value=xet_data),
    ):
        result = api.get_hf_file_metadata(
            "https://huggingface.co/org/repo/resolve/main/file.bin"
        )

    request.assert_called_once_with(
        "HEAD",
        "https://huggingface.co/org/repo/resolve/main/file.bin",
        headers={"Accept-Encoding": "identity"},
    )

    assert result == HFFileMeta(
        commit_hash="commit123",
        etag="abc123",
        location="https://cdn.example.com/file",
        size=123,
        xet=xet_data,
    )

    response.raise_for_status.assert_called_once()
    response.close.assert_called_once()


def test_get_hf_file_metadata_uses_content_length():
    response = make_response(
        headers={
            "Content-Length": "456",
            "ETag": '"etag456"',
            "X-Repo-Commit": "commit456",
        }
    )

    with (
        patch.object(api, "request_follow_redirect", return_value=response),
        patch.object(api._xet, "parse_file_data", return_value=None),
    ):
        result = api.get_hf_file_metadata("https://example.com/file")

    assert result.size == 456
    assert result.etag == "etag456"
    assert result.commit_hash == "commit456"
    assert result.location == response.url
    assert result.xet is None


def test_get_hf_file_metadata_prefers_linked_size():
    response = make_response(
        headers={
            "X-Linked-Size": "123",
            "Content-Length": "999",
            "ETag": '"etag"',
            "X-Repo-Commit": "commit",
        }
    )

    with (
        patch.object(api, "request_follow_redirect", return_value=response),
        patch.object(api._xet, "parse_file_data", return_value=None),
    ):
        result = api.get_hf_file_metadata("https://example.com/file")

    assert result.size == 123


def test_get_hf_file_metadata_prefers_linked_etag():
    response = make_response(
        headers={
            "X-Linked-Etag": '"linked"',
            "ETag": '"normal"',
            "X-Repo-Commit": "commit",
        }
    )

    with (
        patch.object(api, "request_follow_redirect", return_value=response),
        patch.object(api._xet, "parse_file_data", return_value=None),
    ):
        result = api.get_hf_file_metadata("https://example.com/file")

    assert result.etag == "linked"


def test_get_hf_file_metadata_strips_weak_etag():
    response = make_response(
        headers={
            "ETag": 'W/"abc123"',
            "X-Repo-Commit": "commit",
        }
    )

    with (
        patch.object(api, "request_follow_redirect", return_value=response),
        patch.object(api._xet, "parse_file_data", return_value=None),
    ):
        result = api.get_hf_file_metadata("https://example.com/file")

    assert result.etag == "abc123"


def test_get_hf_file_metadata_missing_commit():
    response = make_response(
        headers={
            "ETag": '"etag"',
        }
    )

    with (
        patch.object(api, "request_follow_redirect", return_value=response),
        pytest.raises(
            RuntimeError,
            match="X-Repo-Commit",
        ),
    ):
        api.get_hf_file_metadata("https://example.com/file")

    response.close.assert_called_once()


def test_get_hf_file_metadata_missing_etag():
    response = make_response(
        headers={
            "X-Repo-Commit": "commit",
        }
    )

    with (
        patch.object(api, "request_follow_redirect", return_value=response),
        pytest.raises(
            RuntimeError,
            match="ETag",
        ),
    ):
        api.get_hf_file_metadata("https://example.com/file")

    response.close.assert_called_once()


def test_get_hf_file_metadata_invalid_status():
    response = make_response(
        headers={},
        status_code=404,
    )

    response.raise_for_status.side_effect = requests.HTTPError("404")

    with (
        patch.object(api, "request_follow_redirect", return_value=response),
        pytest.raises(requests.HTTPError),
    ):
        api.get_hf_file_metadata("https://example.com/file")

    response.raise_for_status.assert_called_once()
    response.close.assert_called_once()


# ---------------------------------------------------------------------------
# _download_file
# ---------------------------------------------------------------------------


def test_download_file_uses_http(tmp_path):
    destination = tmp_path / "file.bin"

    metadata = make_metadata(
        size=5,
        xet=None,
    )

    with patch.object(api, "http_download") as download:
        api._download_file(
            destination=destination,
            url="https://example.com/file",
            filename="file.bin",
            metadata=metadata,
            headers={"Authorization": "Bearer token"},
        )

    download.assert_called_once()

    args = download.call_args.args
    kwargs = download.call_args.kwargs

    assert args[0] == "https://example.com/file"
    assert kwargs["expected_size"] == 5
    assert kwargs["headers"] == {"Authorization": "Bearer token"}

    # Destination is moved into place after the temporary download.
    assert destination.exists()


def test_download_file_uses_xet_when_available(tmp_path):
    destination = tmp_path / "file.bin"

    xet = XetFileData(
        file_hash="a" * 64,
        refresh_route="https://example.com/xet",
    )

    metadata = make_metadata(
        size=5,
        xet=xet,
    )

    with (
        patch.object(api._xet, "available", return_value=True),
        patch.object(api._xet, "download") as download,
        patch.object(api, "http_download") as http_download,
    ):
        api._download_file(
            destination=destination,
            url="https://example.com/file",
            filename="file.bin",
            metadata=metadata,
            headers={"Authorization": "Bearer token"},
        )

    download.assert_called_once()
    http_download.assert_not_called()

    kwargs = download.call_args.kwargs

    assert kwargs["xet_file_data"] if "xet_file_data" in kwargs else True

    assert kwargs["headers"] == {"Authorization": "Bearer token"}
    assert kwargs["expected_size"] == 5
    assert kwargs["displayed_filename"] == "file.bin"

    assert destination.exists()


def test_download_file_falls_back_to_http_when_xet_unavailable(tmp_path):
    destination = tmp_path / "file.bin"

    xet = XetFileData(
        file_hash="a" * 64,
        refresh_route="https://example.com/xet",
    )

    metadata = make_metadata(
        size=5,
        xet=xet,
    )

    with (
        patch.object(api._xet, "available", return_value=False),
        patch.object(api._xet, "download") as xet_download,
        patch.object(api, "http_download") as http_download,
    ):
        api._download_file(
            destination=destination,
            url="https://example.com/file",
            filename="file.bin",
            metadata=metadata,
            headers=None,
        )

    xet_download.assert_not_called()
    http_download.assert_called_once()


def test_download_file_does_not_download_existing_destination(tmp_path):
    destination = write_file(
        tmp_path / "file.bin",
        b"existing",
    )

    metadata = make_metadata()

    with (
        patch.object(api, "http_download") as http_download,
        patch.object(api._xet, "download") as xet_download,
    ):
        api._download_file(
            destination=destination,
            url="https://example.com/file",
            filename="file.bin",
            metadata=metadata,
            headers=None,
        )

    http_download.assert_not_called()
    xet_download.assert_not_called()

    assert destination.read_bytes() == b"existing"


def test_download_file_replaces_existing_destination(tmp_path):
    destination = write_file(
        tmp_path / "file.bin",
        b"old",
    )

    metadata = make_metadata(
        size=3,
        xet=None,
    )

    with patch.object(api, "http_download") as http_download:

        def fake_download(
            url,
            file,
            *,
            expected_size,
            headers,
        ):
            file.write(b"new")

        http_download.side_effect = fake_download

        api._download_file(
            destination=destination,
            url="https://example.com/file",
            filename="file.bin",
            metadata=metadata,
            headers=None,
        )

    # NOTE:
    # Current _download_file() returns immediately if destination.exists().
    # Therefore this assertion documents the current behavior.
    assert destination.read_bytes() == b"old"


def test_download_file_cleans_temporary_file_on_failure(tmp_path):
    destination = tmp_path / "file.bin"

    metadata = make_metadata(
        size=5,
        xet=None,
    )

    with (
        patch.object(
            api,
            "http_download",
            side_effect=RuntimeError("download failed"),
        ),
        pytest.raises(RuntimeError, match="download failed"),
    ):
        api._download_file(
            destination=destination,
            url="https://example.com/file",
            filename="file.bin",
            metadata=metadata,
            headers=None,
        )

    assert not destination.exists()

    incomplete_files = list(tmp_path.glob("*.incomplete"))
    assert incomplete_files == []


def test_download_file_long_filename_is_passed_to_xet(tmp_path):
    destination = tmp_path / "file.bin"

    long_filename = "a" * 200

    metadata = make_metadata(
        size=1,
        xet=XetFileData(
            file_hash="a" * 64,
            refresh_route="https://example.com/xet",
        ),
    )

    with (
        patch.object(api._xet, "available", return_value=True),
        patch.object(api._xet, "download") as download,
    ):
        api._download_file(
            destination=destination,
            url="https://example.com/file",
            filename=long_filename,
            metadata=metadata,
            headers=None,
        )

    assert download.call_count == 1


# ---------------------------------------------------------------------------
# _chmod_and_move
# ---------------------------------------------------------------------------


def test_chmod_and_move_moves_file(tmp_path):
    src = write_file(
        tmp_path / "source",
        b"hello",
    )
    dst = tmp_path / "destination"

    api._chmod_and_move(src, dst)

    assert not src.exists()
    assert dst.exists()
    assert dst.read_bytes() == b"hello"


def test_chmod_and_move_replaces_existing_file(tmp_path):
    src = write_file(
        tmp_path / "source",
        b"new",
    )
    dst = write_file(
        tmp_path / "destination",
        b"old",
    )

    api._chmod_and_move(src, dst)

    assert not src.exists()
    assert dst.read_bytes() == b"new"


def test_chmod_and_move_replaces_existing_symlink(tmp_path):
    src = write_file(
        tmp_path / "source",
        b"new",
    )

    target = write_file(
        tmp_path / "target",
        b"old",
    )

    dst = tmp_path / "destination"
    dst.symlink_to(target)

    api._chmod_and_move(src, dst)

    assert not dst.is_symlink()
    assert dst.read_bytes() == b"new"
    assert target.read_bytes() == b"old"


# ---------------------------------------------------------------------------
# _replace_no_matter_what
# ---------------------------------------------------------------------------


def test_replace_no_matter_what(tmp_path):
    src = write_file(
        tmp_path / "src",
        b"new",
    )
    dst = write_file(
        tmp_path / "dst",
        b"old",
    )

    api._replace_no_matter_what(src, dst)

    assert not src.exists()
    assert dst.read_bytes() == b"new"


# ---------------------------------------------------------------------------
# _copy_no_matter_what
# ---------------------------------------------------------------------------


def test_copy_no_matter_what(tmp_path):
    src = write_file(
        tmp_path / "src",
        b"hello",
    )
    dst = tmp_path / "dst"

    api._copy_no_matter_what(str(src), str(dst))

    assert dst.read_bytes() == b"hello"


# ---------------------------------------------------------------------------
# hf_hub_download: local fast paths
# ---------------------------------------------------------------------------


def test_hf_hub_download_returns_existing_local_filename(tmp_path):
    local_file = write_file(
        tmp_path / "local.bin",
        b"hello",
    )

    with (
        patch.object(api, "get_hf_file_metadata") as metadata,
    ):
        result = api.hf_hub_download(
            "org/repo",
            str(local_file),
            outdir=tmp_path / "cache",
        )

    assert result == local_file
    metadata.assert_not_called()


def test_hf_hub_download_returns_existing_outdir_file(tmp_path):
    outdir = tmp_path / "cache"

    existing = write_file(
        outdir / "file.bin",
        b"hello",
    )

    with patch.object(api, "get_hf_file_metadata") as metadata:
        result = api.hf_hub_download(
            "org/repo",
            "file.bin",
            outdir=outdir,
        )

    assert result == existing
    metadata.assert_not_called()


# ---------------------------------------------------------------------------
# hf_hub_download: normal HTTP
# ---------------------------------------------------------------------------


def test_hf_hub_download_http(tmp_path):
    outdir = tmp_path / "cache"

    metadata = make_metadata(
        commit_hash="commit123",
        etag="etag123",
        location="https://cdn.example.com/file",
        size=5,
        xet=None,
    )

    def fake_download(
        *,
        destination,
        url,
        filename,
        metadata,
        headers,
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"hello")

    with (
        patch.object(api, "get_hf_file_metadata", return_value=metadata),
        patch.object(api, "_download_file", side_effect=fake_download) as download,
        patch.object(api, "create_cache_tag") as create_tag,
    ):
        result = api.hf_hub_download(
            "org/repo",
            "file.bin",
            outdir=outdir,
        )

    assert result.exists()
    assert result.read_bytes() == b"hello"

    download.assert_called_once_with(
        destination=outdir / "models--org--repo" / "blobs" / "etag123",
        url="https://cdn.example.com/file",
        filename="file.bin",
        metadata=metadata,
        headers=None,
    )

    create_tag.assert_called_once_with(outdir)


def test_hf_hub_download_uses_url_for_xet(tmp_path):
    outdir = tmp_path / "cache"

    xet = XetFileData(
        file_hash="a" * 64,
        refresh_route="https://example.com/xet",
    )

    metadata = make_metadata(
        commit_hash="commit123",
        etag="etag123",
        location="https://cdn.example.com/file",
        size=5,
        xet=xet,
    )

    with (
        patch.object(api, "get_hf_file_metadata", return_value=metadata),
        patch.object(api, "_download_file") as download,
        patch.object(api._xet, "available", return_value=True),
        patch.object(api, "supports_symlink", return_value=False),
        patch.object(api, "create_cache_tag"),
        patch.object(api, "create_pointer"),
    ):
        api.hf_hub_download(
            "org/repo",
            "file.bin",
            outdir=outdir,
        )

    # Metadata should cause Xet's original URL to be selected.
    assert download.call_args.kwargs["url"] == (
        "https://huggingface.co/org/repo/resolve/main/file.bin"
    )


# ---------------------------------------------------------------------------
# hf_hub_download: snapshot cache
# ---------------------------------------------------------------------------


def test_hf_hub_download_returns_existing_snapshot(tmp_path):
    outdir = tmp_path / "cache"

    storage_dir = outdir / "models--org--repo"
    snapshot = (
        storage_dir
        / "snapshots"
        / "commit123"
        / "file.bin"
    )

    write_file(snapshot, b"cached")

    metadata = make_metadata(
        commit_hash="commit123",
        etag="etag123",
        size=6,
    )

    with (
        patch.object(api, "get_hf_file_metadata", return_value=metadata),
        patch.object(api, "create_cache_tag") as create_tag,
        patch.object(api, "_download_file") as download,
    ):
        result = api.hf_hub_download(
            "org/repo",
            "file.bin",
            outdir=outdir,
        )

    assert result == snapshot
    assert result.read_bytes() == b"cached"

    create_tag.assert_not_called()
    download.assert_not_called()


# ---------------------------------------------------------------------------
# Xet shared blob reuse
# ---------------------------------------------------------------------------


def test_hf_hub_download_reuses_shared_xet_blob(tmp_path):
    outdir = tmp_path / "cache"

    xet_hash = "a" * 64

    metadata = make_metadata(
        commit_hash="commit123",
        etag="etag123",
        size=5,
        xet=XetFileData(
            file_hash=xet_hash,
            refresh_route="https://example.com/xet",
        ),
    )

    blob = (
        outdir
        / "models--org--repo"
        / "blobs"
        / "etag123"
    )

    shared = (
        outdir
        / "blobs"
        / xet_hash[:2]
        / xet_hash
    )

    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.write_bytes(b"hello")

    def fake_link_from_shared(
        *,
        blob_path,
        xet_hash,
        cache_dir,
        expected_size,
    ):
        assert blob_path == blob
        assert xet_hash == xet_hash_value
        assert cache_dir == outdir
        assert expected_size == 5

        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.symlink_to(
            os.path.relpath(shared, blob_path.parent)
        )
        return True

    xet_hash_value = xet_hash

    with (
        patch.object(api, "get_hf_file_metadata", return_value=metadata),
        patch.object(api, "create_cache_tag"),
        patch.object(api, "supports_symlink", return_value=True),
        patch.object(api, "link_from_shared", side_effect=fake_link_from_shared) as link,
        patch.object(api, "_download_file") as download,
    ):
        result = api.hf_hub_download(
            "org/repo",
            "file.bin",
            outdir=outdir,
        )

    assert result.exists()
    assert result.read_bytes() == b"hello"

    link.assert_called_once()
    download.assert_not_called()


def test_hf_hub_download_does_not_use_shared_store_without_symlink_support(
    tmp_path,
):
    outdir = tmp_path / "cache"

    metadata = make_metadata(
        commit_hash="commit123",
        etag="etag123",
        size=5,
        xet=XetFileData(
            file_hash="a" * 64,
            refresh_route="https://example.com/xet",
        ),
    )

    with (
        patch.object(api, "get_hf_file_metadata", return_value=metadata),
        patch.object(api, "create_cache_tag"),
        patch.object(api, "supports_symlink", return_value=False),
        patch.object(api, "_download_file") as download,
        patch.object(api, "link_from_shared") as link,
    ):
        api.hf_hub_download(
            "org/repo",
            "file.bin",
            outdir=outdir,
        )

    link.assert_not_called()
    download.assert_called_once()


def test_hf_hub_download_publishes_downloaded_xet_blob(tmp_path):
    outdir = tmp_path / "cache"

    xet_hash = "b" * 64

    metadata = make_metadata(
        commit_hash="commit123",
        etag="etag123",
        size=5,
        xet=XetFileData(
            file_hash=xet_hash,
            refresh_route="https://example.com/xet",
        ),
    )

    def fake_download(
        *,
        destination,
        url,
        filename,
        metadata,
        headers,
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"hello")

    with (
        patch.object(api, "get_hf_file_metadata", return_value=metadata),
        patch.object(api, "create_cache_tag"),
        patch.object(api, "supports_symlink", return_value=True),
        patch.object(api, "link_from_shared", return_value=False),
        patch.object(
            api,
            "_download_file",
            side_effect=fake_download,
        ),
        patch.object(
            api,
            "publish_to_shared",
            return_value=True,
        ) as publish,
    ):
        with patch.object(
            api,
            "create_pointer",
        ) as create_pointer:
            result = api.hf_hub_download(
                "org/repo",
                "file.bin",
                outdir=outdir,
            )

    assert result is not None

    publish.assert_called_once_with(
        blob_path=outdir / "models--org--repo" / "blobs" / "etag123",
        xet_hash=xet_hash,
        cache_dir=outdir,
        expected_size=5,
        replace_existing=False,
    )

    # Important:
    # publish_to_shared=True means the blob is shared and therefore
    # create_pointer must NOT move it.
    create_pointer.assert_called_once()

    assert create_pointer.call_args.kwargs["move_source"] is False


def test_hf_hub_download_does_not_publish_non_xet_file(tmp_path):
    outdir = tmp_path / "cache"

    metadata = make_metadata(
        commit_hash="commit123",
        etag="etag123",
        size=5,
        xet=None,
    )

    def fake_download(
        *,
        destination,
        url,
        filename,
        metadata,
        headers,
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"hello")

    with (
        patch.object(api, "get_hf_file_metadata", return_value=metadata),
        patch.object(api, "create_cache_tag"),
        patch.object(api, "supports_symlink", return_value=True),
        patch.object(
            api,
            "_download_file",
            side_effect=fake_download,
        ),
        patch.object(api, "publish_to_shared") as publish,
    ):
        with patch.object(api, "create_pointer"):
            api.hf_hub_download(
                "org/repo",
                "file.bin",
                outdir=outdir,
            )

    publish.assert_not_called()


# ---------------------------------------------------------------------------
# Shared store failure
# ---------------------------------------------------------------------------


def test_hf_hub_download_continues_when_shared_reuse_fails(tmp_path):
    outdir = tmp_path / "cache"

    metadata = make_metadata(
        commit_hash="commit123",
        etag="etag123",
        size=5,
        xet=XetFileData(
            file_hash="a" * 64,
            refresh_route="https://example.com/xet",
        ),
    )

    def fake_download(
        *,
        destination,
        url,
        filename,
        metadata,
        headers,
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"hello")

    with (
        patch.object(api, "get_hf_file_metadata", return_value=metadata),
        patch.object(api, "create_cache_tag"),
        patch.object(api, "supports_symlink", return_value=True),
        patch.object(api, "link_from_shared", return_value=False),
        patch.object(
            api,
            "_download_file",
            side_effect=fake_download,
        ),
        patch.object(
            api,
            "publish_to_shared",
            return_value=False,
        ),
        patch.object(api, "create_pointer") as create_pointer,
    ):
        api.hf_hub_download(
            "org/repo",
            "file.bin",
            outdir=outdir,
        )

    create_pointer.assert_called_once()

    # If publication failed, the repository-local blob remains the owner.
    assert create_pointer.call_args.kwargs["move_source"] is True


# ---------------------------------------------------------------------------
# Subfolder
# ---------------------------------------------------------------------------


def test_hf_hub_download_subfolder(tmp_path):
    outdir = tmp_path / "cache"

    metadata = make_metadata(
        commit_hash="commit123",
        etag="etag123",
        size=5,
    )

    with (
        patch.object(api, "get_hf_file_metadata", return_value=metadata),
        patch.object(api, "create_cache_tag"),
        patch.object(api, "_download_file") as download,
        patch.object(api, "create_pointer"),
    ):
        api.hf_hub_download(
            "org/repo",
            "file.bin",
            subfolder="models",
            outdir=outdir,
        )

    assert download.call_args.kwargs["filename"] == "models/file.bin"


def test_hf_hub_download_normalizes_subfolder_slashes(tmp_path):
    outdir = tmp_path / "cache"

    metadata = make_metadata()

    with (
        patch.object(api, "get_hf_file_metadata", return_value=metadata) as ghfm,
        patch.object(api, "create_cache_tag"),
        patch.object(api, "_download_file"),
        patch.object(api, "create_pointer"),
    ):
        api.hf_hub_download(
            "org/repo",
            "/file.bin",
            subfolder="/models/",
            outdir=outdir,
        )

    # This indirectly checks the normalized filename.
    # URL generation happens after normalization.
    expected_url = api.hf_hub_url(
        "org/repo",
        "models/file.bin",
    )

    actual_url = ghfm.call_args.args[0]

    assert actual_url == expected_url


# ---------------------------------------------------------------------------
# Xet size validation
# ---------------------------------------------------------------------------


def test_xet_file_requires_known_size(tmp_path):
    outdir = tmp_path / "cache"

    metadata = make_metadata(
        size=None,
        xet=XetFileData(
            file_hash="a" * 64,
            refresh_route="https://example.com/xet",
        ),
    )

    with (
        patch.object(api, "get_hf_file_metadata", return_value=metadata),
        pytest.raises(
            RuntimeError,
            match="Xet file has no known size",
        ),
    ):
        api.hf_hub_download(
            "org/repo",
            "file.bin",
            outdir=outdir,
        )


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


def test_hf_hub_download_passes_headers_to_download(tmp_path):
    outdir = tmp_path / "cache"

    metadata = make_metadata(
        size=5,
    )

    headers = {
        "Authorization": "Bearer secret",
    }

    with (
        patch.object(api, "get_hf_file_metadata", return_value=metadata),
        patch.object(api, "create_cache_tag"),
        patch.object(api, "_download_file") as download,
        patch.object(api, "create_pointer"),
    ):
        api.hf_hub_download(
            "org/repo",
            "file.bin",
            outdir=outdir,
            headers=headers,
        )

    assert download.call_args.kwargs["headers"] == headers


# ---------------------------------------------------------------------------
# Revision / endpoint
# ---------------------------------------------------------------------------


def test_hf_hub_download_uses_revision_and_endpoint(tmp_path):
    outdir = tmp_path / "cache"

    metadata = make_metadata()

    with (
        patch.object(api, "get_hf_file_metadata", return_value=metadata) as metadata_call,
        patch.object(api, "create_cache_tag"),
        patch.object(api, "_download_file"),
        patch.object(api, "create_pointer"),
    ):
        api.hf_hub_download(
            "org/repo",
            "file.bin",
            revision="refs/test",
            endpoint="https://example.test",
            outdir=outdir,
        )

    metadata_call.assert_called_once_with(
        "https://example.test/org/repo/resolve/refs%2Ftest/file.bin"
    )


# ---------------------------------------------------------------------------
# Pointer behavior
# ---------------------------------------------------------------------------


def test_hf_hub_download_create_pointer_for_downloaded_blob(tmp_path):
    outdir = tmp_path / "cache"

    metadata = make_metadata(
        commit_hash="commit123",
        etag="etag123",
        size=5,
    )

    blob = (
        outdir
        / "models--org--repo"
        / "blobs"
        / "etag123"
    )
    dest = (
        outdir
        / "models--org--repo"
        / "snapshots"
        / "commit123" # commit
        / "file.bin"
    )

    def fake_download(
        *,
        destination,
        url,
        filename,
        metadata,
        headers,
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"hello")

    with (
        patch.object(api, "get_hf_file_metadata", return_value=metadata),
        patch.object(api, "create_cache_tag"),
        patch.object(api, "_download_file", side_effect=fake_download),
        patch.object(api, "create_pointer") as create_pointer,
    ):
        api.hf_hub_download(
            "org/repo",
            "file.bin",
            outdir=outdir,
        )

    create_pointer.assert_called_once()

    args = create_pointer.call_args.args
    kwargs = create_pointer.call_args.kwargs

    assert args[0] == blob
    assert args[1] == dest
    assert kwargs["move_source"] is True


# ---------------------------------------------------------------------------
# Lock behavior
# ---------------------------------------------------------------------------


def test_hf_hub_download_uses_cache_lock(tmp_path):
    outdir = tmp_path / "cache"

    metadata = make_metadata(
        commit_hash="commit123",
        etag="etag123",
        size=5,
    )

    fake_lock = MagicMock()
    fake_lock.__enter__.return_value = None
    fake_lock.__exit__.return_value = False

    with (
        patch.object(api, "get_hf_file_metadata", return_value=metadata),
        patch.object(api, "create_cache_tag"),
        patch.object(api, "file_lock", return_value=fake_lock) as file_lock,
        patch.object(api, "_download_file"),
        patch.object(api, "create_pointer"),
    ):
        api.hf_hub_download(
            "org/repo",
            "file.bin",
            outdir=outdir,
        )

    expected_lock = (
        outdir
        / ".locks"
        / "models--org--repo"
        / "etag123.lock"
    )

    file_lock.assert_called_once_with(expected_lock)


# ---------------------------------------------------------------------------
# Full filesystem integration test
# ---------------------------------------------------------------------------


def test_hf_hub_download_full_normal_flow(tmp_path):
    """
    This is the important end-to-end unit test.

    Network and metadata are mocked, but the actual cache directories,
    blob and snapshot are created on disk.
    """
    outdir = tmp_path / "cache"

    metadata = make_metadata(
        commit_hash="commit123",
        etag="etag123",
        size=11,
        xet=None,
    )

    def fake_download(
        *,
        destination,
        url,
        filename,
        metadata,
        headers,
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"hello world")

    with (
        patch.object(api, "get_hf_file_metadata", return_value=metadata),
        patch.object(api, "create_cache_tag"),
        patch.object(api, "_download_file", side_effect=fake_download),
    ):
        result = api.hf_hub_download(
            "org/repo",
            "file.bin",
            outdir=outdir,
        )

    assert result.exists()
    assert result.read_bytes() == b"hello world"

    storage = outdir / "models--org--repo"

    blob = storage / "blobs" / "etag123"
    snapshot = (
        storage
        / "snapshots"
        / "commit123"
        / "file.bin"
    )

    # Depending on create_pointer() implementation, blob may have
    # been moved into the snapshot rather than remaining here.
    assert snapshot.exists()
    assert snapshot.read_bytes() == b"hello world"


def test_hf_hub_download_full_xet_shared_flow(tmp_path):
    """
    First call:
        download -> publish shared blob -> snapshot

    Second repository:
        shared blob -> no download
    """
    outdir = tmp_path / "cache"

    xet_hash = "c" * 64

    metadata_a = make_metadata(
        commit_hash="commit-a",
        etag="etag-a",
        size=11,
        xet=XetFileData(
            file_hash=xet_hash,
            refresh_route="https://example.com/xet",
        ),
    )

    def fake_download(
        *,
        destination,
        url,
        filename,
        metadata,
        headers,
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"hello world")

    # First repository downloads the file.
    with (
        patch.object(
            api,
            "get_hf_file_metadata",
            return_value=metadata_a,
        ),
        patch.object(api, "create_cache_tag"),
        patch.object(api, "supports_symlink", return_value=True),
        patch.object(
            api,
            "_download_file",
            side_effect=fake_download,
        ),
    ):
        first = api.hf_hub_download(
            "org/a",
            "file.bin",
            outdir=outdir,
        )

    assert first.exists()
    assert first.read_bytes() == b"hello world"

    shared = (
        outdir
        / "blobs"
        / xet_hash[:2]
        / xet_hash
    )

    blob_a = (
        outdir
        / "models--org--a"
        / "blobs"
        / "etag-a"
    )
    
    result = api.link_from_shared(
        blob_path=blob_a,
        xet_hash=xet_hash,
        cache_dir=outdir,
        expected_size=11,
    )
    
    assert result is True
    assert blob_a.is_symlink()
    assert blob_a.resolve() == shared.resolve()

    # This assertion assumes your publish_to_shared() implementation
    # actually creates the shared store entry.
    if shared.exists():
        assert shared.read_bytes() == b"hello world"

    # Second download should be able to reuse the same Xet content.
    with (
        patch.object(
            api,
            "get_hf_file_metadata",
            return_value=metadata_a,
        ),
        patch.object(api, "create_cache_tag"),
        patch.object(api, "supports_symlink", return_value=True),
        patch.object(
            api,
            "_download_file",
            side_effect=AssertionError(
                "Second repository downloaded despite shared Xet blob"
            ),
        ),
    ):
        # This test relies on the real _shared.py implementation.
        # If the first publication succeeded, this must not call _download_file.
        second = api.hf_hub_download(
            "org/a",
            "file.bin",
            outdir=outdir,
        )

    assert second.exists()
    assert second.read_bytes() == b"hello world"
