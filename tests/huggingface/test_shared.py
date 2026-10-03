from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from iantirta.models.remote import _shared as shared


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


XET_HASH = "c" * 64
PAYLOAD = b"hello world"
PAYLOAD_SIZE = len(PAYLOAD)


def repo_blob(
    cache_dir: Path,
    repo: str,
    etag: str,
) -> Path:
    path = (
        cache_dir
        / f"models--{repo}"
        / "blobs"
        / etag
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def shared_path(
    cache_dir: Path,
    xet_hash: str = XET_HASH,
) -> Path:
    return (
        cache_dir
        / "blobs"
        / xet_hash[:2]
        / xet_hash
    )


def write_blob(
    path: Path,
    data: bytes = PAYLOAD,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def initialize_shared_store(cache_dir: Path) -> Path:
    store = cache_dir / "blobs"
    store.mkdir(parents=True, exist_ok=True)

    assert shared._ensure_shared_blobs_dir(cache_dir) is True
    assert shared.is_shared_blobs_dir(store)

    return store


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def test_shared_blobs_dir(tmp_path: Path):
    cache_dir = tmp_path / "cache"

    assert shared.shared_blobs_dir(cache_dir) == (
        cache_dir / "blobs"
    )


def test_shared_blob_path(tmp_path: Path):
    cache_dir = tmp_path / "cache"

    expected = (
        cache_dir
        / "blobs"
        / "cc"
        / XET_HASH
    )

    assert shared.shared_blob_path(
        cache_dir,
        XET_HASH,
    ) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "c",
        "c" * 63,
        "c" * 65,
        "C" * 64,
        "g" * 64,
        "c" * 63 + "g",
        "../" + "c" * 64,
        "c" * 64 + "\n",
    ],
)
def test_shared_blob_path_rejects_invalid_hash(
    tmp_path: Path,
    value: str,
):
    with pytest.raises(ValueError):
        shared.shared_blob_path(
            tmp_path / "cache",
            value,
        )


# ---------------------------------------------------------------------------
# Shared store marker
# ---------------------------------------------------------------------------


def test_shared_store_is_not_initialized_by_default(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    store = cache_dir / "blobs"

    assert not store.exists()
    assert shared.is_shared_blobs_dir(store) is False


def test_ensure_shared_blobs_dir_creates_store(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)

    result = shared._ensure_shared_blobs_dir(cache_dir)

    assert result is True

    store = cache_dir / "blobs"
    marker = store / shared.SHARED_BLOBS_MARKER_NAME

    assert store.is_dir()
    assert marker.is_file()
    assert marker.read_text() == "1\n"

    assert shared.is_shared_blobs_dir(store) is True


def test_ensure_shared_blobs_dir_is_idempotent(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    assert shared._ensure_shared_blobs_dir(cache_dir) is True
    assert shared._ensure_shared_blobs_dir(cache_dir) is True

    store = cache_dir / "blobs"
    marker = store / shared.SHARED_BLOBS_MARKER_NAME

    assert marker.read_text() == "1\n"


def test_unmarked_store_is_rejected_when_not_empty(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    store = cache_dir / "blobs"

    store.mkdir(parents=True)
    (store / "foreign-file").write_text("foreign")

    assert shared._ensure_shared_blobs_dir(cache_dir) is False
    assert shared.is_shared_blobs_dir(store) is False


def test_wrong_marker_version_is_rejected(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    store = cache_dir / "blobs"

    store.mkdir(parents=True)
    marker = store / shared.SHARED_BLOBS_MARKER_NAME
    marker.write_text("999\n")

    assert shared.is_shared_blobs_dir(store) is False


def test_invalid_marker_contents_are_rejected(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    store = cache_dir / "blobs"

    store.mkdir(parents=True)
    marker = store / shared.SHARED_BLOBS_MARKER_NAME
    marker.write_text("not-valid\n")

    assert shared.is_shared_blobs_dir(store) is False


# ---------------------------------------------------------------------------
# Repository path validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "repo_dir",
    [
        "models--org--repo",
        "datasets--org--repo",
        "spaces--org--repo",
        "kernels--org--repo",
    ],
)
def test_relative_blob_path_accepts_supported_repo(
    tmp_path: Path,
    repo_dir: str,
):
    cache_dir = tmp_path / "cache"
    blob = (
        cache_dir
        / repo_dir
        / "blobs"
        / "etag"
    )

    assert shared._relative_blob_path(
        blob,
        cache_dir,
    ) == f"{repo_dir}/blobs/etag"


@pytest.mark.parametrize(
    "blob",
    [
        "blob",
        "models--org--repo/file",
        "models--org--repo/blobs",
        "models--org--repo/blobs/a/b",
        "random--org--repo/blobs/etag",
        "models--/blobs/etag",
        "models--org--repo/other/etag",
    ],
)
def test_relative_blob_path_rejects_invalid_paths(
    tmp_path: Path,
    blob: str,
):
    cache_dir = tmp_path / "cache"

    assert shared._relative_blob_path(
        cache_dir / blob,
        cache_dir,
    ) is None


def test_relative_blob_path_rejects_newline(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"

    blob = (
        cache_dir
        / "models--org--repo"
        / "blobs"
        / "etag\n"
    )

    assert shared._relative_blob_path(
        blob,
        cache_dir,
    ) is None


# ---------------------------------------------------------------------------
# Store entry validation
# ---------------------------------------------------------------------------


def test_store_entry_requires_known_size(
    tmp_path: Path,
):
    path = tmp_path / "blob"
    write_blob(path)

    assert shared._is_usable_store_entry(
        path,
        None,
    ) is False


def test_store_entry_accepts_correct_file(
    tmp_path: Path,
):
    path = tmp_path / "blob"
    write_blob(path)

    assert shared._is_usable_store_entry(
        path,
        PAYLOAD_SIZE,
    ) is True


def test_store_entry_rejects_wrong_size(
    tmp_path: Path,
):
    path = tmp_path / "blob"
    write_blob(path)

    assert shared._is_usable_store_entry(
        path,
        PAYLOAD_SIZE + 1,
    ) is False


def test_store_entry_rejects_directory(
    tmp_path: Path,
):
    path = tmp_path / "blob"
    path.mkdir()

    assert shared._is_usable_store_entry(
        path,
        PAYLOAD_SIZE,
    ) is False


def test_store_entry_rejects_symlink(
    tmp_path: Path,
):
    real = tmp_path / "real"
    link = tmp_path / "link"

    write_blob(real)
    link.symlink_to(real)

    assert shared._is_usable_store_entry(
        link,
        PAYLOAD_SIZE,
    ) is False


# ---------------------------------------------------------------------------
# Prefix directory
# ---------------------------------------------------------------------------


def test_ensure_prefix_dir_creates_store_and_prefix(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    prefix = shared._ensure_prefix_dir(
        cache_dir,
        XET_HASH,
    )

    assert prefix is not None
    assert prefix == cache_dir / "blobs" / "cc"
    assert prefix.is_dir()

    assert shared.is_shared_blobs_dir(
        cache_dir / "blobs"
    )


def test_ensure_prefix_dir_is_idempotent(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    first = shared._ensure_prefix_dir(
        cache_dir,
        XET_HASH,
    )

    second = shared._ensure_prefix_dir(
        cache_dir,
        XET_HASH,
    )

    assert first == second
    assert first is not None
    assert first.is_dir()


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------


def test_publish_to_shared_creates_shared_blob(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    blob = repo_blob(
        cache_dir,
        "org--repo",
        "etag-a",
    )
    write_blob(blob)

    result = shared.publish_to_shared(
        blob_path=blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    )

    assert result is True

    store = cache_dir / "blobs"
    shared_file = shared_path(cache_dir)

    assert shared.is_shared_blobs_dir(store)
    assert shared_file.is_file()
    assert not shared_file.is_symlink()

    assert shared_file.read_bytes() == PAYLOAD
    assert blob.is_symlink()
    assert blob.resolve() == shared_file.resolve()
    assert blob.read_bytes() == PAYLOAD


def test_publish_creates_manifest(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    blob = repo_blob(
        cache_dir,
        "org--repo",
        "etag-a",
    )
    write_blob(blob)

    assert shared.publish_to_shared(
        blob_path=blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    )

    manifest = shared_path(cache_dir).with_name(
        f"{XET_HASH}.refs"
    )

    assert manifest.is_file()
    assert manifest.read_text() == (
        "models--org--repo/blobs/etag-a\n"
    )


def test_publish_reuses_existing_shared_blob(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    first_blob = repo_blob(
        cache_dir,
        "org--a",
        "etag-a",
    )
    write_blob(first_blob, PAYLOAD)

    assert shared.publish_to_shared(
        blob_path=first_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    )

    shared_file = shared_path(cache_dir)

    second_blob = repo_blob(
        cache_dir,
        "org--b",
        "etag-b",
    )
    write_blob(second_blob, PAYLOAD)

    assert shared.publish_to_shared(
        blob_path=second_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    )

    assert shared_file.read_bytes() == PAYLOAD

    assert first_blob.is_symlink()
    assert second_blob.is_symlink()

    assert first_blob.resolve() == shared_file.resolve()
    assert second_blob.resolve() == shared_file.resolve()


def test_publish_does_not_replace_existing_shared_blob(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    first_blob = repo_blob(
        cache_dir,
        "org--a",
        "etag-a",
    )
    write_blob(first_blob, PAYLOAD)

    assert shared.publish_to_shared(
        blob_path=first_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    )

    shared_file = shared_path(cache_dir)

    original = shared_file.read_bytes()

    second_payload = b"REPLACEMENT"

    assert len(second_payload) == PAYLOAD_SIZE

    second_blob = repo_blob(
        cache_dir,
        "org--b",
        "etag-b",
    )
    write_blob(second_blob, second_payload)

    assert shared.publish_to_shared(
        blob_path=second_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    )

    assert shared_file.read_bytes() == original
    assert second_blob.is_symlink()
    assert second_blob.read_bytes() == original


def test_publish_can_replace_existing_shared_blob(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    first_blob = repo_blob(
        cache_dir,
        "org--a",
        "etag-a",
    )
    write_blob(first_blob, PAYLOAD)

    assert shared.publish_to_shared(
        blob_path=first_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    )

    shared_file = shared_path(cache_dir)

    replacement = b"REPLACEMENT"

    assert len(replacement) == PAYLOAD_SIZE

    second_blob = repo_blob(
        cache_dir,
        "org--b",
        "etag-b",
    )
    write_blob(second_blob, replacement)

    assert shared.publish_to_shared(
        blob_path=second_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
        replace_existing=True,
    )

    assert shared_file.read_bytes() == replacement
    assert second_blob.is_symlink()
    assert second_blob.read_bytes() == replacement


def test_publish_rejects_unknown_size(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    blob = repo_blob(
        cache_dir,
        "org--repo",
        "etag-a",
    )
    write_blob(blob)

    assert shared.publish_to_shared(
        blob_path=blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=None,
    ) is False

    assert blob.is_file()
    assert not blob.is_symlink()


def test_publish_rejects_invalid_hash(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    blob = repo_blob(
        cache_dir,
        "org--repo",
        "etag-a",
    )
    write_blob(blob)

    assert shared.publish_to_shared(
        blob_path=blob,
        xet_hash="invalid",
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    ) is False

    assert blob.is_file()
    assert not blob.is_symlink()


# ---------------------------------------------------------------------------
# Link from shared
# ---------------------------------------------------------------------------


def test_link_from_shared_reuses_existing_payload(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    source_blob = repo_blob(
        cache_dir,
        "org--a",
        "etag-a",
    )
    write_blob(source_blob)

    assert shared.publish_to_shared(
        blob_path=source_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    )

    target_blob = repo_blob(
        cache_dir,
        "org--b",
        "etag-b",
    )

    result = shared.link_from_shared(
        blob_path=target_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    )

    assert result is True
    assert target_blob.is_symlink()

    shared_file = shared_path(cache_dir)

    assert target_blob.resolve() == shared_file.resolve()
    assert target_blob.read_bytes() == PAYLOAD


def test_link_from_shared_updates_manifest(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    source_blob = repo_blob(
        cache_dir,
        "org--a",
        "etag-a",
    )
    write_blob(source_blob)

    assert shared.publish_to_shared(
        blob_path=source_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    )

    target_blob = repo_blob(
        cache_dir,
        "org--b",
        "etag-b",
    )

    assert shared.link_from_shared(
        blob_path=target_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    )

    manifest = shared_path(cache_dir).with_name(
        f"{XET_HASH}.refs"
    )

    lines = manifest.read_text().splitlines()

    assert "models--org--a/blobs/etag-a" in lines
    assert "models--org--b/blobs/etag-b" in lines


def test_link_from_shared_rejects_unknown_size(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    source_blob = repo_blob(
        cache_dir,
        "org--a",
        "etag-a",
    )
    write_blob(source_blob)

    assert shared.publish_to_shared(
        blob_path=source_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    )

    target_blob = repo_blob(
        cache_dir,
        "org--b",
        "etag-b",
    )

    assert shared.link_from_shared(
        blob_path=target_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=None,
    ) is False

    assert not target_blob.exists()
    assert not target_blob.is_symlink()


def test_link_from_shared_rejects_wrong_size(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    source_blob = repo_blob(
        cache_dir,
        "org--a",
        "etag-a",
    )
    write_blob(source_blob)

    assert shared.publish_to_shared(
        blob_path=source_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    )

    target_blob = repo_blob(
        cache_dir,
        "org--b",
        "etag-b",
    )

    assert shared.link_from_shared(
        blob_path=target_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE + 1,
    ) is False

    assert not target_blob.exists()
    assert not target_blob.is_symlink()


def test_link_from_shared_rejects_missing_payload(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    initialize_shared_store(cache_dir)

    target_blob = repo_blob(
        cache_dir,
        "org--b",
        "etag-b",
    )

    assert shared.link_from_shared(
        blob_path=target_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    ) is False

    assert not target_blob.exists()


def test_link_from_shared_rejects_unmarked_store(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    store = cache_dir / "blobs"
    store.mkdir(parents=True)

    target_blob = repo_blob(
        cache_dir,
        "org--b",
        "etag-b",
    )

    shared_file = shared_path(cache_dir)
    shared_file.parent.mkdir(parents=True)
    write_blob(shared_file)

    assert shared.link_from_shared(
        blob_path=target_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    ) is False

    assert not target_blob.exists()


def test_link_from_shared_rejects_invalid_repo_path(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    source_blob = repo_blob(
        cache_dir,
        "org--a",
        "etag-a",
    )
    write_blob(source_blob)

    assert shared.publish_to_shared(
        blob_path=source_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    )

    invalid_blob = (
        cache_dir
        / "invalid-repo"
        / "blobs"
        / "etag-b"
    )

    assert shared.link_from_shared(
        blob_path=invalid_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    ) is False


# ---------------------------------------------------------------------------
# Exact two-repository scenario
# ---------------------------------------------------------------------------


def test_shared_two_repository_flow(
    tmp_path: Path,
):
    """
    Simulate the exact scenario needed by hf_hub_download():

        repository A
            download
                -> publish shared payload

        repository B
            same Xet hash
                -> link existing shared payload
                -> no download
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    blob_a = repo_blob(
        cache_dir,
        "org--a",
        "etag-a",
    )

    write_blob(blob_a)

    # First repository publishes its downloaded payload.
    published = shared.publish_to_shared(
        blob_path=blob_a,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    )

    assert published is True

    shared_file = shared_path(cache_dir)

    assert shared_file.exists()
    assert shared_file.is_file()
    assert not shared_file.is_symlink()
    assert shared_file.read_bytes() == PAYLOAD

    assert blob_a.is_symlink()
    assert blob_a.resolve() == shared_file.resolve()

    # Second repository starts with no local blob.
    blob_b = repo_blob(
        cache_dir,
        "org--b",
        "etag-b",
    )

    assert not blob_b.exists()
    assert not blob_b.is_symlink()

    reused = shared.link_from_shared(
        blob_path=blob_b,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    )

    assert reused is True

    assert blob_b.exists()
    assert blob_b.is_symlink()
    assert blob_b.resolve() == shared_file.resolve()
    assert blob_b.read_bytes() == PAYLOAD


# ---------------------------------------------------------------------------
# Existing target behavior
# ---------------------------------------------------------------------------


def test_link_from_shared_existing_target(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    source_blob = repo_blob(
        cache_dir,
        "org--a",
        "etag-a",
    )
    write_blob(source_blob)

    assert shared.publish_to_shared(
        blob_path=source_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    )

    target_blob = repo_blob(
        cache_dir,
        "org--b",
        "etag-b",
    )
    write_blob(target_blob, b"existing!")

    # The current implementation calls os.replace() on the target,
    # so this verifies that behavior explicitly.
    assert shared.link_from_shared(
        blob_path=target_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=PAYLOAD_SIZE,
    ) is True

    assert target_blob.is_symlink()
    assert target_blob.read_bytes() == PAYLOAD


# ---------------------------------------------------------------------------
# Temporary symlink helper
# ---------------------------------------------------------------------------


def test_temporary_symlink_uses_relative_target(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    blob = repo_blob(
        cache_dir,
        "org--a",
        "etag-a",
    )

    store = shared_path(cache_dir)

    store.parent.mkdir(parents=True)
    write_blob(store)

    blob.parent.mkdir(parents=True, exist_ok=True)

    tmp_link = shared._make_temporary_symlink(
        blob,
        store,
    )

    try:
        assert tmp_link.is_symlink()
        assert not os.path.isabs(
            os.readlink(tmp_link)
        )
        assert tmp_link.resolve() == store.resolve()
    finally:
        tmp_link.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def test_manifest_path(
    tmp_path: Path,
):
    store = shared_path(tmp_path)

    expected = (
        tmp_path
        / "blobs"
        / "cc"
        / f"{XET_HASH}.refs"
    )

    assert shared._get_manifest_path(store) == expected


def test_lock_path(
    tmp_path: Path,
):
    store = shared_path(tmp_path)

    expected = (
        tmp_path
        / "blobs"
        / "cc"
        / f"{XET_HASH}.lock"
    )

    assert shared._get_lock_path(store) == expected


# ---------------------------------------------------------------------------
# Permission helpers
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.name == "nt",
    reason="Unix permission semantics",
)
def test_shared_blob_mode_is_readable(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(mode=0o755)
    cache_dir.chmod(0o755)

    mode = shared._shared_blob_mode(cache_dir)

    assert mode & stat.S_IRUSR
    assert mode & stat.S_IROTH


@pytest.mark.skipif(
    os.name == "nt",
    reason="Unix permission semantics",
)
def test_shared_directory_mode_follows_cache(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(mode=0o755)

    mode = shared._shared_directory_mode(cache_dir)

    assert mode & stat.S_IRUSR
    assert mode & stat.S_IWUSR
    assert mode & stat.S_IXUSR


# ---------------------------------------------------------------------------
# End-to-end shared API contract
# ---------------------------------------------------------------------------


def test_publish_then_link_is_zero_download_reuse(
    tmp_path: Path,
):
    """
    This is intentionally independent of hf_hub_download().

    It verifies the actual contract between the downloader and the
    shared store:

        downloaded local blob
            ↓
        publish_to_shared()
            ↓
        content-addressed shared payload
            ↓
        link_from_shared()
            ↓
        second repository local blob
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    first_blob = repo_blob(
        cache_dir,
        "org--first",
        "etag-first",
    )
    write_blob(first_blob)

    assert shared.publish_to_shared(
        blob_path=first_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=len(PAYLOAD),
    )

    shared_blob = shared_path(cache_dir)

    assert shared_blob.read_bytes() == PAYLOAD
    assert first_blob.resolve() == shared_blob.resolve()

    second_blob = repo_blob(
        cache_dir,
        "org--second",
        "etag-second",
    )

    assert shared.link_from_shared(
        blob_path=second_blob,
        xet_hash=XET_HASH,
        cache_dir=cache_dir,
        expected_size=len(PAYLOAD),
    )

    assert second_blob.resolve() == shared_blob.resolve()
    assert second_blob.read_bytes() == PAYLOAD

    manifest = shared_blob.with_name(
        f"{XET_HASH}.refs"
    )

    references = manifest.read_text().splitlines()

    assert "models--org--first/blobs/etag-first" in references
    assert "models--org--second/blobs/etag-second" in references
