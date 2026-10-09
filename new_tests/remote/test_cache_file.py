import pytest
from iantirta.models.remote.cache_file.mixin import (
    CachedFile
)
from rich import inspect as i


def test_cache_file_instance_creation():
    c = CachedFile("test.tmp")
    assert c is not None

    # Not Change
    assert c.cache_dir == CachedFile.cache_dir
    assert c.storage_dir == CachedFile.cache_dir / "Downloaded"
    i(c)