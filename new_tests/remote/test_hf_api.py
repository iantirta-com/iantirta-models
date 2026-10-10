
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from rich import inspect

from iantirta.models.remote.hf_api import HFApi, HFCachedFile, HFHTTPApi, _xet
from iantirta.models.remote.hf_api.errors import *


@pytest.fixture
def api():
    return HFApi()


@pytest.fixture
def url(api, repo_id) -> str:
    return api.hf_hub_url(repo_id, "config.json")

@pytest.fixture
def is_support_symlink():
    return HFCachedFile.support_symlink(HFCachedFile._cache_dir)

class TestMockHFApi:
    filename = "config.json"
    
    def test_api_creation(self):
        api = HFApi()
        assert api.endpoint is not None
        assert api.endpoint == HFHTTPApi.endpoint

    def test_api_custom_endpoint(self):
        api = HFApi(endpoint="http://example.com")
        assert api.endpoint == "http://example.com"

    def test_http_shared_session(self, api):
        api2 = HFApi(endpoint="http://example.com")
        assert api.session == api2.session

    @pytest.mark.parametrize(
        ("repo_type", "revision"),
        [
            (None, None),
            ("model", None),
            (None, "main"),
            ("model", "main")
        ]
    )
    def test_hf_hub_url(self, api, repo_id, repo_type, revision):
        expected = f"{repo_id}/resolve/main/" + self.filename
        url = api.hf_hub_url(repo_id, self.filename, repo_type=repo_type, revision=revision)
        assert url is not None
        assert url == expected

    def test_hf_hub_custom_repo_type(self, api, repo_id):
        url = api.hf_hub_url(repo_id, self.filename, repo_type="kernel")
        assert url is not None
        assert url == f"kernels/{repo_id}/resolve/main/" + self.filename

    @pytest.mark.parametrize(
        ("warnings", "expected"),
        (
            (["simple warning"], 1),
            ([
                "simple warning",
                "unauthenticated; Warning..."
            ], 1),
            ([
                "deprecation; The old API is deprecated.",
                "rate-limit; You are approaching your hourly limit.",
                "formatting;   Extra spaces should be stripped   "
            ], 3)
        )
    )
    def test_hf_raise_for_status_warning_headers(self, api, mock_response, caplog, warnings, expected):
        caplog.set_level(logging.WARNING)
        resp = mock_response(
            headers={"X-HF-Warning" : warnings},
            status_code=200
        )
        api._warn_on_warning_headers(resp.raw.headers)
        assert len(caplog.records) == expected

    @pytest.mark.manual
    def test_real_hf_raise_for_status_warning_headers(self, api, mock_response, caplog, repo_id):
        url = api.hf_hub_url(repo_id, "config.json")
        caplog.set_level(logging.WARNING)
        resp = api.request("head", url)
        api.raise_for_status(resp)
        assert resp is not None

    @pytest.mark.manual
    def test_resolve_revision(self, api, repo_id,):
        revision = api.resolve_revision(repo_id)
        assert revision.resolved != revision.initial

    @pytest.mark.manual
    def test_resolve_revision_local_only(self, api, repo_id):
        revision = api.resolve_revision(repo_id, local_only=True)
        assert revision.resolved != revision.initial

    @pytest.mark.manual
    def test_resolve_revision_local_only_non_existant(self, api, mock_repo):
        revision = None
        with pytest.raises(RevisionResolutionError):
            revision = api.resolve_revision(mock_repo, local_only=True)
        assert revision == None

    @pytest.mark.manual
    def test_get_file_metadata(self, api, repo_id):
        url = api.hf_hub_url(repo_id, "model.safetensors")
        fmd = api.get_file_metadata(url)
        assert fmd.commit_hash is not None

    @pytest.mark.manual
    def test_cache_non_existant_file_on_server(self, api, tmp_path, repo_id):
        # Create cache first so that it uses this
        cache = api.get_cache_file(repo_id, "model", cache_dir=tmp_path, revision="main", filename="hello.json")
        with pytest.raises(RemoteEntryNotFoundError):
            _ = api.hf_hub_download(repo_id, "hello.json", cache_dir=tmp_path)
        
        assert cache.ref_path.exists()

        # need to resolve revision first before checking the no_exist_file
        cache.revision = api.resolve_revision(repo_id, local_only=True).resolved
        assert cache.no_exist_file_path.exists()

    def test_get_cache_file(self, api, repo_id):
        # Test create cache here so that we can check the cache length
        cache = api.get_cache_file("other_repo/model_type", "model")
        cache = api.get_cache_file(repo_id, "model")
        assert cache is not None
        assert len(HFCachedFile._cached_files) > 1

    # this should be on the normal not hf test
    # def test_cache_dir_tag(self, tmp_path, api, repo_id):
    #     cache = api.get_cache_file(repo_id, "model", cache_dir=tmp_path)
    #     assert cache.cache_dir == tmp_path
    #     assert 
        
    def test_cache_file_on_change_revision(self, api, repo_id, tmp_path):
        cache = api.get_cache_file(repo_id, "model", cache_dir=tmp_path)
        assert cache.cache_dir == tmp_path
        assert cache.revision is None
        with pytest.raises(OSError):
            assert cache.ref_path is None

        # first pass, ref path exist but not yet exist.
        cache.revision = "main"
        assert cache.ref_path is not None and not cache.ref_path.exists()
        main_ref_path = cache.ref_path

        # second pass
        cache.revision = "0"*40
        assert cache.revision_is_commit
        assert main_ref_path.exists()
        assert main_ref_path.read_text() == "0"*40

    @pytest.mark.manual
    def test_hf_hub_download(self, api, tmp_path, repo_id, caplog):
        caplog.set_level(logging.DEBUG)
        config_path = api.hf_hub_download(repo_id, "config.json", cache_dir=tmp_path)
        assert config_path.is_symlink()

        # Tried checking with the cache_file
        cache_file = api.get_cache_file(repo_id, "model", filename="config.json", cache_dir=tmp_path)
        assert cache_file.pointer_path == config_path
        assert cache_file.blob_path == config_path.resolve()

    @pytest.mark.manual
    def test_hf_hub_download_second_time_uses_existing_cache(self, api, tmp_path, repo_id):

        with patch.object(api, "cache_download") as cache_download:
            _ = api.hf_hub_download(repo_id, "config.json", cache_dir=tmp_path)

        cache_download.assert_called_once()

        with patch.object(api, "cache_download") as cache_download:
            _ = api.hf_hub_download(repo_id, "config.json", cache_dir=tmp_path)

        assert cache_download.call_count == 1

    @pytest.mark.manual
    def test_hf_hub_download_on_connection_error_uses_cache(self, api, repo_id, mock_response, tmp_path):
        resp = mock_response(
            status_code=500
        )
        # Real download for start.
        _ = api.hf_hub_download(repo_id, "config.json", cache_dir=tmp_path)

        # Create the cache and revision second for assertion.
        cache_file = api.get_cache_file(repo_id, "model", filename="config.json", cache_dir=tmp_path)
        cache_file.revision = api.resolve_revision(repo_id).resolved
        assert cache_file.pointer_path.exists()

        with (patch.object(api, "request", return_value=resp),
        ):
            config = api.hf_hub_download(repo_id, "config.json", cache_dir=tmp_path)
        assert config is not None
        assert config == cache_file.pointer_path

    @pytest.mark.manual
    def test_hf_hub_download_using_cache_blob_with_pointer_path_no_exist(self, api, repo_id, mock_response, tmp_path):
        config = api.hf_hub_download(repo_id, "config.json", cache_dir=tmp_path)
        assert config.is_symlink()
        config.unlink()
        assert not config.is_file()

        # Create the cache and revision second for assertion.
        cache_file = api.get_cache_file(repo_id, "model", filename="config.json", cache_dir=tmp_path)
        cache_file.revision = api.resolve_revision(repo_id).resolved
        assert not cache_file.pointer_path.exists()
        assert cache_file.blob_path.exists()

        with patch.object(api, "cache_download") as cache_download:
            config = api.hf_hub_download(repo_id, "config.json", cache_dir=tmp_path)

        assert cache_file.pointer_path.exists()
        assert cache_file.pointer_path.is_symlink()
        assert config == cache_file.pointer_path
        assert cache_download.call_count == 0

    @pytest.mark.manual
    def test_hf_hub_download_using_xet(self, api, tmp_path, xet_repo, caplog):
        caplog.set_level(logging.DEBUG)
        with patch.object(api, "xet_download", wraps=api.xet_download) as xet_download:
            result = api.hf_hub_download(xet_repo, "tokenizer.json", cache_dir=tmp_path)

        xet_download.assert_called_once()
        assert result.is_file()
        assert result.is_symlink()

    @pytest.mark.manual
    def test_hf_xet_download_storing_blob(self, api, xet_repo, tmp_path, caplog, is_support_symlink):
        if not is_support_symlink:
            pytest.skip("Shared blob reuse requires symlink support")

        caplog.set_level(logging.DEBUG)
        
        with patch.object(api, "xet_download", wraps=api.xet_download) as xet_download:
            result = api.hf_hub_download(xet_repo, "tokenizer.json", cache_dir=tmp_path)

        xet_download.assert_called_once()

        # Setup cache file for modification
        cache_file = api.get_cache_file(xet_repo, "model", "tokenizer.json", cache_dir=tmp_path)
        
        fmd = api.get_file_metadata(api.hf_hub_url(xet_repo, "tokenizer.json"))
        
        cache_file.revision = fmd.commit_hash
        cache_file.etag = fmd.etag
        cache_file.xet_hash = fmd.xet_file_data.file_hash

        assert cache_file.is_xet_hash_valid
        
        assert cache_file.is_dir(cache_file.shared_blob_dir)
        assert cache_file._ensure_shared_blobs_dir()

        # Storing Manifest
        manifest_path = cache_file.shared_blob_path.with_name(f"{cache_file.shared_blob_path.name}.refs")
        assert manifest_path.exists() and manifest_path.is_file()
        assert manifest_path.read_text() == f"{cache_file.relative_blob_path}\n"

        # now everything became a symlink to shared
        if is_support_symlink:
            assert cache_file.blob_path.is_symlink()
            assert cache_file.pointer_path.is_symlink()
            assert not cache_file.shared_blob_path.is_symlink() and cache_file.shared_blob_path.is_file()

            assert cache_file.blob_path.resolve() == result.resolve()
            assert cache_file.pointer_path == result
            assert cache_file.shared_blob_path == result.resolve()

    @pytest.mark.manual
    def test_hf_xet_download_using_cached_blob(self, api, xet_repo, tmp_path, caplog, is_support_symlink):
        if not is_support_symlink:
            pytest.skip("Shared blob reuse requires symlink support")

        caplog.set_level(logging.DEBUG)

        # Setup cache file for modification
        cache_file = api.get_cache_file(xet_repo, "model", "tokenizer.json", cache_dir=tmp_path)
        
        fmd = api.get_file_metadata(api.hf_hub_url(xet_repo, "tokenizer.json"))
        
        cache_file.revision = fmd.commit_hash
        cache_file.etag = fmd.etag
        cache_file.xet_hash = fmd.xet_file_data.file_hash

        assert cache_file.is_xet_hash_valid
        
        with patch.object(api, "xet_download", wraps=api.xet_download) as xet_download:
            result = api.hf_hub_download(xet_repo, "tokenizer.json", cache_dir=tmp_path)
            cache_file.pointer_path.unlink()
            assert not cache_file.pointer_path.is_file()
            assert not result.is_file()
            cache_file.blob_path.unlink()
            assert not cache_file.blob_path.is_file()

            assert cache_file.is_dir(cache_file.shared_blob_dir)
            assert cache_file._ensure_shared_blobs_dir()

            result = api.hf_hub_download(xet_repo, "tokenizer.json", cache_dir=tmp_path)

        xet_download.assert_called_once()

    @pytest.mark.manual
    def test_hf_xet_download_fallback_using_http(self, tmp_path, xet_repo, caplog, monkeypatch):
        monkeypatch.setitem(sys.modules, "hf_xet", None)
        _xet.abort_xet_session()

        api = HFApi()

        with (patch.object(_xet, "available", return_value=False),
              patch.object(api, "xet_download") as xet_download,
              patch.object(api, "download") as http_download,
        ):
            _ = api.hf_hub_download(xet_repo, "tokenizer.json", cache_dir=tmp_path)

        xet_download.assert_not_called()
        http_download.assert_called_once()

    def test_hf_xet_unavailable(self, monkeypatch):
        try:
            import hf_xet

            monkeypatch.setitem(sys.modules, "hf_xet", None)
            _xet.abort_xet_session()
        except ImportError:
            pass
        
        api = HFApi()
        assert getattr(api, "xet_session", False) == False

        with pytest.raises(NotImplementedError):
            _ = api.xet_download()

    @pytest.mark.manual
    def test_hf_load_file_from_cache(self, repo_id, api, tmp_path):
        _ = api.hf_hub_download(repo_id, "config.json", cache_dir=tmp_path)

        cached_file = api.load_file_from_cache(repo_id, "config.json", cache_dir=tmp_path)
        assert cached_file is not None and cached_file.is_file()

    @pytest.mark.manual
    def test_hf_list_repo_tree(self, qwen_repo, api,):
        templates = api.list_repo_tree(qwen_repo, "additional_chat_templates", recursive=False)
    