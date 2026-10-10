
import pytest
from iantirta.models.remote.hf_api import HFApi, HFCachedFile, HFHTTPApi
from iantirta.models.remote.hf_api.errors import *
from rich import inspect
import logging
from unittest.mock import MagicMock, call, patch
from pathlib import Path

@pytest.fixture
def api():
    return HFApi()


@pytest.fixture
def url(api, repo_id) -> str:
    return api.hf_hub_url(repo_id, "config.json")


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

        # second pass
        cache.revision = "0"*40
        assert cache.revision_is_commit
        assert bool(cache.ref_path.exists())
        assert cache.ref_path.read_text() == "0"*40