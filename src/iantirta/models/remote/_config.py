


import os
from urllib.parse import urlsplit

_HF_DEFAULT_ENDPOINT = "https://huggingface.co"
_HF_DEFAULT_STAGING_ENDPOINT = "https://hub-ci.huggingface.co"
ENDPOINT = os.getenv("HF_ENDPOINT", _HF_DEFAULT_ENDPOINT).rstrip("/")

HF_URL_HOSTS: frozenset[str] = frozenset(
    {"hf.co"}
    | {
        host.lower()
        for host in (
            urlsplit(_HF_DEFAULT_ENDPOINT).hostname,
            urlsplit(_HF_DEFAULT_STAGING_ENDPOINT).hostname,
            urlsplit(ENDPOINT).hostname,
        )
        if host
    }
)