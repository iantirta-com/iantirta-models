
import threading
import requests
from pathlib import Path
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextlib import ExitStack
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from typing import Any
import time
import os























def _test_progress(group_report, _item_reports: dict | None = None):
    bytes_inc = max(0, group_report.total_bytes_completed)
    transfer_inc = max(0, group_report.total_transfer_bytes_completed)
    print(bytes_inc, transfer_inc)
    print(_item_reports)



def _should_retry(res):
    return False


if __name__ == "__main__":
    from rich import inspect
    url = "https://huggingface.co/adefossez/Demucs-mdx_extra_q/resolve/main/83fc094f.safetensors"
    metadata = get_hf_file_metadata(url)
    #print(xet_download(metadata.xet_fd))
    print(http_download(url))
    # print(
    #     http_get(
    #         "https://huggingface.co/adefossez/Demucs-mdx_extra_q/resolve/main/83fc094f.safetensors",
    #         Path("test.txt"),
    #     )
    # )
    #res = http_stream_request("HEAD", url)
    # 