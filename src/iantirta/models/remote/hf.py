import os
from pathlib import Path
from dataclasses import dataclass
import errno
import shutil

from .utils import WeakFileLock
from .xet import try_link_from_shared_store, publish_blob_to_shared_store

CACHE_DIR = Path("~/.cache/iantirta").expanduser().resolve()
CACHEDIR_TAG_CONTENT = (
    "Signature: 8a477f597d28d172789f06886806bc55\n"
    "# This file is a cache directory tag created by huggingface_hub.\n"
    "# For information about cache directory tags, see:\n"
    "#\thttps://bford.info/cachedir/\n"
)
SYMLINK_CACHE_SUPPORTED: dict[str, bool] = {}








