from pathlib import Path

DEFAULT_CACHE_DIR = Path("~/.cache/iantirta").expanduser().resolve()

CACHEDIR_TAG_CONTENT = (
    "Signature: 8a477f597d28d172789f06886806bc55\n"
    "# This file is a cache directory tag created by iantirta.models.\n"
    "# For information about cache directory tags, see:\n"
    "#\thttps://bford.info/cachedir/\n"
)