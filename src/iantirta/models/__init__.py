# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

import importlib
import types
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from .vendor.transformers.utils import _LazyModule
from .vendor.transformers.utils.import_utils import define_import_structure

__version__ = version("iantirta-models")


if TYPE_CHECKING:
    from .demucs import *
    from .vendor.transformers import *
else:
    import sys
    
    # Purge any pre-existing system modules and submodules from the cache
    for key in list(sys.modules.keys()):
        if (
            key == "transformers" or key.startswith(("transformers.", "huggingface_hub.")) or key == "huggingface_hub"
        ):
            del sys.modules[key]

    from .vendor import huggingface_hub, transformers

    sys.modules["huggingface_hub"] = huggingface_hub
    sys.modules["transformers"] = transformers

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        define_import_structure(_file),
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
