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

    def _create_module_alias(alias: str, target: str) -> None:
        """
        Lazily redirect legacy module paths to their replacements without importing heavy deps.
        """
        module = types.ModuleType(alias)
        module.__doc__ = f"Alias module for backward compatibility with `{target}`."
        # Set __file__ explicitly so that inspect.py's hasattr(module, '__file__') check
        # never falls through to __getattr__ and triggers a premature (possibly circular) import.
        module.__file__ = None

        def _get_target():
            return importlib.import_module(target, __name__)

        module.__getattr__ = lambda name: getattr(_get_target(), name)
        module.__dir__ = lambda: dir(_get_target())

        sys.modules[alias] = module
        setattr(sys.modules[__name__], alias.rsplit(".", 1)[-1], module)

    _create_module_alias("transformers", ".vendor.transformers")
    _create_module_alias("transformers.models", ".vendor.transformers.models")

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        define_import_structure(_file),
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
