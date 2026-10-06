# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from importlib.metadata import version
from typing import TYPE_CHECKING

from iantirta.models._lazy_import import _LazyModule, define_import_structure

__version__ = version("iantirta-models")


if TYPE_CHECKING:
    from .demucs import *
else:
    import sys
    
    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        define_import_structure(_file),
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
