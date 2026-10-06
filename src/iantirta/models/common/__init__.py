

from typing import TYPE_CHECKING

from iantirta.models._lazy_import import _LazyModule

_import_structure = {
    "auto": [
        "AutoConfig",
        "AutoModel",
    ]
}

if TYPE_CHECKING:
    pass
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__
    )
