

from typing import TYPE_CHECKING

from iantirta.models._lazy_import import _LazyModule, define_import_structure

if TYPE_CHECKING:
    pass
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(
        __name__,
        _file,
        define_import_structure(_file),
        module_spec=__spec__
    )
