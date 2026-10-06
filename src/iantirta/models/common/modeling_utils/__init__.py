# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of transformers, improved by iantirta.com

from typing import TYPE_CHECKING

from iantirta.models._lazy_import import _LazyModule

_import_structure = {
    "pretrained": [
        "PreTrainedModel",
    ]
}

if TYPE_CHECKING:
    from .mixin import *
    from .utils import *
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__
    )
