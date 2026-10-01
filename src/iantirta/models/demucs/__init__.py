# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from typing import TYPE_CHECKING

from iantirta.models.vendor.transformers.utils import _LazyModule
from iantirta.models.vendor.transformers.utils.import_utils import (
    define_import_structure,
)

if TYPE_CHECKING:
    from .apply import *
    from .configuration_utils import *
    from .modeling_utils import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
