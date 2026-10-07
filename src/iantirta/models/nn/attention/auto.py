import logging
from collections.abc import Callable

from iantirta.models.tools.interface import GeneralInterface

from .eager_paged import eager_paged_attention_forward
from .flash import flash_attention_forward
from .flash_paged import paged_attention_forward
from .flex import flex_attention_forward
from .sdpa import sdpa_attention_forward
from .sdpa_paged import sdpa_attention_paged_forward

logger = logging.getLogger(__name__)


class AttentionInterface(GeneralInterface):
    """
    Dict-like object keeping track of allowed attention functions. You can easily add a new attention function
    with a call to `register()`. If a model needs to locally overwrite an existing attention function, say `sdpa`,
    it needs to declare a new instance of this class inside the `modeling_<model>.py`, and declare it on that instance.
    """

    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {  # noqa: RUF012
        "flash_attention_4": flash_attention_forward,
        "flash_attention_3": flash_attention_forward,
        "flash_attention_2": flash_attention_forward,
        "flex_attention": flex_attention_forward,
        "sdpa": sdpa_attention_forward,
        "paged|flash_attention_4": paged_attention_forward,
        "paged|flash_attention_3": paged_attention_forward,
        "paged|flash_attention_2": paged_attention_forward,
        "paged|sdpa": sdpa_attention_paged_forward,
        "paged|eager": eager_paged_attention_forward,
    }

    def get_interface(self, attn_implementation: str, default: Callable) -> Callable:
        """Return the requested `attn_implementation`. Also strictly check its validity, and raise if invalid."""
        if attn_implementation is None:
            logger.warning_once(
                "You tried to access the `AttentionInterface` with a `config._attn_implementation` set to `None`. This "
                "is expected if you use an Attention Module as a standalone Module. If this is not the case, something went "
                "wrong with the dispatch of `config._attn_implementation`"
            )
        elif attn_implementation != "eager" and attn_implementation not in self:
            raise KeyError(
                f"`{attn_implementation}` is not a valid attention implementation registered in the `AttentionInterface`"
            )
        return super().get(attn_implementation, default)


# Global AttentionInterface shared by all models which do not need to overwrite any of the existing ones
ALL_ATTENTION_FUNCTIONS: AttentionInterface = AttentionInterface()
