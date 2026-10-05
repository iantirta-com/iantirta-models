
import re

import torch

from ..configuration_utils import PreTrainedConfig


def split_attention_implementation(implementation: str | None) -> tuple[bool, str | None]:
    """
    Split the optional `paged|` prefix from an attention implementation string.

    Note that `None` means using the default attention implementation, which is either torch's native `sdpa` or `eager` (if `sdpa` is not implemented for that model).
    """
    if implementation is None:
        return False, None

    is_paged = implementation.startswith("paged|")
    return is_paged, implementation.removeprefix("paged|")


def is_flash_attention_requested(
    config=None, requested_attention_implementation: str | None = None, version: int | list[int] | None = None
) -> bool:
    """
    Checks whether some flavor of flash attention is requested or not. Optionally, checks for specific versions of
    flash attention.

    This is checked against one of the two arguments, i.e. either the `config` or the directly passed value
    `requested_attention_implementation`. Otherwise, an error will be raised (ambiguity).

    The different versions of flash attention are usually
    - Implementations based on the original flash attention repo: https://github.com/Dao-AILab/flash-attention
    - Kernels implementations such as: https://huggingface.co/kernels-community/vllm-flash-attn3
    """
    if config is not None and requested_attention_implementation is not None:
        raise ValueError(
            "Requested attention implementation is ambiguous: "
            "Please pass either the config or the name of the attention implementation, not both."
        )

    if config is not None:
        checked_attention_implementation = config._attn_implementation
    else:
        checked_attention_implementation = requested_attention_implementation

    # theoretically can happen, equivalent to default implementation (sdpa/eager)
    if checked_attention_implementation is None:
        return False

    # If a specific version is requested, look for a pattern of type "flash...{version}"
    if version is not None:
        if isinstance(version, int):
            version = [version]
        return any(re.match(r".*flash.*" + str(v), checked_attention_implementation) is not None for v in version)

    # Otherwise, just check "flash" is in the attention implementation
    return "flash" in checked_attention_implementation


def get_max_seqlen(
    cu_seqlens: torch.Tensor,
    config: PreTrainedConfig,
    kwargs: dict | None = None,
    kwarg_name: str = "max_seqlen",
) -> int | None:
    """Get the maximum packed sequence length, or pop it from `kwargs` if precomputed.

    Args:
        cu_seqlens: `(num_sequences + 1,)` cumulative sequence boundaries.
        config: model configuration used to determine the attention implementation.
        kwargs: optional caller kwargs containing a precomputed maximum sequence length.
        kwarg_name: key used to pop the precomputed value from `kwargs`.

    Returns:
        Maximum packed sequence length as a Python integer, or `None` when Flash Attention is not requested
        and no precomputed value is provided.
    """
    if kwargs is not None and (max_seqlen := kwargs.pop(kwarg_name, None)) is not None:
        return max_seqlen
    if not is_flash_attention_requested(config):
        return None
    return (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
