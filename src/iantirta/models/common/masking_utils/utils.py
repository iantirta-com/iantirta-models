
from collections.abc import Callable

import torch
import torch.nn.functional as F

from iantirta.models.tools import (
    is_torch_flex_attn_available,
    is_torch_greater_or_equal,
    is_torch_xpu_available,
    is_torchdynamo_exporting,
    is_tracing,
)

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
else:
    # Register a fake type to avoid crashing for annotations and `isinstance` checks
    BlockMask = torch.Tensor


_is_torch_greater_or_equal_than_2_6 = is_torch_greater_or_equal("2.6", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()


def and_masks(*mask_functions: Callable) -> Callable:
    """Returns a mask function that is the intersection of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_ones((), dtype=torch.bool)
        for mask in mask_functions:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return and_mask


def or_masks(*mask_functions: Callable) -> Callable:
    """Returns a mask function that is the union of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def or_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_zeros((), dtype=torch.bool)
        for mask in mask_functions:
            result = result | mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return or_mask


def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a basic lower-diagonal causal mask.
    """
    return kv_idx <= q_idx


def bidirectional_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a full bidirectional mask.

    NOTE: It is important to keep an index-based version for non-vmap expansion.
    """
    return q_idx >= 0


def prepare_padding_mask(attention_mask: torch.Tensor | None, kv_length: int, kv_offset: int) -> torch.Tensor | None:
    """
    From the 2D attention mask, prepare the correct padding mask to use by potentially padding it.
    """
    local_padding_mask = attention_mask
    # Pad it if necessary
    if (
        attention_mask is not None
        and (
            padding_length := kv_length + kv_offset - attention_mask.shape[-1]
        ) > 0
    ):
        local_padding_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))
    return local_padding_mask


def maybe_pad_block_sequence_ids(
    block_sequence_ids: torch.Tensor, attention_mask: torch.Tensor | None, kv_length: int, kv_offset: int
) -> torch.Tensor:
    """
    Pads the `block_sequence_ids` in case the total length is less than `kv_length`.
    Usually that happens with `StaticCache` generation or generating without cache.
    Pads to the right with `-1`.
    """
    if (padding_length := kv_length + kv_offset - block_sequence_ids.shape[-1]) > 0:
        block_sequence_ids = F.pad(block_sequence_ids, pad=(0, padding_length), value=-1)
    return block_sequence_ids


def fast_all(tensor: torch.BoolTensor) -> torch.BoolTensor:
    """Similar to `tensor.all()`, but uses an implementation with `tensor.sum()`, which is actually much faster."""
    return tensor.sum() == tensor.numel()


def _ignore_causal_mask_sdpa(
    padding_mask: torch.Tensor | None,
    q_length: int,
    kv_length: int,
    q_offset: int,
    kv_offset: int,
    local_attention_size: int | None = None,
) -> bool:
    """
    Detects whether the causal mask can be ignored in case PyTorch's SDPA is used, rather relying on SDPA's `is_causal` argument.

    In case no token is masked in the 2D `padding_mask` argument, if `query_length == 1` or
    `key_value_length == query_length`, we rather rely on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is
    passed).
    """
    if padding_mask is not None and padding_mask.shape[-1] > kv_length:
        mask_indices = torch.arange(kv_length, device=padding_mask.device) + kv_offset
        padding_mask = padding_mask[:, mask_indices]

    # When using `torch.export` or `torch.onnx.dynamo_export`, we must pass an example input, and `is_causal` behavior is
    # hard-coded to the forward. If a user exports a model with query_length > 1, the exported model will hard-code `is_causal=True`
    # which is in general wrong (see https://github.com/pytorch/pytorch/issues/108108). Thus, we only set
    # `ignore_causal_mask = True` if we are not tracing
    # NOTE: under `torch.compile` we can still skip, but only if we do not have to read the values of the
    # `padding_mask`. This requires torch>=2.14: before pytorch#176499, dynamo replaced
    # `torch.compiler.is_exporting()` by a constant `True`, so older versions keep the previous behavior of
    # never skipping while compiling.
    if is_torchdynamo_exporting() or (padding_mask is not None and is_tracing(padding_mask)):
        return False
    # In this case, we need to add special patterns to the mask no matter what, so we cannot use any of the later skip conditions
    if local_attention_size is not None and kv_length >= local_attention_size:
        return False

    # If `q_length == 1`, we then use `is_causal=False` in sdpa integration to mimic lower-right alignment. If `kv_length == q_length`,
    # we use `is_causal=True` as upper-left alignment (torch's default) is the same as lower-right in this case. If we have padding,
    # we need to add padding to the mask, so cannot be skipped
    if (q_length == 1 or kv_length == q_length) and (padding_mask is None or fast_all(padding_mask)):
        return True
    # Additional case to optimize prefill: if the cache is empty (`q_offset == 0`), we can use `is_causal=True` even
    # with a padding_mask, if the padding_mask only contains padding related to "future k/v tokens" of the static k/v states
    # returned by StaticCaches. This works thanks to the upper-left alignment of sdpa's `is_causal` mask
    return bool(q_offset == 0 and (padding_mask is None or fast_all(padding_mask[:, :q_length]) and fast_all(~padding_mask[:, q_length:])))


def _can_skip_bidirectional_mask_xpu(
    padding_mask: torch.Tensor | None,
    kv_length: int,
    local_attention_size: int | None,
) -> bool:
    """
    XPU-specific logic for determining if we can skip bidirectional mask creation.

    For XPU devices, we have special handling:
    - Skip if no padding and no local attention constraint
    """

    if is_torchdynamo_exporting() or (padding_mask is not None and is_tracing(padding_mask)):
        return False

    # Check local attention constraint (same as CUDA)
    if local_attention_size is not None and kv_length >= local_attention_size:
        return False

    if padding_mask is None:
        # Without padding mask, can always skip for full bidirectional attention
        return True

    # Skip only if no padding tokens present
    return padding_mask.all()


def _ignore_bidirectional_mask_sdpa(
    padding_mask: torch.Tensor | None,
    kv_length: int,
    local_attention_size: int | None = None,
) -> bool:
    """
    Detects whether the bidirectional mask can be ignored in case PyTorch's SDPA is used.

    In case no token is masked in the 2D `padding_mask` argument and no local attention constraint applies
    (i.e. `local_attention_size` is None or `kv_length < local_attention_size`), we skip mask creation,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is
    passed).
    """
    if _is_torch_xpu_available:
        # XPU devices have special handling for mask skipping:
        # - Skip if no padding and no local attention constraint
        return _can_skip_bidirectional_mask_xpu(padding_mask, kv_length, local_attention_size)

    # When using `torch.export` or `torch.onnx.dynamo_export`, we need to avoid to check the contents of the mask;
    # otherwise, we will encounter dynamic control flows
    return bool(
        not (is_torchdynamo_exporting() or (padding_mask is not None and is_tracing(padding_mask)))
        and (padding_mask is None or padding_mask.all())
        # in this case we need to add special patterns to the mask so cannot be skipped otherwise
        and (local_attention_size is None or kv_length < local_attention_size)
    )


def _vmap_expansion_sdpa(mask_function: Callable) -> Callable:
    """
    Used to vmap our mask_functions over the all 4 dimensions (b_idx, h_idx, q_idx, kv_idx) of the inputs.
    Using vmap here allows us to keep the performance of vectorized ops, while having a single set of primitive
    functions between attention interfaces (i.e. between flex and sdpa/eager, FA2 being a bit different).
    """
    # We vmap the function over all 4 dimensions, broadcasting [b_idx, h_idx, q_idx, kv_idx]
    dimensions = [(None, None, None, 0), (None, None, 0, None), (None, 0, None, None), (0, None, None, None)]
    for dims in dimensions:
        mask_function = torch.vmap(mask_function, in_dims=dims, out_dims=0)
    return mask_function


def _non_vmap_expansion_sdpa(
    batch_indices: torch.Tensor, head_indices: torch.Tensor, q_indices: torch.Tensor, kv_indices: torch.Tensor
):
    """
    Used to broadcast our mask_functions over the all 4 dimensions (b_idx, h_idx, q_idx, kv_idx) of the inputs.
    Allows the usage of any index-based mask function without relying on vmap.

    NOTE: This is limited to index based functions only and is not guaranteed to work otherwise.

    Reference:
        - https://github.com/huggingface/optimum-onnx/blob/c123e8f4fab61b54a8e0e31ce74462bcacca576e/optimum/exporters/onnx/model_patcher.py#L362-L365
    """
    batch_indices = batch_indices[:, None, None, None]
    head_indices = head_indices[None, :, None, None]
    q_indices = q_indices[None, None, :, None]
    kv_indices = kv_indices[None, None, None, :]
    return batch_indices, head_indices, q_indices, kv_indices


def sliding_window_overlay(sliding_window: int) -> Callable:
    """
    This is an overlay depicting a sliding window pattern. Add it on top of a causal mask for a proper sliding
    window mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return kv_idx > q_idx - sliding_window

    return inner_mask


def chunked_overlay(chunk_size: int, left_padding: torch.Tensor) -> Callable:
    """
    This is an overlay depicting a chunked attention pattern. Add it on top of a causal mask for a proper chunked
    attention mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return (kv_idx - left_padding[batch_idx]) // chunk_size == (q_idx - left_padding[batch_idx]) // chunk_size

    return inner_mask


def blockwise_overlay(block_sequence_ids: torch.Tensor) -> Callable:
    """
    This is an overlay depicting a blockwise masking pattern. Instead of a single
    token, each block consists of arbitrary length tokens. In causal setup, each block
    can attend to prev block causally and can't attend to future blocks. Within one block
    the attention is always bidirectional.
    Mostly used in MLLMs when non-text data attends bidirectionally to itself.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # Unmask if the q and kv come from same group which is not -1 (i.e. non-text)
        q_group = block_sequence_ids[batch_idx, q_idx]
        kv_group = block_sequence_ids[batch_idx, kv_idx]
        return (q_group == kv_group) & (q_group >= 0)

    return inner_mask


def sliding_window_causal_mask_function(sliding_window: int) -> Callable:
    """
    This return the mask_function function to create a sliding window mask.
    """
    return and_masks(sliding_window_overlay(sliding_window), causal_mask_function)


def sliding_window_bidirectional_overlay(sliding_window: int) -> Callable:
    """
    This is an overlay depicting a bidirectional sliding window pattern.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        """A token can attend to any other token if their absolute distance is within
        the (inclusive) sliding window size (distance <= sliding_window)."""
        return abs(q_idx - kv_idx) <= sliding_window

    return inner_mask


def sliding_window_bidirectional_mask_function(sliding_window: int) -> Callable:
    """
    This return the mask_function function to create a bidirectional sliding window mask.
    """
    return and_masks(sliding_window_bidirectional_overlay(sliding_window), bidirectional_mask_function)


def chunked_causal_mask_function(chunk_size: int, left_padding: torch.Tensor) -> Callable:
    """
    This return the mask_function function to create a chunked attention mask.
    """
    return and_masks(chunked_overlay(chunk_size, left_padding), causal_mask_function)


def padding_mask_function(padding_mask: torch.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D padding mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # Note that here the mask should ALWAYS be at least of the max `kv_index` size in the dimension 1. This is because
        # we cannot pad it here in the mask_function as we don't know the final size, and we cannot try/except, as it is not
        # vectorizable on accelerator devices
        return padding_mask[batch_idx, kv_idx]

    return inner_mask


def packed_sequence_mask_function(packed_sequence_mask: torch.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D packed sequence mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]

    return inner_mask


def add_offsets_to_mask_function(mask_function: Callable, q_offset: int, kv_offset: int) -> Callable:
    """
    This function adds the correct offsets to the `q_idx` and `kv_idx` as the torch API can only accept lengths,
    not start and end indices.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return mask_function(batch_idx, head_idx, q_idx + q_offset, kv_idx + kv_offset)

    return inner_mask
