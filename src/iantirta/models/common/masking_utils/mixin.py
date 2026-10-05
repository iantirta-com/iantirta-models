from collections.abc import Callable

import torch

from ..cache_utils import Cache
from ..configuration_utils import PreTrainedConfig
from .attention_mask import ALL_MASK_ATTENTION_FUNCTIONS
from .utils import (
    BlockMask,
    _is_torch_greater_or_equal_than_2_6,
    and_masks,
    bidirectional_mask_function,
    is_tracing,
    or_masks,
)


def find_packed_sequence_indices(position_ids: torch.Tensor) -> torch.Tensor | None:
    """
    Find the indices of the sequence to which each new query token in the sequence belongs when using packed
    tensor format (i.e. several sequences packed in the same batch dimension).

    Args:
        position_ids (`torch.Tensor`)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.

    Returns:
        A 2D tensor where each similar integer indicates that the tokens belong to the same sequence. For example, if we
        pack 3 sequences of 2, 3 and 1 tokens respectively along a single batch dim, this will return [[0, 0, 1, 1, 1, 2]].

        If the there is only one sequence in each batch item (and we don't compile), then we return `None` indicating
        no packed sequences. This is the same as [[0, 0, 0, 0, 0, 0]] for the example above.
    """
    # What separate different sequences is when 2 consecutive positions_ids are separated by more than 1. So
    # taking the diff (by prepending the first value - 1 to keep correct indexing) and applying cumsum to the result
    # gives exactly the sequence indices
    # Note that we assume that a single sequence cannot span several batch dimensions, i.e. 1 single sequence
    # cannot be part of the end of the first batch dim and the start of the 2nd one for example
    first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
    position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
    packed_sequence_mask = (position_diff != 1).cumsum(-1)

    # Sadly this is a dynamic control flow, so we cannot enable this check on anything compile related
    if not is_tracing(packed_sequence_mask) and (packed_sequence_mask[:, -1] == 0).all():
        return None

    return packed_sequence_mask


def _preprocess_mask_arguments(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | BlockMask | None, # type: ignore
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None,
    layer_idx: int | None,
    encoder_hidden_states: torch.Tensor | None = None,
) -> tuple[bool, torch.Tensor | BlockMask | None, int, int]: # type: ignore
    """
    Perform some common pre-processing of the mask arguments we get from the modeling code. Mostly determine the
    key-value length and offsets, and if we should early exit or not.

    Args:
        config (`PreTrainedConfig`):
            The model config.
        inputs_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        layer_idx (`int`, optional):
            If `past_key_values` is not None, this is the layer index of the cache from which to get the key-value
            length and offset. Indeed, for hybrid caches, different layers may return different lengths.
        encoder_hidden_states (`torch.Tensor`, optional):
            The input embeddings of shape (batch_size, kv_length, hidden_dim). If provided, it is used instead of
            `inputs_embeds` to infer the kv length.

    Returns:
        early_exit (`bool`):
            Whether we should early exit mask creation, and return the mask as-is.
        attention_mask (`torch.Tensor` or `BlockMask` or `None`):
            The attention mask to either return immediately, or to use in downstream mask creation.
        packed_sequence_mask (`torch.Tensor`, optional):
            In case we detected packed sequence format, this is a tensor where each similar integer indicates that
            the tokens belong to the same sequence.
        q_length (`int`):
            The size that the query states will have during the attention computation.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        q_offset (`int`, optional):
            An optional offset to indicate at which first position the query states will refer to.
        kv_offset (`int`):
            An offset to indicate at which first position the key and values states will refer to.
    """
    # If the mask is already 4D, simply return as-is (it was already prepared, or it is custom)
    if isinstance(attention_mask, (torch.Tensor, BlockMask)) and len(attention_mask.shape) == 4:
        return True, attention_mask, None, None, None, None, None

    # For TGI/vLLM backends, or other custom attention without equivalent mask creation: we don't need a mask!
    # Note: it's not ideal to check the `_global_mapping` attribute instead of the object itself, however otherwise
    # full graph dynamo tracing (i.e. torch.export or compile with `fullgraph=True`) will fail on Python<3.11
    # with `torch._dynamo.exc.Unsupported: 'inline in skipfiles:Mapping.__contains__ | __contains__, skipped
    # according trace_rules.lookup SKIP_DIRS'` -- can be removed when we require Python>=3.11
    if config._attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        return True, None, None, None, None, None, None

    # Move the mask to correct device, and potentially switch dtype for efficiency
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)

    q_length = inputs_embeds.shape[1]
    # If using a cache, it can give all information about mask sizes based on seen tokens
    if past_key_values is not None:
        q_offset = past_key_values.get_query_offset(layer_idx)
        # To avoid graph breaks, StaticLayer returns a tensor instead of an int -> this has no impact on the ops, but
        # we need the correct device
        q_offset = q_offset.to(inputs_embeds.device) if isinstance(q_offset, torch.Tensor) else q_offset
        kv_length, kv_offset = past_key_values.get_mask_sizes(q_length, layer_idx)
    # Otherwise, we infer based on our input
    else:
        q_offset = 0
        # 1. Rely on input directly
        if attention_mask is None:
            # For encoder-decoders, use encoder_hidden_states to infer kv_length if provided
            kv_length = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else q_length
            kv_offset = 0
        # 2. Rely on the mask instead - needed for special cases like prefix tuning in PEFT
        #
        # This is a very unique and special case where an encoder utilizes a cache and expects its length
        # to be accounted for (usually, they should never use a cache). In general, the mask should always
        # match with the input sizes nonetheless (i.e. it does not affect others).
        # Conclusion: "prefix tuning is evil"
        else:
            kv_length, kv_offset = attention_mask.shape[-1], 0

    # We check the position_ids for potential packed sequence format (only if the 2D attention mask is explicitly None,
    # and we don't have past_key_values, i.e. generally a training setup)
    packed_sequence_mask = None
    if position_ids is not None and attention_mask is None and past_key_values is None:
        batch_size = inputs_embeds.shape[0]
        # The position ids are sometimes just unsqueezed, without being expanded
        if batch_size != position_ids.shape[0]:
            position_ids = position_ids.expand(batch_size, -1)
        packed_sequence_mask = find_packed_sequence_indices(position_ids)

    return False, attention_mask, packed_sequence_mask, q_length, kv_length, q_offset, kv_offset


def create_bidirectional_mask(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    encoder_hidden_states: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
    allow_is_bidirectional_skip: bool = True,
    **kwargs,
) -> torch.Tensor | BlockMask | None: # type: ignore
    """
    Create a standard bidirectional mask based on the attention implementation used (stored in the config).

    Args:
        config (`PreTrainedConfig`):
            The model config.
        inputs_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is only used to infer metadata
            such as the batch size, query length, dtype, and device.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, kv_length).
            It can also be an already prepared 4D mask of shape (batch_size, 1, query_length, kv_length),
            in which case it is returned as-is.
        encoder_hidden_states (`torch.Tensor`, optional):
            The input embeddings of shape (batch_size, kv_length, hidden_dim). If provided, it is used instead of
            `inputs_embeds` to infer the batch size, kv length and dtype.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the base mask function (by doing the union of both). This is
            useful to easily overlay another mask on top, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the base mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top, for example for image tokens handling.
        allow_is_bidirectional_skip (`bool`, optional):
            Whether to allow returning `None` (no bias) when the mask is plain bidirectional with no padding. Set to
            `False` to always materialize the mask, e.g. when it is later concatenated with another mask. Defaults to
            `True`.
    """
    # If we have a hybrid cache structure, here we want to create the mask for the full layers
    if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(False)
    else:
        layer_idx = 0

    # We ignore a few irrelevant arguments at the end as we do not have a (growing) cache here
    early_exit, attention_mask, _, q_length, kv_length, q_offset, kv_offset = _preprocess_mask_arguments(
        config, inputs_embeds, attention_mask, past_key_values, None, layer_idx, encoder_hidden_states
    )
    if early_exit:
        return attention_mask

    embeds = encoder_hidden_states if encoder_hidden_states is not None else inputs_embeds
    batch_size, dtype = embeds.shape[0], embeds.dtype
    # Use `inputs_embeds.device` to stay consistent with `_preprocess_mask_arguments`, which moves the 2D
    # `attention_mask` to that device. In model parallel setups, `encoder_hidden_states` may live on a different
    # device than `inputs_embeds` (e.g. cross-attention from a decoder to encoder states).
    device = inputs_embeds.device
    mask_factory_function = bidirectional_mask_function
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    # Defaulting to using non-vmap based mask creations except when detecting
    # users passing custom mask functions (as we cannot guarantee that they
    # are properly index-based as required by our implementation).
    use_vmap = False

    # Allow slight deviations from the base mask
    # Note that it is very important to apply this before any other deviations of the mask (such as packed sequence mask,
    # padding mask, etc) as the resulting mask may otherwise not be correct!
    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_bidirectional_skip = False
        use_vmap = True
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_bidirectional_skip = False
        use_vmap = True

    # We now create the mask
    attention_mask = mask_interface(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        # Additional kwargs for sdpa
        allow_is_causal_skip=False,
        allow_is_bidirectional_skip=allow_is_bidirectional_skip,
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
        use_vmap=use_vmap,  # Short-circuit to non-vmap expansions for the mask
        device=device,
    )
    return attention_mask
