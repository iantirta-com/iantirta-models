import logging
from abc import ABC, abstractmethod

import torch

from iantirta.models.tools import is_torch_greater_or_equal, is_torchdynamo_compiling
from iantirta.models.tools.deprecated import deprecate_kwarg

logger = logging.getLogger(__name__)


_is_torch_greater_or_equal_than_2_7 = is_torch_greater_or_equal("2.7", accept_dev=True)


class CacheLayerMixin(ABC):
    """Base, abstract class for a single layer's cache."""

    is_compileable = False
    is_croppable = False
    supports_early_init = True
    # Subclasses can set `_layer_type` to auto-register themselves in the mappings, if the class definition lives in a modeling
    # file instead of this file. This allows to update the mapping only when the modeling file is imported, which simplifies imports
    _layer_type: str | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._layer_type is not None:
            if issubclass(cls, StaticLayer):
                STATIC_LAYER_TYPE_MAPPING[cls._layer_type] = cls
            else:
                DYNAMIC_LAYER_TYPE_MAPPING[cls._layer_type] = cls

    def __init__(self, **kwargs):
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.is_initialized = False

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None: ...

    @abstractmethod
    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def get_mask_sizes(self, query_length: int) -> tuple[int, int]: ...

    @abstractmethod
    def get_seq_length(self) -> int: ...

    @abstractmethod
    def get_max_length(self) -> int:
        """
        Returns the maximum sequence length the layer can hold. A value of `-1` means no maximum, or an undefined
        maximum, for example a dynamic attention layer that grows indefinitely or a linear attention layer that has no
        sequence length dimension.
        """
        ...

    def offload(self):
        """Offload this layer's data to CPU device."""
        if self.is_initialized:
            self.keys = self.keys.to("cpu", non_blocking=True)
            self.values = self.values.to("cpu", non_blocking=True)

    def prefetch(self):
        """In case of layer offloading, this allows to move the data back to the layer's device ahead of time."""
        if self.is_initialized and self.keys.device != self.device:
            self.keys = self.keys.to(self.device, non_blocking=True)
            self.values = self.values.to(self.device, non_blocking=True)

    def reset(self) -> None:
        """Resets the cache values while preserving the objects"""
        if self.is_initialized:
            self.keys.zero_()
            self.values.zero_()
        # This attribute is set on several Layers
        if hasattr(self, "cumulative_length"):
            # It can either be an int for dynamic layers, or a tensor for static layers
            if isinstance(self.cumulative_length, int):
                self.cumulative_length = 0
            else:
                self.cumulative_length.zero_()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorders this layer's cache for beam search."""
        if self.get_seq_length() > 0:
            self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
            self.values = self.values.index_select(0, beam_idx.to(self.values.device))

    def get_max_cache_shape(self) -> int:
        logger.warning(
            "`get_max_cache_shape` is deprecated, and will be removed in version 5.16. Please use `get_max_length` instead"
        )
        return self.get_max_length()


class LinearAttentionCacheLayerMixin(ABC):
    """Base, abstract class for a linear attention single layer's cache."""

    # All shapes are static by essence in a LinearAttention layer, so it is compilable
    is_compileable = True
    # Linear attention layers track their own conv/recurrent states; they don't use the key/value early-init path.
    supports_early_init = False

    def __init__(self, number_of_states: int = 1, **kwargs):
        self.number_of_states = number_of_states
        # We allow to have an arbitrary number of cached states inside a single layer
        self.conv_states: dict[int, torch.Tensor | None] = dict.fromkeys(range(number_of_states))
        self.recurrent_states: dict[int, torch.Tensor | None] = dict.fromkeys(range(number_of_states))
        self.is_conv_states_initialized = dict.fromkeys(range(number_of_states), False)
        self.is_recurrent_states_initialized = dict.fromkeys(range(number_of_states), False)
        self.has_previous_state = dict.fromkeys(range(number_of_states), False)
        self.conv_kernel_size = dict.fromkeys(range(number_of_states))
        self.device = None
        self.dtype = None
        self.record_past = False

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def lazy_initialization(
        self,
        conv_states: torch.Tensor | None = None,
        recurrent_states: torch.Tensor | None = None,
        state_idx: int = 0,
    ) -> None: ...

    @abstractmethod
    def update_conv_state(self, conv_states: torch.Tensor, state_idx: int = 0) -> torch.Tensor: ...

    @abstractmethod
    def update_recurrent_state(self, recurrent_states: torch.Tensor, state_idx: int = 0) -> torch.Tensor: ...

    def offload(self):
        """Offload this layer's data to CPU device."""
        for i in range(self.number_of_states):
            if self.is_conv_states_initialized[i]:
                self.conv_states[i] = self.conv_states[i].to("cpu", non_blocking=True)
            if self.is_recurrent_states_initialized[i]:
                self.recurrent_states[i] = self.recurrent_states[i].to("cpu", non_blocking=True)

    def prefetch(self):
        """In case of layer offloading, this allows to move the data back to the layer's device ahead of time."""
        for i in range(self.number_of_states):
            if self.is_conv_states_initialized[i] and self.conv_states[i].device != self.device:
                self.conv_states[i] = self.conv_states[i].to(self.device, non_blocking=True)
            if self.is_recurrent_states_initialized[i] and self.recurrent_states[i].device != self.device:
                self.recurrent_states[i] = self.recurrent_states[i].to(self.device, non_blocking=True)

    def reset(self) -> None:
        """Resets the cache values while preserving the objects"""
        for i in range(self.number_of_states):
            if self.is_conv_states_initialized[i]:
                self.conv_states[i].zero_()
            if self.is_recurrent_states_initialized[i]:
                self.recurrent_states[i].zero_()
            self.has_previous_state[i] = False

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for i in range(self.number_of_states):
            if self.is_conv_states_initialized[i]:
                self.conv_states[i] = self.conv_states[i].index_select(0, beam_idx.to(self.device))
            # recurrent_states can stay empty sometimes, see e.g. lfm2 which only uses the conv_states
            if self.is_recurrent_states_initialized[i]:
                self.recurrent_states[i] = self.recurrent_states[i].index_select(0, beam_idx.to(self.device))

    @property
    def is_croppable(self) -> bool:
        """
        Whether `crop` can put this layer back as it was. This is only supported when there are no recurrent states.
        """
        if any(self.is_recurrent_states_initialized.values()):
            return False
        # If nothing is initialized, return False as we don't yet know whether we will have any recurrent states or no, so let's be
        # extra careful. If a conv states is initialized but no recurrent states are, then we return True as we know that we will never
        # have any recurrent state (they are updated in the same forward)
        return any(self.is_conv_states_initialized.values())

    def activate_past_recording(self):
        """
        Calling this function will activate past state recording, meaning that a call to `update_conv_states` will
        wait for a call to `crop` before restricting the size of the `conv_states` to `conv_kernel_size`, to be able
        to retrieve previous full states.
        """
        self.record_past = True

    def crop(self, tokens_to_remove: int):
        """
        Remove `tokens_to_remove` tokens from the current cache layer. This will also restrict the size of the cached states back to their
        minimal working size, i.e. `conv_kernel_size`. This means that `crop(0)` will not necessarily always be a no-op, as it may
        still remove useless states (i.e. states that are not needed for the next `forward`).
        """
        if not self.record_past:
            raise RuntimeError(
                "`crop` was called, but the current layer does not track past states. Call `activate_past_recording` before "
                "`crop` to be able to rollback the cache."
            )
        if tokens_to_remove > 0:
            raise RuntimeError(
                "Linear attention layers can only be cropped by passing a negative int, to specify how many tokens to remove"
            )
        for i in range(self.number_of_states):
            tokens_to_remove = abs(tokens_to_remove)
            # In this case, simply restrict the size back to `conv_kernel_size` without cropping
            if tokens_to_remove == 0:
                self.conv_states[i] = self.conv_states[i][..., -self.conv_kernel_size[i] :]
            # This both crop the last `tokens_to_remove`, as well as resize the conv states to `conv_kernel_size` as we never
            # need more for the next forward
            else:
                self.conv_states[i] = self.conv_states[i][
                    ..., -tokens_to_remove - self.conv_kernel_size[i] : -tokens_to_remove
                ]

    def get_max_length(self) -> int:
        # LinearAttention layer have no sequence length dimension, so simply return -1 here
        return -1


class Cache:
    """
    A `Cache` is mostly a list of `CacheLayerMixin` objects, one per model layer. It serves as a container for
    the Cache of each layer.

    Args:
        layers (`Optional`, *optional*):
            A list of pre-created `CacheLayerMixin` or `LinearAttentionCacheLayerMixin`. If omitted (`None`), then `layer_class_to_replicate`
            will be used.
        layer_class_to_replicate (`type[CacheLayerMixin | LinearAttentionCacheLayerMixin]`, *optional*):
            Only used if `layers` is omitted (`None`), in which case it will be used as the base class for each layer,
            and the layers will be added lazily as soon as `update` is called with a `layer_idx` greater than the current
            list of layers.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
        offload_only_non_sliding (`bool`, *optional*, defaults to `True`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).
    """

    def __init__(
        self,
        layers: list[CacheLayerMixin | LinearAttentionCacheLayerMixin] | None = None,
        layer_class_to_replicate: type[CacheLayerMixin | LinearAttentionCacheLayerMixin] | None = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
    ):
        if layers is not None and layer_class_to_replicate is not None:
            raise ValueError(
                "You can construct a Cache either from a list `layers` of all the predefined `CacheLayer`, or from a "
                "`layer_class_to_replicate`, in which case the Cache will append a new layer corresponding to "
                "`layer_class_to_replicate` for each new call to `update` with an idx not already in the Cache."
            )
        if layers is None and layer_class_to_replicate is None:
            raise ValueError(
                "You should provide exactly one of `layers` or `layer_class_to_replicate` to initialize a Cache."
            )
        self.layers = layers if layers is not None else []
        self.layer_class_to_replicate = layer_class_to_replicate
        self.offloading = offloading
        if self.offloading:
            self.only_non_sliding = offload_only_non_sliding
            self.prefetch_stream = torch.Stream() if _is_torch_greater_or_equal_than_2_7 else torch.cuda.Stream()

    def __repr__(self):
        return f"{self.__class__.__name__}(layers={self.layers})"

    def __len__(self):
        """
        This value corresponds to the number of layers in the model.
        """
        # Note: for DynamicCache, layers are initialized lazily, so this will not be accurate before the first
        # forward through all the layers
        return len(self.layers)

    def prefetch(self, layer_idx: int, only_non_sliding: bool = True):
        """
        Prefetch the next offloaded layer on its device, starting at `layer_idx` and circling back to the beginning
        if needed. Linear-attention layers are never offloaded and are skipped, as are sliding layers when
        `only_non_sliding`. Note that we use a non-default stream for this, to avoid blocking.
        """
        # Whether each layer is offloaded, hence worth prefetching: linear-attention layers never go through the
        # offloading `update` path, and sliding layers are skipped when `only_non_sliding` (kept resident).
        is_offloaded = [
            not is_linear and not (only_non_sliding and is_sliding)
            for is_linear, is_sliding in zip(self.is_linear, self.is_sliding)
        ]
        try:
            # Try to find the next offloaded layer, starting at `layer_idx`
            layer_idx = layer_idx + is_offloaded[layer_idx:].index(True)
        # In this case, we need to circle back to the beginning
        except ValueError:
            layer_idx = is_offloaded.index(True)

        # Prefetch
        with self.prefetch_stream if _is_torch_greater_or_equal_than_2_7 else torch.cuda.stream(self.prefetch_stream):
            self.layers[layer_idx].prefetch()

    def offload(self, layer_idx: int, only_non_sliding: bool = True):
        """
        Offload a given `layer_idx`. If `only_non_sliding` is True, it will offload `layer_idx` only if it is a
        non-sliding layer. Note that we do it on the default stream, so that we ensure all earlier
        computation in the layer's `update` methods are finished.
        """
        if not (only_non_sliding and self.is_sliding[layer_idx]):
            self.layers[layer_idx].offload()

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.

        Return:
            A tuple containing the updated key and value states.
        """
        # In this case, the `layers` were not provided, and we must append as much as `layer_idx`
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        if self.offloading:
            # Wait for the stream to finish if needed, and start prefetching the next layer
            torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream)
            self.prefetch(layer_idx + 1, self.only_non_sliding)

        keys, values = self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)

        if self.offloading:
            self.offload(layer_idx, self.only_non_sliding)

        return keys, values

    def update_conv_state(
        self, conv_states: torch.Tensor, layer_idx: int, state_idx: int = 0, **kwargs
    ) -> torch.Tensor:
        """
        Updates the cache with the new `conv_states` for the layer `layer_idx`.

        Parameters:
            conv_states (`torch.Tensor`):
                The new conv states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.

        Return:
            `torch.Tensor`: The updated conv states.
        """
        # NOTE: if we slightly break `update` arg order, we could combine this with it, and allow offloading support
        # out of the box
        if not isinstance(self.layers[layer_idx], LinearAttentionCacheLayerMixin):
            raise TypeError("Cannot call `update_conv_state` on a non-LinearAttention layer!")
        conv_states = self.layers[layer_idx].update_conv_state(conv_states, state_idx, **kwargs)
        return conv_states

    def update_recurrent_state(
        self, recurrent_states: torch.Tensor, layer_idx: int, state_idx: int = 0, **kwargs
    ) -> torch.Tensor:
        """
        Updates the cache with the new `recurrent_states` for the layer `layer_idx`.

        Parameters:
            smm_states (`torch.Tensor`):
                The new ssm states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.

        Return:
            `torch.Tensor`: The updated ssm states.
        """
        # NOTE: if we slightly break `update` arg order, we could combine this with it, and allow offloading support
        # out of the box
        if not isinstance(self.layers[layer_idx], LinearAttentionCacheLayerMixin):
            raise TypeError("Cannot call `update_conv_state` on a non-LinearAttention layer!")
        recurrent_states = self.layers[layer_idx].update_recurrent_state(recurrent_states, state_idx, **kwargs)
        return recurrent_states

    def update_indexer(self, indexer_key_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Updates the indexer key cache for layer `layer_idx`.

        Parameters:
            indexer_key_states (`torch.Tensor`):
                The new indexer key states to cache, shape `[batch_size, seq_len, index_head_dim]`.
            layer_idx (`int`):
                The index of the layer to cache the states for.

        Return:
            `torch.Tensor`: The updated indexer key states (full cache).
        """
        if not hasattr(self.layers[layer_idx], "update_indexer"):
            raise ValueError(
                f"Cannot call `update_indexer` on layer {layer_idx} which is a "
                f"{type(self.layers[layer_idx]).__name__}; it has no indexer key cache "
                f"(expected a `DynamicIndexedLayer` or `StaticIndexedLayer`)."
            )
        return self.layers[layer_idx].update_indexer(indexer_key_states)

    def early_initialization(
        self,
        batch_size: int,
        num_heads: int | list[int],
        head_dim: int | list[int],
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Initialize all the layers in advance (it's otherwise lazily initialized on the first `update` call).
        This is useful for our `export` recipes, as `export` needs everything in advance.
        """
        # To allow different num_heads and head_dim depending on layers, we accept lists
        if isinstance(num_heads, int):
            num_heads = [num_heads] * len(self)
        if isinstance(head_dim, int):
            head_dim = [head_dim] * len(self)

        if len(num_heads) != len(self.layers):
            raise ValueError(
                f"`num_head` was provided as a list of length {len(num_heads)}, but the Cache currently has {len(self.layers)} layers"
            )
        if len(head_dim) != len(self.layers):
            raise ValueError(
                f"`head_dim` was provided as a list of length {len(num_heads)}, but the Cache currently has {len(self.layers)} layers"
            )

        for layer, layer_num_heads, layer_head_dim in zip(self.layers, num_heads, head_dim):
            if not layer.supports_early_init or layer.is_initialized:
                continue
            # Note that the initialization needs all dimensions (except -2), as well as device and dtype, so we use
            # this fake tensor approach. It has size 0 on the -2 dimension, so it does not allocate any data (it only
            # creates an empty tensor with correct shape, dtype and device), which is very efficient and practical
            fake_kv_tensor = torch.zeros((batch_size, layer_num_heads, 0, layer_head_dim), dtype=dtype, device=device)
            # Init the layer
            layer.lazy_initialization(fake_kv_tensor, fake_kv_tensor)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cache for the given layer."""
        if layer_idx >= len(self.layers):
            return 0

        # For alternating attention/linear attention  caches, `get_seq_length` needs to use attention layer idx when called with default layer_idx
        if not isinstance(self.layers[layer_idx], CacheLayerMixin):
            # If this is called with non-default arg, raise
            if layer_idx != 0:
                raise ValueError(
                    f"You called `get_seq_length` on layer index {layer_idx}, but this layer is a LinearAttention layer, which "
                    "does not track sequence length."
                )
            try:
                # Use the first attention layer
                layer_idx = next(idx for idx in range(len(self)) if isinstance(self.layers[idx], CacheLayerMixin))
            except StopIteration:
                raise ValueError(
                    "`get_seq_length` can only be called on Attention layers, and the current Cache seem to only contain "
                    "LinearAttention layers."
                )

        return self.layers[layer_idx].get_seq_length()

    def get_max_length(self, layer_idx: int | None = None) -> int:
        """
        Returns the maximum length of the cache. If `layer_idx` is not provided (default), this returns the maximum
        across all layers. Otherwise, return the maximum supported value for the given layer.
        A value of `-1` means no maximum, or undefined maximum, e.g. for dynamic attention layers that can grow indefinitely,
        or linear attention layer that do not have a sequence length dimension.
        """
        # For DynamicCache, where the layers are created at runtime
        if layer_idx is not None and layer_idx >= len(self.layers):
            return -1

        if layer_idx is None:
            return max(layer.get_max_length() for layer in self.layers)
        else:
            return self.layers[layer_idx].get_max_length()

    def has_previous_state(self, layer_idx: int | None = None, state_idx: int | None = None) -> bool:
        """Returns whether the LinearAttention layer at index `layer_idx` has previous state or not."""
        if layer_idx is not None and layer_idx >= len(self.layers):
            return False

        # In this case, use last LinearAttention layer
        if layer_idx is None:
            try:
                layer_idx = next(
                    idx
                    for idx in range(len(self) - 1, -1, -1)
                    if isinstance(self.layers[idx], LinearAttentionCacheLayerMixin)
                )
            except StopIteration:
                raise ValueError(
                    "`has_previous_state` can only be called on LinearAttention layers, and the current Cache seem to "
                    "only contain Attention layers."
                )
        elif not isinstance(self.layers[layer_idx], LinearAttentionCacheLayerMixin):
            raise ValueError(
                f"You called `has_previous_state` on layer index {layer_idx}, but this layer is an Attention layer, which "
                "does not support calling it."
            )

        # We may have several conv/recurrent states in the same layers. In this case, if `state_idx` is not provided, check if all
        # of them have previous state
        if state_idx is None:
            return all(self.layers[layer_idx].has_previous_state.values())
        return self.layers[layer_idx].has_previous_state[state_idx]

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
        """
        # For DynamicCache, where the layers are created at runtime -> if it was not yet created, the size is
        # simply the query_length
        if layer_idx >= len(self.layers):
            return query_length, 0

        # For alternating attention/linear attention caches, `get_mask_sizes` needs to use attention layer idx when called with default layer_idx
        if not isinstance(self.layers[layer_idx], CacheLayerMixin):
            # If this is called with non-default arg, raise
            if layer_idx != 0:
                raise ValueError(
                    f"You called `get_mask_sizes` on layer index {layer_idx}, but this layer is a LinearAttention layer, which "
                    "does not track sequence length."
                )
            try:
                # Use the first attention layer
                layer_idx = next(idx for idx in range(len(self)) if isinstance(self.layers[idx], CacheLayerMixin))
            except StopIteration:
                raise ValueError(
                    "`get_mask_sizes` can only be called on Attention layers, and the current Cache seem to only contain "
                    "LinearAttention layers."
                )

        return self.layers[layer_idx].get_mask_sizes(query_length)

    def get_query_offset(self, layer_idx: int = 0) -> int:
        """Returns the current offset of the query for the given `layer_idx`. It's always equal to the cache length, i.e.
        `get_seq_length(layer_idx)`, except for MTP layers.
        """
        # It's simply equal to the length of the past states, except in very specific cases, see `MtpCache`
        return self.get_seq_length(layer_idx=layer_idx)

    def reset(self):
        """Recursively reset all layers tensors"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].reset()

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder the cache for beam search"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].reorder_cache(beam_idx)

    def crop(self, tokens_to_remove: int) -> None:
        """
        Remove `tokens_to_remove` tokens from the current Cache. For layers that do not need to keep all the past states in memory,
        such as sliding window layers or linear attention layers, this will also restrict the size of the cached states back to their
        minimal working size. This means that `crop(0)` will not necessarily always be a no-op, as it may still remove useless states
        (i.e. states that are not needed for the next `forward`) from the Cache.
        """
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].crop(tokens_to_remove)

    def batch_repeat_interleave(self, repeats: int):
        """Repeat and interleave the cache"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        """Select indices from the cache"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].batch_select_indices(indices)

    def activate_past_recording(self):
        """
        Calling this function will activate past state recording, meaning that cache with fixed size such as a linear cache will
        wait for a call to `crop` before restricting the size of its cached states, in order to be able to retrieve previous full states.
        """
        for layer_idx in range(len(self.layers)):
            if hasattr(self.layers[layer_idx], "activate_past_recording"):
                self.layers[layer_idx].activate_past_recording()

    @property
    def batch_size(self) -> int:
        """Return the batch size of the cache, or ``-1`` if no layer has been initialized yet
        (e.g. an all-linear-attention cache queried before the first forward)."""
        # ``LinearAttentionLayer`` sets ``batch_size`` lazily — skip layers that haven't been
        # initialized yet (``generate`` queries this on a fresh cache during cache-reuse checks).
        values = [layer.batch_size for layer in self.layers if hasattr(layer, "batch_size")]
        if not values:
            return -1
        if len(set(values)) > 1:
            raise ValueError(f"The batch size is not consistent across layers: {values}")
        return values[0]

    @property
    def is_compileable(self) -> bool:
        """Return whether the cache is compilable"""
        # For DynamicCache dispatching the layers lazily (otherwise, all([]) is True)
        if len(self.layers) == 0:
            return False
        return all(layer.is_compileable for layer in self.layers)

    @property
    def is_initialized(self) -> bool:
        """Return whether the cache data is initialized"""
        layers = [layer for layer in self.layers if layer.supports_early_init]
        return len(layers) > 0 and all(layer.is_initialized for layer in layers)

    @property
    def is_croppable(self) -> bool:
        """Whether `crop` can put the whole cache back as it was, so a rollback leaves no trace."""
        return all(layer.is_croppable for layer in self.layers)

    @property
    def is_sliding(self) -> list[bool]:
        """Return whether the layers of the cache are sliding window"""
        return [getattr(layer, "is_sliding", False) for layer in self.layers]

    @property
    def is_linear(self) -> list[bool]:
        """Return whether the layers of the cache are linear attention (Mamba/SSM) layers. Note that layers containing
        both linear and full attention states will return False by this function"""
        return [
            isinstance(layer, LinearAttentionCacheLayerMixin)
            and not isinstance(layer, LinearAttentionAndFullAttentionLayer)
            for layer in self.layers
        ]

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        logger.warning_once(
            "`get_max_cache_shape` is deprecated, and will be removed in version 5.16. Please use `get_max_length` instead"
        )
        return self.get_max_length(layer_idx)

    @property
    def max_cache_len(self) -> int:
        logger.warning_once(
            "`max_cache_len` is deprecated, and will be removed in version 5.16. Please use `get_max_length()` instead"
        )
        return self.get_max_length()

    @property
    def max_batch_size(self) -> int:
        logger.warning_once(
            "`max_batch_size` is deprecated, and will be removed in version 5.16. Please use the simpler `batch_size` instead"
        )
        return self.batch_size


# Static Cache

class StaticLayer(CacheLayerMixin):
    """
    A static cache layer that stores the key and value states as static tensors of shape `[batch_size, num_heads, max_cache_len), head_dim]`.
    It lazily allocates its full backing tensors, and then mutates them in-place. Built for `torch.compile` support.

    Args:
        max_cache_len (`int`):
            Maximum number of tokens that can be stored, used for tensor preallocation.
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int, **kwargs):
        super().__init__()
        self.max_cache_len = max_cache_len
        # Very important that it's a tensor here, to avoid recompiling when we update it and use it to create positions
        self.cumulative_length = torch.tensor(0, dtype=int)

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """
        Lazy initialization of the keys and values tensors. This allows to get all properties (dtype, device,
        num_heads in case of TP etc...) at runtime directly, which is extremely practical as it avoids moving
        devices, dtypes etc later on for each `update` (which could break the static dynamo addresses as well).

        If this is unwanted, one can call `early_initialization(...)` on the Cache directly, which will call this
        function ahead-of-time (this is required for `torch.export` for example). It is also required whenever the
        prefill itself ends up in a compiled region (with chunked prefill for instance).
        """
        self.dtype, self.device = key_states.dtype, key_states.device
        self.batch_size, self.num_heads = key_states.shape[:2]
        self.v_head_dim = value_states.shape[-1]
        self.k_head_dim = key_states.shape[-1]

        self.keys = torch.zeros(
            (self.batch_size, self.num_heads, self.max_cache_len, self.k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.batch_size, self.num_heads, self.max_cache_len, self.v_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.cumulative_length = self.cumulative_length.to(self.device)
        # Note: `mark_static_address` is used to tag the tensors as a fixed data pointer, preventing compiled graph
        # breaks or cudagraph skips due to inplace mutations when updating the cache. However, it is not supported when
        # tracing the graph, so we skip it in this case. As prefill should never be compiled, this is not an issue and it
        # will still be run (except when users compile prefill explicitly, but this should be avoided!)
        # Without this, we cannot use cudagraphs!!
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)
            torch._dynamo.mark_static_address(self.cumulative_length)

        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        # Create a tensor to slice the static kv at the correct indices
        kv_length = key_states.shape[-2]
        cache_position = torch.arange(kv_length, device=self.device) + self.cumulative_length
        # Note that has to be performed in-place, as we have a static address that we need to keep
        self.cumulative_length.add_(kv_length)

        # Update the cache
        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            # Fallback for devices like MPS where index_copy_ might not be supported.
            self.keys[:, :, cache_position] = key_states
            self.values[:, :, cache_position] = value_states

        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        kv_offset = 0
        kv_length = self.max_cache_len
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length if self.is_initialized else 0

    def get_max_length(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.max_cache_len


class LinearAttentionLayer(LinearAttentionCacheLayerMixin):
    def lazy_initialization(
        self,
        conv_states: torch.Tensor | None = None,
        recurrent_states: torch.Tensor | None = None,
        state_idx: int = 0,
        conv_kernel_size: int | None = None,
    ) -> None:
        if conv_states is not None:
            if self.device is None:
                self.dtype, self.device = conv_states.dtype, conv_states.device
            # Even if prefill is larger/shorter than the conv_size, the tensor is usually either padded or truncated, except if
            # self.record_past is true and conv_kernel_size is provided explicitly
            conv_kernel_size = conv_states.shape[-1] if conv_kernel_size is None else conv_kernel_size
            self.conv_kernel_size[state_idx] = conv_kernel_size
            # The shape is always static, so we init as such
            self.conv_states[state_idx] = torch.zeros(
                (*conv_states.shape[:-1], conv_kernel_size),
                dtype=conv_states.dtype,
                device=conv_states.device,
            )
            # Mark as static address to be able to use cudagraphs
            if not is_torchdynamo_compiling() and not self.record_past:
                torch._dynamo.mark_static_address(self.conv_states[state_idx])
            self.is_conv_states_initialized[state_idx] = True

        if recurrent_states is not None:
            # The shape is always static, so we init as such
            self.recurrent_states[state_idx] = torch.zeros_like(recurrent_states)
            # Mark as static address to be able to use cudagraphs
            if not is_torchdynamo_compiling():
                torch._dynamo.mark_static_address(self.recurrent_states[state_idx])
            self.is_recurrent_states_initialized[state_idx] = True

    def update_conv_state(
        self, conv_states: torch.Tensor, state_idx: int = 0, conv_kernel_size: int | None = None, **kwargs
    ) -> torch.Tensor:
        """
        Update the linear attention cache in-place, and return the necessary conv states.

        Args:
            conv_states (`torch.Tensor`): The new conv states to cache.

        Returns:
            `torch.Tensor`: The updated conv states.
        """
        # Lazy initialization
        if not self.is_conv_states_initialized[state_idx]:
            self.lazy_initialization(conv_states=conv_states, state_idx=state_idx, conv_kernel_size=conv_kernel_size)

        # This is prefill, simply pad the conv_states if necessary
        if not self.has_previous_state[state_idx]:
            full_conv_states = conv_states
            self.has_previous_state[state_idx] = True
            # In this case, need to pad it to fit the conv_kernel_size
            if not self.record_past and full_conv_states.shape[-1] < self.conv_kernel_size[state_idx]:
                padding_length = self.conv_kernel_size[state_idx] - full_conv_states.shape[-1]
                full_conv_states = torch.nn.functional.pad(full_conv_states, (padding_length, 0), value=0)
        # We need to return the concatenation of the current state and the full new one so that the causal conv can see the
        # correct left context - however we usually cache only the last part
        else:
            full_conv_states = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)

        # Usually, keep only the last `conv_kernel_size` tokens
        if not self.record_past:
            # Copy instead of assigning to keep the static address
            self.conv_states[state_idx].copy_(full_conv_states[..., -self.conv_kernel_size[state_idx] :])
        # If we need to record the past, keep the full states for now to be able to rollback later
        else:
            self.conv_states[state_idx] = full_conv_states

        # Return full states no matter what
        return full_conv_states

    def update_recurrent_state(self, recurrent_states: torch.Tensor, state_idx: int = 0, **kwargs) -> torch.Tensor:
        """
        Update the linear attention cache in-place, and return the necessary ssm states.

        Args:
            smm_states (`torch.Tensor`): The new ssm states to cache.

        Returns:
            `torch.Tensor`: The updated ssm states.
        """
        if not self.is_recurrent_states_initialized[state_idx]:
            self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
        # Note that we copy instead of assigning, to preserve the static address for cudagraphs
        self.recurrent_states[state_idx].copy_(recurrent_states)
        return self.recurrent_states[state_idx]


# Dyanmic Cache

class DynamicLayer(CacheLayerMixin):
    """
    A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as tensors of shape `[batch_size, num_heads, seq_len, head_dim]`.
    """

    is_sliding = False
    is_croppable = True

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask"""
        kv_offset = 0
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]

    def get_max_length(self) -> int:
        """Returns the maximum sequence length of the cache object. DynamicLayer does not have a maximum length."""
        return -1

    def reset(self) -> None:
        """Resets the cache values while preserving the objects."""
        # Dropped rather than zeroed, as `update` grows them by concatenation. Clearing `is_initialized` first skips
        # the zeroing in `super`, which is still called to reset the `cumulative_length` of the inheriting layers.
        self.keys = self.values = None
        self.is_initialized = False
        super().reset()

    @deprecate_kwarg("max_length", new_name="tokens_to_remove", version="5.18")
    def crop(self, tokens_to_remove: int) -> None:
        """
        Remove `tokens_to_remove` tokens from the current cache layer.
        """
        # Legacy path: `tokens_to_remove` represents the final absolute size that the cache should have
        if tokens_to_remove > 0:
            logger.warning_once(
                "Calling `crop` with a positive value is deprecated and will be removed in version 5.18. Please use a negative "
                "integer to remove that number of tokens from the cache instead."
            )
            current_length = self.get_seq_length()
            # If the absolute value requested is larger than current length, just do nothing
            if tokens_to_remove >= current_length:
                return
            else:
                tokens_to_remove = self.get_seq_length() - tokens_to_remove

        # Nothing to do in this case
        if tokens_to_remove == 0:
            return

        # Crop the cache
        self.keys = self.keys[..., : -abs(tokens_to_remove), :]
        self.values = self.values[..., : -abs(tokens_to_remove), :]

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat the cache `repeats` times in the batch dimension."""
        if self.get_seq_length() > 0:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Only keep the `indices` in the batch dimension of the cache."""
        if self.get_seq_length() > 0:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]


class LinearAttentionAndFullAttentionLayer(LinearAttentionLayer, DynamicLayer):
    # The dynamic Attention part makes it non-compilable
    is_compileable = False

    def __init__(self, number_of_states: int = 1, **kwargs):
        DynamicLayer.__init__(self)
        LinearAttentionLayer.__init__(self, number_of_states=number_of_states)

    def lazy_initialization(self, *args, **kwargs) -> None:
        # When the Attention cache is used with `update`, `lazy_initialization` is called with 2 positional args
        if len(args) == 2 and len(kwargs) == 0:
            DynamicLayer.lazy_initialization(self, *args)
        # Otherwise, for the LinearAttention cache, when it's called in `update_conv_state` or `update_recurrent_state`, it's
        # always called with 1, 2 or 3 kwarg(s) (cause it needs to know if it's for the conv or ssm states)
        if len(args) == 0 and len(kwargs) in (1, 2, 3):
            LinearAttentionLayer.lazy_initialization(self, **kwargs)

    def offload(self):
        DynamicLayer.offload(self)
        LinearAttentionLayer.offload(self)

    def prefetch(self):
        DynamicLayer.prefetch(self)
        LinearAttentionLayer.prefetch(self)

    def reset(self) -> None:
        LinearAttentionLayer.reset(self)
        DynamicLayer.reset(self)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        LinearAttentionLayer.reorder_cache(self, beam_idx)
        DynamicLayer.reorder_cache(self, beam_idx)

    @deprecate_kwarg("max_length", new_name="tokens_to_remove", version="5.18")
    def crop(self, tokens_to_remove: int) -> None:
        LinearAttentionLayer.crop(self, tokens_to_remove)
        DynamicLayer.crop(self, tokens_to_remove)


# Mapping

# Mappings from layer_type to layer cache class
DYNAMIC_LAYER_TYPE_MAPPING = {
    "full_attention": DynamicLayer,
    # # From a cache point of view, sliding and chunked are the same in how they should behave, only the mask differs
    # "sliding_attention": DynamicSlidingWindowLayer,
    # "chunked_attention": DynamicSlidingWindowLayer,
    # "indexed_attention": DynamicIndexedLayer,
    # # Linear-attention-shaped placeholders (no per-token KV; recurrent state only).
    # # "conv" reuses the same cache shape as linear attention but stores a conv state buffer rather than recurrent SSM state
    "conv": LinearAttentionLayer,
    "linear_attention": LinearAttentionLayer,
    # # Hybrid layers carry both a linear-attention state and a dynamic-attention state.
    "hybrid": LinearAttentionAndFullAttentionLayer,
    # "hybrid_sliding": LinearAttentionAndSlidingWindowAttentionLayer,
    # Note: we want `moe` and `mlp` layers to be LinearAttentionLayer, so that we can correctly grab sequence length etc from
    # attention layers. Since they will stay empty (they don't need any cache), we don't want them to collide for mask creation etc
    # TODO: maybe use a dummy layer in those cases, or a dictionary {idx: Layer} for self.layers, so that we can skipthe indices
    # we don't need
    "moe": LinearAttentionLayer,
    "mlp": LinearAttentionLayer,
}
# Same but for StaticCache
STATIC_LAYER_TYPE_MAPPING = {
    "full_attention": StaticLayer,
    # # From a cache point of view, sliding and chunked are the same in how they should behave, only the mask differs
    # "sliding_attention": StaticSlidingWindowLayer,
    # "chunked_attention": StaticSlidingWindowLayer,
    # "indexed_attention": StaticIndexedLayer,
    # # LinearAttention layers are considered both static and dynamic (they are static, but are used as-is for any cache type)
    "conv": LinearAttentionLayer,
    "linear_attention": LinearAttentionLayer,
    # # Hybrid layers carry both a linear-attention state and a dynamic-attention state.
    # "hybrid": LinearAttentionAndStaticFullAttentionLayer,
    # "hybrid_sliding": LinearAttentionAndStaticSlidingWindowAttentionLayer,
    # Note: we want `moe` and `mlp` layers to be LinearAttentionLayer, so that we can correctly grab sequence length etc from
    # attention layers. Since they will stay empty (they don't need any cache), we don't want them to collide for mask creation etc
    # TODO: maybe use a dummy layer in those cases, or a dictionary {idx: Layer} for self.layers, so that we can skipthe indices
    # we don't need
    "moe": LinearAttentionLayer,
    "mlp": LinearAttentionLayer,
}
