from __future__ import annotations

import math
import copy
import logging
import os
import re
import shutil
import sys
import typing as t
from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from .config import ModelConfig
from .files import ensure_file
from .loading import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    load_state_dict,
)

logger = logging.getLogger(__name__)

def _compute_linear_scaling_rope_parameters(
    config: "ModelConfig" | None = None,
    device: "torch.device" | None = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with linear scaling. Credits to the Reddit user /u/kaiokendev
    """
    # Sloppy workaround for per-layer-config in gemma4 family which raises error for other models (DS4)
    try:
        config = config.per_layer_config[layer_type] if layer_type is not None else config
    except ValueError:
        pass

    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
    factor = rope_parameters_dict["factor"]

    # Gets the default RoPE parameters
    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    # Then applies linear scaling to the frequencies.
    # NOTE: originally, scaling was applied to the position_ids. However, we get `embs = inv_freq @ position_ids`, so
    # applying scaling to the inverse frequencies is equivalent.
    inv_freq /= factor
    return inv_freq, attention_factor


def _compute_proportional_rope_parameters(
    config: t.Optional["ModelConfig"] = None,
    device: t.Optional["torch.device"] = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
    head_dim_key: str = "head_dim",
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with proportional RoPE.
    """
    # Sloppy workaround for per-layer-config in gemma4 family which raises error for other models (DS4)
    try:
        config = config.per_layer_config[layer_type] if layer_type is not None else config
    except ValueError:
        pass

    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    head_dim = getattr(config, head_dim_key, None) or config.hidden_size // config.num_attention_heads
    base = rope_parameters_dict["rope_theta"]
    factor = rope_parameters_dict.get("factor", 1.0)
    rope_proportion = rope_parameters_dict.get("partial_rotary_factor", 1.0)

    attention_factor = 1.0  # Unused in this type of RoPE

    rope_angles = int(rope_proportion * head_dim // 2)

    inv_freq_rotated = 1.0 / (
        base
        ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / head_dim)
    )

    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        inv_freq = torch.cat(
            (
                inv_freq_rotated,
                torch.zeros(nope_angles, dtype=torch.float32, device=device),
            ),
            dim=0,
        )
    else:
        inv_freq = inv_freq_rotated

    inv_freq /= factor
    return inv_freq, attention_factor


def _compute_dynamic_ntk_parameters(
    config: t.Optional["ModelConfig"] = None,
    device: t.Optional["torch.device"] = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla
    """
    # Sloppy workaround for per-layer-config in gemma4 family which raises error for other models (DS4)
    try:
        config = config.per_layer_config[layer_type] if layer_type is not None else config
    except ValueError:
        pass

    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    factor = rope_parameters_dict["factor"]
    attention_factor = 1.0  # Unused in this type of RoPE

    # seq_len: default to max_position_embeddings, e.g. at init time
    if seq_len is None:
        seq_len = config.max_position_embeddings
    elif isinstance(seq_len, torch.Tensor):
        seq_len = torch.maximum(
            seq_len,
            torch.tensor(config.max_position_embeddings, dtype=seq_len.dtype, device=seq_len.device),
        )
    else:
        seq_len = max(seq_len, config.max_position_embeddings)

    # Compute the inverse frequencies
    base = base * ((factor * seq_len / config.max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor

def _compute_yarn_parameters(
    config: "ModelConfig",
    device: t.Optional["torch.device"] = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with NTK scaling. Please refer to the
    [original paper](https://huggingface.co/papers/2309.00071)
    """
    # Sloppy workaround for per-layer-config in gemma4 family which raises error for other models (DS4)
    try:
        config = config.per_layer_config[layer_type] if layer_type is not None else config
    except ValueError:
        pass

    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    factor = rope_parameters_dict["factor"]
    attention_factor = rope_parameters_dict.get("attention_factor")
    mscale = rope_parameters_dict.get("mscale")
    mscale_all_dim = rope_parameters_dict.get("mscale_all_dim")
    original_max_position_embeddings = rope_parameters_dict["original_max_position_embeddings"]

    # NOTE: DeepSeek-V3 (and potentially other models) have `original_max_position_embeddings` field
    # containing the pretrained value. They use the ratio between `max_position_embeddings` and this value
    # to compute the default attention scaling factor, instead of using `factor`.
    if factor is None:
        factor = config.max_position_embeddings / original_max_position_embeddings

    def get_mscale(scale, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    # Sets the attention factor as suggested in the paper
    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
        else:
            attention_factor = get_mscale(factor)

    # Optional config options
    # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
    beta_fast = rope_parameters_dict.get("beta_fast") or 32
    beta_slow = rope_parameters_dict.get("beta_slow") or 1

    # Compute the inverse frequencies
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        """Inverse dimension formula to find the dimension based on the number of rotations"""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate):
        """Find dimension range bounds based on rotations"""
        low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
        high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
    # to expand the possible context length. In other words, interpolation = apply scaling factor.
    pos_freqs = base ** (torch.arange(0, dim, 2).to(device=device, dtype=torch.float) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    truncate = config.rope_parameters.get("truncate", True)
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate)

    # Get n-dimensional rotational scaling corrected for extrapolation
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(device=device, dtype=torch.float)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return inv_freq, attention_factor

def _compute_longrope_parameters(
    config: "ModelConfig",
    device: t.Optional["torch.device"] = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with LongRoPE scaling. Please refer to the
    [original implementation](https://github.com/microsoft/LongRoPE)
    """
    # Sloppy workaround for per-layer-config in gemma4 family which raises error for other models (DS4)
    try:
        config = config.per_layer_config[layer_type] if layer_type is not None else config
    except ValueError:
        pass

    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    long_factor = rope_parameters_dict["long_factor"]
    short_factor = rope_parameters_dict["short_factor"]
    factor = rope_parameters_dict.get("factor")
    attention_factor = rope_parameters_dict.get("attention_factor")
    original_max_position_embeddings = rope_parameters_dict["original_max_position_embeddings"]

    # NOTE: Phi3 (and potentially other models) modify `max_position_embeddings` and have a
    # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
    # values to compute the default attention scaling factor, instead of using `factor`.
    if factor is None:
        factor = config.max_position_embeddings / original_max_position_embeddings

    # Sets the attention factor as suggested in the paper
    if attention_factor is None:
        if factor <= 1.0:
            attention_factor = 1.0
        else:
            attention_factor = math.sqrt(1 + math.log(factor) / math.log(original_max_position_embeddings))

    # Compute the inverse frequencies -- scaled based on the target sequence length
    if seq_len and seq_len > original_max_position_embeddings:
        ext_factors = torch.tensor(long_factor, dtype=torch.float32, device=device)
    else:
        ext_factors = torch.tensor(short_factor, dtype=torch.float32, device=device)
    inv_freq_shape = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim
    inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)

    return inv_freq, attention_factor

def _compute_llama3_parameters(
    config: "ModelConfig",
    device: t.Optional["torch.device"] = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies for llama 3.1.
    """
    # Sloppy workaround for per-layer-config in gemma4 family which raises error for other models (DS4)
    try:
        config = config.per_layer_config[layer_type] if layer_type is not None else config
    except ValueError:
        pass

    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    # Gets the default RoPE parameters
    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    factor = rope_parameters_dict["factor"]  # `8` in the original implementation
    low_freq_factor = rope_parameters_dict["low_freq_factor"]  # `1` in the original implementation
    high_freq_factor = rope_parameters_dict["high_freq_factor"]  # `4` in the original implementation
    old_context_len = rope_parameters_dict["original_max_position_embeddings"]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor

ROPE_INIT_FUNCTIONS: dict[str, t.Callable[..., tuple["torch.Tensor", float]]] = {
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "yarn": _compute_yarn_parameters,
    "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,
    "proportional": _compute_proportional_rope_parameters,
}

# For I/O bound operations (i.e. here reading files), it is better to have fewer threads, e.g. 4 is a good default.
# Having too many is actually harming performances quite a lot, i.e. using 16 can sometimes lead to taking TWICE
# as much time to load the same model
GLOBAL_WORKERS = min(4, os.cpu_count() or 4)

# Record all the torch primitives in advance, so that we can use them without them being modified when we patch torch
# in context managers
TORCH_INIT_FUNCTIONS = {
    "uniform_": torch.nn.init.uniform_,
    "normal_": torch.nn.init.normal_,
    "constant_": torch.nn.init.constant_,
    "ones_": torch.nn.init.ones_,
    "zeros_": torch.nn.init.zeros_,
    "eye_": torch.nn.init.eye_,
    "dirac_": torch.nn.init.dirac_,
    "xavier_uniform_": torch.nn.init.xavier_uniform_,
    "xavier_normal_": torch.nn.init.xavier_normal_,
    "kaiming_uniform_": torch.nn.init.kaiming_uniform_,
    "kaiming_normal_": torch.nn.init.kaiming_normal_,
    "trunc_normal_": torch.nn.init.trunc_normal_,
    "orthogonal_": torch.nn.init.orthogonal_,
    "sparse_": torch.nn.init.sparse_,
}


# Here, we need to check several modules imported, and hot patch all of them, as sometimes torch does
# something like `from torch.nn.init import xavier_uniform_` in their internals (e.g in torch.nn.modules.activations,
# where MultiHeadAttention lives), so the function name is binded at import time and just doing
# `setattr(torch.nn.init, name, globals()[name])` is thus not enough
# The following list should be enough for all torch versions we work with
TORCH_MODULES_TO_PATCH = (
    "torch.nn.init",
    "torch.nn.modules.activation",
    "torch.nn.modules.transformer",
    "torch.nn.modules.linear",
    "torch.nn.modules.loss",
    "torch.nn.modules.batchnorm",
    "torch.nn.modules.conv",
    "torch.nn.modules.normalization",
    "torch.nn.modules.rnn",
    "torch.nn.modules.sparse",
)

def uniform_(
    tensor: torch.Tensor, a: float = 0.0, b: float = 1.0, generator: torch.Generator | None = None
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["uniform_"](tensor, a=a, b=b, generator=generator)
    return tensor


def normal_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, generator: torch.Generator | None = None
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["normal_"](tensor, mean=mean, std=std, generator=generator)
    return tensor


def constant_(tensor: torch.Tensor, val: float) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["constant_"](tensor, val=val)
    return tensor


def ones_(tensor: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["ones_"](tensor)
    return tensor


def zeros_(tensor: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["zeros_"](tensor)
    return tensor


def eye_(tensor: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["eye_"](tensor)
    return tensor


def dirac_(tensor: torch.Tensor, groups: int = 1) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["dirac_"](tensor, groups=groups)
    return tensor


def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["xavier_uniform_"](tensor, gain=gain, generator=generator)
    return tensor


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["xavier_normal_"](tensor, gain=gain, generator=generator)
    return tensor


def kaiming_uniform_(
    tensor: torch.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["kaiming_uniform_"](
            tensor, a=a, mode=mode, nonlinearity=nonlinearity, generator=generator
        )
    return tensor


def kaiming_normal_(
    tensor: torch.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["kaiming_normal_"](
            tensor, a=a, mode=mode, nonlinearity=nonlinearity, generator=generator
        )
    return tensor


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["trunc_normal_"](tensor, mean=mean, std=std, a=a, b=b, generator=generator)
    return tensor


@contextmanager
def guard_torch_init_functions():
    """
    Guard the `torch.nn.init` primitive functions to behave exactly like the functions in this file, i.e. be
    protected against the `_is_hf_initialized` flag to avoid re-init if the param was already loaded.

    Usually, all models are using the init from `transformers` which are already guarded, but just to make extra sure
    and for remote code, we also use this context manager.
    """
    originals = defaultdict(dict)
    try:
        # Replace all torch funcs by the ones in this file
        for module_name in TORCH_MODULES_TO_PATCH:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                for func_name in TORCH_INIT_FUNCTIONS.keys():
                    if hasattr(module, func_name):
                        originals[module][func_name] = getattr(module, func_name)
                        setattr(module, func_name, globals()[func_name])
        yield
    finally:
        # Set back the original functions on all modules
        for module, functions in originals.items():
            for func_name, func in functions.items():
                setattr(module, func_name, func)


def orthogonal_(
    tensor: torch.Tensor,
    gain: float = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        # The QR decomposition (`geqrf`) used by `torch.nn.init.orthogonal_` is only implemented for
        # float32/float64, so for lower-precision dtypes we run the init in float32 and copy back
        if tensor.is_floating_point() and torch.finfo(tensor.dtype).bits < 32:
            fp32_tensor = torch.empty_like(tensor, dtype=torch.float32)
            TORCH_INIT_FUNCTIONS["orthogonal_"](fp32_tensor, gain=gain, generator=generator)
            with torch.no_grad():
                return tensor.copy_(fp32_tensor)
        return TORCH_INIT_FUNCTIONS["orthogonal_"](tensor, gain=gain, generator=generator)
    return tensor


def sparse_(
    tensor: torch.Tensor, sparsity: float, std: float = 0.01, generator: torch.Generator | None = None
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["sparse_"](tensor, sparsity=sparsity, std=std, generator=generator)
    return tensor


def copy_(tensor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        with torch.no_grad():
            return tensor.copy_(other)
    return tensor


class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `fastcore` library.
    """

    def __init__(
        self,
        context_managers: list[AbstractContextManager]
    ):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)


@contextmanager
def local_torch_dtype(
    dtype: torch.dtype,
    model_class_name: str | None = None
):
    """
    Locally change the torch default dtype to `dtype`, and restore the old one upon exiting the context.
    If `model_class_name` is provided, it's used to provide a more helpful error message if `dtype` is not valid.
    """
    # Just a more helping error before we set `torch.set_default_dtype` later on which would crash in this case
    if not dtype.is_floating_point:
        if model_class_name is not None:
            error_message = (
                f"{model_class_name} cannot be instantiated under `dtype={dtype}` as it's not a floating-point dtype"
            )
        else:
            error_message = f"Cannot set `{dtype}` as torch's default as it's not a floating-point dtype"
        raise ValueError(error_message)

    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(original_dtype)


def _get_dtype(
    dtype: str | torch.dtype | dict | None,
    files: list[Path] | None,
    config: ModelConfig,
    weights_only: bool,
) -> tuple[ModelConfig, torch.dtype]:
    """Find the correct `dtype` to use based on provided arguments. Also update the `config` based on the
    inferred dtype. We do the following:
    1. If dtype is "auto", we try to read the config, else auto-detect dtype from the loaded state_dict, by checking
    its first weights entry that is of a floating type - we assume all floating dtype weights are of the same dtype
    2. Else, use the dtype provided as a dict or str
    """
    if dtype is not None:
        if isinstance(dtype, str):
            if dtype == "auto":
                if hasattr(config, "dtype") and config.dtype is not None:
                    dtype = config.dtype
                    logger.info(f"Will use dtype={dtype} as defined in model's config object")
                else:
                    if files is not None and str(files[0]).endswith(".gguf"):
                        dtype = None
                    else:
                        raise NotImplementedError
                        state_dict = load_state_dict(
                            files[0],
                            map_location="meta",
                            weights_only=weights_only
                        )
                        dtype = get_state_dict_dtype(state_dict)
                    if dtype is not None:
                        logger.info(
                            f"Since the `dtype` attribute can't be found in model's config object, "
                            f"will use dtype={dtype} as derived from model's weights"
                        )
            elif hasattr(torch, dtype):
                dtype = getattr(torch, dtype)
            else:
                raise ValueError(
                    "`dtype` provided as a `str` can only be `'auto'`, or a string representation of a valid `torch.dtype`"
                )

            # cast it to a proper `torch.dtype` object
            dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        elif not isinstance(dtype, (dict, torch.dtype)):
            raise ValueError(
                f"`dtype` can be one of: `torch.dtype`, `'auto'`, a string of a valid `torch.dtype` or a `dict` with valid `dtype` "
                f"for each sub-config in composite configs, but received {dtype}"
            )
    else:
        # set torch.get_default_dtype() (usually fp32) as the default dtype if `None` is provided
        dtype = torch.get_default_dtype()

    # Get the main dtype
    if isinstance(dtype, dict):
        main_dtype = dtype.get("", torch.get_default_dtype())
        main_dtype = getattr(torch, main_dtype) if isinstance(main_dtype, str) else main_dtype

        logger.warning_once(
            "Using different dtypes per module is deprecated and will be removed in future versions "
            "Setting different dtypes per backbone model might cause device errors downstream, therefore "
            f"setting the dtype={main_dtype} for all modules."
        )

    else:
        main_dtype = dtype

    # Set it on the config and subconfigs
    config.dtype = main_dtype
    for sub_config_key in config.sub_configs:
        raise NotImplementedError
        if (sub_config := getattr(config, sub_config_key)) is not None:
            sub_config.dtype = main_dtype

    return config, main_dtype


def process_target_pattern(pattern: str) -> tuple[str, str | None]:
    """
    Process a target pattern for reverse mapping (when targets become sources).

    This handles several edge cases in checkpoint conversion mappings:
    - Removes `^` prefix and `$` suffix (start/end of string anchors)
    - Removes negative lookahead/lookbehind assertions
    - Detects capturing groups and replaces them with `\\1` backreference

    Args:
        pattern: The target pattern to process for reverse mapping.

    Returns:
        A tuple of (processed_pattern, captured_group) where captured_group is
        the original capturing group found (e.g., "(encoder|decoder)") or None.
    """
    # Some mapping contains `^` to notify start of string when matching -> remove it during reverse mapping
    pattern = pattern.removeprefix("^")
    # Some mapping contains `$` to notify end of string when matching -> remove it during reverse mapping
    pattern = pattern.removesuffix("$")
    # Remove negative lookahead/behind if any. This is ugly but needed for reverse mapping of
    # Qwen2.5, Sam3, Ernie4.5 VL MoE! It needs to be non greedy in case there are several
    pattern = re.sub(r"\(\?.+?\)?\)", "", pattern)
    # Remove the backslash for literal dots
    pattern = pattern.replace(r"\.", ".")
    # Allow capturing groups in patterns, i.e. to add/remove a prefix to all keys (e.g. timm_wrapper, sam3)
    capturing_group_match = re.search(r"\(.+?\)", pattern)
    captured_group = None
    if capturing_group_match:
        captured_group = capturing_group_match.group(0)
        pattern = pattern.replace(captured_group, r"\1", 1)
    return pattern, captured_group


def process_source_pattern(source_pattern: str, target_pattern: str) -> str:
    """
    Process a source pattern for reverse mapping (when sources become targets).
    This is useful because usually if the original source (so now the target in reverse mode) had a `^` or `$`
    to restrict to start/end of string, we should do the same in reverse mode. This is why this method is conditioned
    on the target pattern, we want to do it only for pairs (source, target) when the original source (so the current target
    in reverse mode) had it.
    """
    if target_pattern.startswith("^"):
        source_pattern = f"^{source_pattern}" if not source_pattern.startswith("^") else source_pattern
    if target_pattern.endswith("$"):
        source_pattern = f"{source_pattern}$" if not source_pattern.endswith("$") else source_pattern

    return source_pattern


class WeightTransform:
    # Restrict the attributes that can be attached
    __slots__ = (
        "source_patterns",
        "target_patterns",
        "compiled_sources",
        
        "_original_source_patterns",
        "_original_target_patterns",
        "_was_used",
        "scope_prefix",
        "base_model_prefix",
    )
    
    def __init__(
        self,
        source_patterns: str | list[str],
        target_patterns: str | list[str]
    ):
        self.source_patterns: list[str] = source_patterns
        self.target_patterns: list[str] = target_patterns
        # Those are needed to be able to reverse correctly the transform, as the patterns may be processed
        self._original_source_patterns = self.source_patterns.copy()
        self._original_target_patterns = self.target_patterns.copy()

        # Flag to notice if the Transform was used
        self._was_used = False
        
        # Optional scope_prefix/base_model_prefix. When used, the transform will only match and apply to keys containing
        # either `base_model_prefix.scope_prefix.` or `scope_prefix.` prefixes
        self.scope_prefix: str | None = None
        self.base_model_prefix: str | None = None

        # We need to process a few exceptions here when instantiating the reverse mapping (i.e. the targets become
        # sources, and sources become targets). The issues lie in the sources usually, so here we need to check the
        # targets for the reversed mapping

        # Process target_patterns: detect capturing groups and replace with \1
        # Store the original capturing group patterns for reverse mapping
        target_capturing_groups: list[str] = []
        for i, pattern in enumerate(self.target_patterns):
            self.target_patterns[i], captured_group = process_target_pattern(pattern)
            if captured_group is not None:
                target_capturing_groups.append(captured_group)

        # Validate that we only have one unique capturing group pattern across all targets
        # This ensures deterministic reverse mapping when sources have \1 backreferences
        unique_capturing_groups = set(target_capturing_groups)
        if len(unique_capturing_groups) > 1:
            raise ValueError(
                f"Multiple different capturing groups found in target_patterns: {unique_capturing_groups}. "
                f"All target patterns must use the same capturing group pattern."
            )
        unique_capturing_group = unique_capturing_groups.pop() if unique_capturing_groups else None

        # We also need to check capturing groups in the sources during reverse mapping (e.g. timm_wrapper, sam3)
        for i, pattern in enumerate(self.source_patterns):
            # Replace capturing groups
            if r"\1" in pattern:
                if unique_capturing_group is None:
                    raise ValueError(
                        f"Source pattern '{pattern}' contains \\1 backreference, but no capturing groups "
                        f"found in target_patterns."
                    )
                # Use the unique capturing group from target_patterns for all sources
                pattern = pattern.replace(r"\1", unique_capturing_group, 1)
            # Potentially process a bit more for consistency - only if they are consistent pairs, i.e. the length is the same
            if len(self.source_patterns) == len(self.target_patterns):
                pattern = process_source_pattern(pattern, self._original_target_patterns[i])
            self.source_patterns[i] = pattern

        # Construct the regex we will use to rename keys from the sources to the targets
        branches = []
        for i, source_pattern in enumerate(self.source_patterns):
            group_name = f"g{i}"
            pattern = source_pattern.replace("*.", r".*\.")
            branches.append(f"(?P<{group_name}>{pattern})")
        self.compiled_sources = re.compile("|".join(branches))


    def __setattr__(self, name, value):
        if name in ("source_patterns", "target_patterns"):
            # We do not allow to re-set the patterns, as they are linked between each other and changing one
            # without the other can mess-up with the capturing groups/compiled sources
            if hasattr(self, name):
                raise ValueError(f"Cannot assign to field {name}, you should create a new instance")
            # Switch str to list
            elif isinstance(value, str):
                value = [value]
        object.__setattr__(self, name, value)

    def _scoped_match(self, source_key: str) -> tuple[str | None, str, re.Match[str]] | None:
        """
        Strip `scope_prefix` (if any) from `source_key`, then match `compiled_sources` against the
        remaining suffix.

        Returns `(prefix_dot, key_to_match, match_object)` on match, else `None`. `prefix_dot` is
        the prefix consumed from `source_key`: either `f"{scope_prefix}."` or that same string with
        one `base_model_prefix` level stripped or prepended when the former didn't match.
        `None` when `scope_prefix` is unset.
        """
        key_to_match = source_key
        prefix = None
        if self.scope_prefix is not None:
            scope_prefix = f"{self.scope_prefix}." if self.scope_prefix != "" else ""
            base_model_prefix = f"{self.base_model_prefix}." if self.base_model_prefix != "" else ""
            # First, try to match the longest sequence, i.e. base_model_prefix + scope_prefix
            if source_key.startswith(base_model_prefix + scope_prefix):
                prefix = base_model_prefix + scope_prefix
            # Then, try to strip the base_model_prefix, in case we load a ForXXX model from BaseModel weights
            elif source_key.startswith(scope_prefix):
                prefix = scope_prefix
            # In this case, no match is ever possible
            else:
                return None
            key_to_match = source_key.removeprefix(prefix)

        match_object = self.compiled_sources.search(key_to_match)
        if match_object is None:
            return None
        return (prefix, key_to_match, match_object)


    def rename_source_key(self, source_key: str) -> tuple[str, str | None]:
        """
        Return a tuple (renamed_key, source_pattern_producing_the_match).
        Try renaming `source_key` according to the source and target patterns of the current WeightTransform.
        In case of a one-to-many transform, i.e. we have several target patterns, the matching source pattern
        will be replaced by the first of all the target patterns (they are then correctly expanded in the Operations).
        """
        matched = self._scoped_match(source_key)
        if matched is None:
            return source_key, None

        prefix_dot, key_to_match, match_object = matched

        # We have a match, so the Transform was used
        self._was_used = True

        # Find the source that produced the match (it's the first group that matched, as the search stops after first branch match)
        matching_group_name = next(name for name, val in match_object.groupdict().items() if val is not None)
        source_pattern_that_matched = self.source_patterns[int(matching_group_name[1:])]
        # If we matched, we always replace with the first target pattern, in case we have several (one to many transform)
        replacement = self.target_patterns[0]
        # Allow capturing groups in patterns, i.e. to add a prefix to all keys (e.g. timm_wrapper, sam3)
        if r"\1" in replacement:
            # The index of the internal group we need to replace is the index of the matched named group as it comes
            # inside that matched named group
            replaced_group_idx = self.compiled_sources.groupindex[matching_group_name] + 1
            replacement = replacement.replace(r"\1", match_object.group(replaced_group_idx))
        renamed_key = key_to_match.replace(match_object.group(0), replacement, 1)
        if prefix_dot is not None:
            renamed_key = prefix_dot + renamed_key
        return renamed_key, source_pattern_that_matched

    def was_used(self) -> bool:
        """
        Return whether the current Transform matched any weights during loading/saving. This is needed as some
        weight renaming transforms are not bijective, i.e. if we drop/add full parts of a name with PrefixChange, we
        lose some information that we cannot get back if we don't know if the Transform was used before already (say we
        have a prefix to drop, we need to know whether the checkpoints we loaded before contained the said prefix or not
        before adding it back, or not, during saving).
        """
        return self._was_used

class WeightRenaming(WeightTransform):
    # Special case of WeightTransform that only renames keys without any conversion.

    # Needs to be empty, otherwise the class will not be slotted
    __slots__ = ()


class WeightConverter(WeightTransform):
    __slots__ = ("operations", "force_cpu")

    def __init__(
        self,
        source_patterns: str | list[str],
        target_patterns: str | list[str],
        operations: list[ConversionOps],
        force_cpu: bool = False,
    ):
        super().__init__(source_patterns, target_patterns)
        

def _build_checkpoint_conversion_mapping():
    mapping = {
        "legacy": [
            WeightRenaming(
                source_patterns="LayerNorm.gamma",
                target_patterns="LayerNorm.weight",
            ),
            WeightRenaming(
                source_patterns="LayerNorm.beta",
                target_patterns="LayerNorm.bias",
            ),
        ],
    }

    mapping["legacy"] += [
        WeightRenaming(
            source_patterns=".weight_g$",
            target_patterns=".parametrizations.weight.original0",
        ),
        WeightRenaming(
            source_patterns=".weight_v$",
            target_patterns=".parametrizations.weight.original1",
        ),
    ]

    return mapping
    
_checkpoint_conversion_mapping_cache = None

def get_checkpoint_conversion_mapping(model_type):
    global _checkpoint_conversion_mapping_cache
    if _checkpoint_conversion_mapping_cache is None:
        _checkpoint_conversion_mapping_cache = _build_checkpoint_conversion_mapping()
    return copy.deepcopy(_checkpoint_conversion_mapping_cache.get(model_type))

    
def get_model_conversion_mapping(
    model: Model,
    add_legacy: bool = True,
) -> list[WeightTransform]:
    """
    Collect the ordered list of weight transforms for `model` (used during
    loading and, when reversed, during saving).

    Each `PreTrainedModel` sub-module is looked up by class name then
    `model_type`.  Root transforms are applied globally; sub-module transforms
    have their `scope_prefix` set so they only match keys under that prefix.  After any
    sub-module is processed, both its class name and `model_type` are marked
    seen to prevent `XForY` / `XModel` pairs from applying the same mapping
    twice via different lookup paths.
    """

    # note: this function is used in PEFT, so changing the API requires coordination
    weight_conversions = []

    # Maps each identifier (class name or model_type) to the module paths that have
    # already claimed it.  A later module is skipped only when one of those paths is
    # an ancestor of the current module path — siblings are never ancestors of each
    # other, so two sibling sub-models with the same model_type both get their own
    # scoped transforms.  A child is an ancestor of everything nested under it, which
    # prevents a parent's transforms from being duplicated with a scoped copy for the child.
    seen_identifiers: defaultdict[str, list[str]] = defaultdict(list)

    for module_name, submodule in model.named_modules():
        # Skip if it's not a submodel
        if not isinstance(submodule, Model):
            continue

        class_name = type(submodule).__name__
        model_type = submodule.config.model_type

        # Skip it if it's custom code and it was NOT registered by the user directly: it may have the same `model_type`/ClassName
        # as a native model inside the library, but it uses custom modeling so it should not share the conversions
        # if (
        #     submodule.is_custom_code()
        #     and class_name not in USER_REGISTERED_MAPPINGS
        #     and model_type not in USER_REGISTERED_MAPPINGS
        # ):
        #     continue

        # Skip if an ancestor already claimed this class (its unscoped transforms already cover this subtree).
        if any(seen == "" or module_name.startswith(seen + ".") for seen in seen_identifiers[class_name]):
            continue

        # Class name takes priority — a class-specific mapping bypasses the model_type
        # deduplication check (e.g. LlavaModel nested inside LlavaForConditionalGeneration
        # must still get its own scoped mapping even after "llava" is marked seen).
        conversions = get_checkpoint_conversion_mapping(class_name)
        found_via_class = conversions is not None
        
        if not found_via_class:
            # Same ancestor check as above, but via model_type for modules without a class-specific mapping.
            if model_type and any(
                seen == "" or module_name.startswith(seen + ".") for seen in seen_identifiers[model_type]
            ):
                continue
            if model_type is not None:
                conversions = get_checkpoint_conversion_mapping(model_type)

        if conversions is None:
            continue

        is_root_model = module_name == ""
        if not is_root_model:
            raise NotImplementedError(f"{module_name} - {submodule}\n : {conversions}")
            # Scope each transform so it only matches keys under this sub-module's prefix - but we still allow to
            # arbitrary add/remove base_model_prefix to load ForXXX model from BaseModel and the opposite
            # Note that we need 2 removeprefix calls here, as only one level of nesting would not have the ending dot to module_name
            scope_prefix = module_name.removeprefix(model.base_model_prefix)
            scope_prefix = scope_prefix.removeprefix(".")
            for transform in conversions:
                transform.scope_prefix = scope_prefix
                transform.base_model_prefix = model.base_model_prefix
        raise NotImplementedError(f"{module_name} - {submodule}\n : {conversions}")
        weight_conversions.extend(conversions)

        seen_identifiers[class_name].append(module_name)
        # Only record model_type when the hit was via model_type. When the hit was via
        # class name, other sub-modules sharing the same model_type but without a
        # class-specific mapping (e.g. DetrModel under DetrForSegmentation) must still
        # be reachable so their base transforms are picked up and scoped.
        if not found_via_class and model_type:
            seen_identifiers[model_type].append(module_name)

    if add_legacy:
        weight_conversions.extend(get_checkpoint_conversion_mapping("legacy"))

    return weight_conversions


def get_torch_context_manager_or_global_device():
    """
    Test if a device context manager is currently in use, or if it is not the case, check if the default device
    is not "cpu". This is used to infer the correct device to load the model on, in case `device_map` is not provided.
    """
    device_in_context = torch.tensor([]).device
    default_device = torch.get_default_device()
    # This case means no context manager was used -> we still check if the default that was potentially set is not cpu
    if device_in_context == default_device:
        if default_device != torch.device("cpu"):
            return default_device
        return None
    return device_in_context


def check_and_set_device_map(
    device_map: "torch.device | int | str | dict | None"
) -> dict | str | None:
    # Potentially detect context manager or global device, and use it (only if no device_map was provided)
    if device_map is None:
        device_in_context = get_torch_context_manager_or_global_device()
        if device_in_context == torch.device("meta"):
            raise RuntimeError(
                "You are using `from_pretrained` with a meta device context manager or `torch.set_default_device('meta')`.\n"
                "This is an anti-pattern as `from_pretrained` wants to load existing weights.\nIf you want to initialize an "
                "empty model on the meta device, use the context manager or global device with `from_config`, or `ModelClass(config)`"
            )
        device_map = device_in_context

    # change device_map into a map if we passed an int, a str or a torch.device
    if isinstance(device_map, torch.device):
        device_map = {"": device_map}
    elif (
        isinstance(device_map, str) and
        device_map not in [
            "auto", "balanced",
            "balanced_low_0", "sequential"
        ]
    ):
        try:
            if device_map == "cuda":
                # setting to the local rank
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                device_map = f"cuda:{local_rank}"
            device_map = {"": torch.device(device_map)}
        except RuntimeError:
            raise ValueError(
                "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
                f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
            )
    elif isinstance(device_map, int):
        if device_map < 0:
            raise ValueError(
                "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
            )
        else:
            device_map = {"": device_map}

    if device_map is not None:
        if not is_accelerate_available():
            raise ValueError(
                "Using a `device_map`, `tp_plan`, `torch.device` context manager or setting `torch.set_default_device(device)` "
                "requires `accelerate`. You can install it with `pip install accelerate`"
            )
    return device_map


PALETTE = {
    "reset": "[0m",
    "red": "[31m",
    "yellow": "[33m",
    "orange": "[38;5;208m",
    "purple": "[35m",
    "green": "[32m",
    "bold": "[1m",
    "italic": "[3m",
    "dim": "[2m",
}

_DIGIT_RX = re.compile(r"(?<=\.)(\d+)(?=\.|$)")  # numbers between dots or at the end


def _pattern_of(key: str) -> str:
    """Replace every dot-delimited integer with '*' to get the structure."""
    return _DIGIT_RX.sub("*", key)


def _fmt_indices(values: list[int], cutoff=10) -> str:
    """Format a list of ints as single number, {a, ..., b}, or first...last."""
    if len(values) == 1:
        return str(values[0])
    values = sorted(values)
    if len(values) > cutoff:
        return f"{values[0]}...{values[-1]}"
    return ", ".join(map(str, values))


def update_key_name(mapping: dict[str, t.Any]) -> dict[str, t.Any]:
    """
    Merge keys like 'layers.0.x', 'layers.1.x' into 'layers.{0, 1}.x'
    BUT only merge together keys that have the exact same value.
    Returns a new dict {merged_key: value}.
    """
    # (pattern, value) -> list[set[int]] (per-star index values)
    not_mapping = False
    if not isinstance(mapping, dict):
        mapping = {k: k for k in mapping}
        not_mapping = True

    bucket: dict[str, list[set[int] | t.Any]] = defaultdict(list)
    for key, val in mapping.items():
        digs = _DIGIT_RX.findall(key)
        patt = _pattern_of(key)
        for i, d in enumerate(digs):
            if len(bucket[patt]) <= i:
                bucket[patt].append(set())
            bucket[patt][i].add(int(d))
        bucket[patt].append(val)

    out_items = {}
    for patt, values in bucket.items():
        sets, val = values[:-1], values[-1]
        parts = patt.split("*")  # stars are between parts
        final = parts[0]
        for i in range(1, len(parts)):
            if i - 1 < len(sets) and sets[i - 1]:
                insert = _fmt_indices(sorted(sets[i - 1]))
                if len(sets[i - 1]) > 1:
                    final += "{" + insert + "}"
                else:
                    final += insert
            else:
                final += "*"
            final += parts[i]

        out_items[final] = val
    out = OrderedDict(out_items)
    if not_mapping:
        return out.keys()
    return out

_ansi_re = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ansi_re.sub("", str(s))


def _pad(text, width):
    t = str(text)
    pad = max(0, width - len(_strip_ansi(t)))
    return t + " " * pad


def _make_table(rows, headers):
    # compute display widths while ignoring ANSI codes
    cols = list(zip(*([headers] + rows))) if rows else [headers]
    widths = [max(len(_strip_ansi(x)) for x in col) for col in cols]
    header_line = " | ".join(_pad(h, w) for h, w in zip(headers, widths))
    sep_line = "-+-".join("-" * w for w in widths)
    body = [" | ".join(_pad(c, w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join([header_line, sep_line] + body)


def _style(s, color):
    """Return color/style-formatted input `s` if `sys.stdout` is interactive, e.g. connected to a terminal."""
    if sys.stdout is not None and sys.stdout.isatty():
        return f"{PALETTE[color]}{s}{PALETTE['reset']}"
    else:
        return s


def _get_terminal_width(default=80):
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return default

def log_state_dict_report(
    model,
    pretrained_model_name_or_path: str,
    ignore_mismatched_sizes: bool,
    loading_info: LoadStateDictInfo,
    logger: logging.Logger | None = None,
):
    """
    Log a readable report about state_dict loading issues.

    This version is terminal-size aware: for very small terminals it falls back to a compact
    Key | Status view so output doesn't wrap badly.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Re-raise errors early if needed
    if loading_info.error_msgs:
        error_msg = "\n\t".join(loading_info.error_msgs)
        if "size mismatch" in error_msg:
            error_msg += (
                "\n\tYou may consider adding `ignore_mismatched_sizes=True` to `from_pretrained(...)` if appropriate."
            )
        raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

    # Pipeline-parallel details require walking the model state dict, so only collect them at info verbosity.
    report_model = model if logger.isEnabledFor(logging.INFO) else None
    report = loading_info.create_loading_report(report_model)
    if report is None:
        return

    prelude = f"{PALETTE['bold']}{model.__class__.__name__} LOAD REPORT{PALETTE['reset']} from: {pretrained_model_name_or_path}\n"

    # Log the report as warning
    logger.warning(prelude + report)

    # Re-raise in those case, after the report
    if loading_info.conversion_errors:
        raise RuntimeError(
            "We encountered some issues during automatic conversion of the weights. For details look at the `CONVERSION` entries of "
            "the above report!"
        )
    if not ignore_mismatched_sizes and loading_info.mismatched_keys:
        raise RuntimeError(
            "You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!"
        )


@dataclass(frozen=True)
class LoadStateDictConfig:
    """
    Config for loading weights. This allows bundling arguments that are just
    passed around.
    """

    pretrained_model_name_or_path: str | None = None
    
    use_safetensors: bool | None = None
    ignore_mismatched_sizes: bool = False
    sharded_metadata: dict | None = None
    device_map: dict | None = None
    disk_offload_folder: str | None = None
    offload_buffers: bool = False
    dtype: torch.dtype | None = None
    dtype_plan: dict = field(default_factory=dict)

    device_mesh: "DeviceMeshLike | None" = None
    weights_only: bool = True
    weight_mapping: list[WeightConverter | WeightRenaming] | None = None
    disable_mmap: bool | None = None

@dataclass
class LoadStateDictInfo:
    """
    Mutable container for state-dict loading results and diagnostics. Each entry in this structure is mutable,
    and will usually be mutated in-place during the loading pipeline.

    Attributes:
        missing_keys (`set[str]`):
            Keys that are missing from the loaded checkpoints but expected in the model's architecture.
        unexpected_keys (`set[str]`):
            Keys that are found in the checkpoints, but not expected in the model's architecture.
        skipped_pp_keys (`set[str]`):
            Checkpoint keys owned by another pipeline-parallel rank and intentionally not loaded here.
        mismatched_keys (`set[tuple[str, tuple[int], tuple[int]]]`):
            Keys that are found in the checkpoints and are expected in the model's architecture, but with a different shape.
        error_msgs ( `list[str]`):
            Some potential error messages.
        conversion_errors (`dict[str, str]`):
            Errors happening during the on-the-fly weight conversion process.
    """

    missing_keys: set[str]
    unexpected_keys: set[str]
    mismatched_keys: set[tuple[str, tuple[int], tuple[int]]]
    error_msgs: list[str]
    conversion_errors: dict[str, str]
    skipped_pp_keys: set[str]

    def missing_and_mismatched(self):
        """Return all effective missing keys, including `missing` and `mismatched` keys."""
        return self.missing_keys | {k[0] for k in self.mismatched_keys}


    def create_loading_report(self, model=None) -> str | None:
        """Generate the minimal table of a loading report."""
        term_w = _get_terminal_width()

        rows = []
        tips = "\n\nNotes:"

        if self.unexpected_keys:
            tips += f"\n- {_style('UNEXPECTED:', 'orange')}\t" + _style(
                "can be ignored when loading from different task/architecture; not ok if you expect identical arch.",
                "italic",
            )
            for k in update_key_name(self.unexpected_keys):
                status = _style("UNEXPECTED", "orange")
                rows.append([k, status, "", ""])

        if self.missing_keys:
            tips += f"\n- {_style('MISSING:', 'red')}\t" + _style(
                "those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.",
                "italic",
            )
            for k in update_key_name(self.missing_keys):
                status = _style("MISSING", "red")
                rows.append([k, status, ""])

        if self.mismatched_keys:
            tips += f"\n- {_style('MISMATCH:', 'yellow')}\t" + _style(
                "ckpt weights were loaded, but they did not match the original empty weight shapes.", "italic"
            )
            iterator = {a: (b, c) for a, b, c in self.mismatched_keys}
            for key, (shape_ckpt, shape_model) in update_key_name(iterator).items():
                status = _style("MISMATCH", "yellow")
                data = [
                    key,
                    status,
                    f"Reinit due to size mismatch - ckpt: {str(shape_ckpt)} vs model:{str(shape_model)}",
                ]
                rows.append(data)

        if self.conversion_errors:
            tips += f"\n- {_style('CONVERSION:', 'purple')}\t" + _style(
                "originate from the conversion scheme", "italic"
            )
            for k, v in update_key_name(self.conversion_errors).items():
                status = _style("CONVERSION", "purple")
                _details = f"\n\n{v}\n\n"
                rows.append([k, status, _details])

        stage = None
        if model is not None:
            stage = getattr(model, "_pp_stage", None)
        if stage is not None:
            owned, skipped = _pp_report_key_owners(stage, model, self.skipped_pp_keys)
            for status, color, note, key_owners in (
                ("OWNED", "green", "checkpoint weights loaded on this pipeline stage.", owned),
                ("SKIPPED", "dim", "checkpoint weights owned by another pipeline stage.", skipped),
            ):
                if key_owners:
                    tips += f"\n- {_style(f'{status}:', color)}\t{note}"
                    for key, owner in update_key_name(key_owners).items():
                        rows.append([key, _style(status, color), f"PP rank {owner}"])

        # If nothing is wrong, return None
        if len(rows) == 0:
            return None

        headers = ["Key", "Status"]
        if term_w > 200:
            headers += ["Details"]
        else:
            headers += ["", ""]
        table = _make_table(rows, headers=headers)
        report = table + tips

        return report


def dot_natural_key(s: str):
    """
    Sort key for state-dict names: split on `"."` and sort digits numerically and strings alphabetically. It emits a
    tuple at each point to sort ints first and strings second to avoid int-string comparison failures.
    """
    parts = []
    for part in s.split("."):
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            # This will remove all trailing digit characters, as `rstrip` actually considers it as a set of chars
            text_part = part.rstrip("0123456789")
            trailing_digits = part[len(text_part) :]
            # Sort numeric suffixes numerically, so `shard_2` precedes `shard_11` for example
            if trailing_digits != "":
                parts.append((1, text_part, int(trailing_digits)))
            else:
                parts.append((1, text_part))
    return parts


def rename_source_key(
    source_key: str,
    weight_renamings: list[WeightRenaming],
    weight_converters: list[WeightConverter],
    base_model_prefix: str | None = None,
    meta_state_dict: dict | None = None,
    reverse: bool = False,
) -> tuple[str, str | None]:
    """
    Rename a checkpoint key by first applying all `WeightRenaming`s, then at most one `WeightConverter`.
    This means that the `WeightConverter` must specify the final name for the key, but some `WeightRenaming`s can act on other
    parts of that key beforehand. If `reverse` is True, i.e. when reverting all the `WeightTransform`s, the opposite is performed
    to be coherent and correctly respect the `scope_prefix` of all `WeightTransform`s: first try to match 1 `WeightConverter`, and only
    then try to apply all `WeightRenaming`s. Indeed, in reverse mode, all reverse transforms should be applied in the opposite order to
    be consistent.
    Note that we proceed in this way because there is no need for a Converter-then-Rename order because Converters act only on specific
    leaf patterns, so no subsequent Renamings should ever target their output.

    Args:
        source_key (`str`):
            The original checkpoint key to rename.
        weight_renamings (`list[WeightRenaming]`):
            Applied in order; every matching renaming fires (they may chain).
        weight_converters (`list[WeightConverter]`):
            Applied after all renamings; at most one may match. Subsequent converters are skipped.
        base_model_prefix (`str`, *optional*):
            Base-model prefix to add or strip when both `base_model_prefix` and `meta_state_dict` are given.
        meta_state_dict (`dict`, *optional*):
            Meta state dict used to decide whether `base_model_prefix` should be added or stripped.
        reverse (`bool`, *optional*):
            This specifies if we are reverting all the `WeightTransform`s (saving back original format).

    Returns:
        `tuple[str, str | None]`: The renamed key and the matched converter's source pattern
        (or `None` if no converter matched).
    """
    renamed_key = source_key
    converter_source_pattern = None
    # 1. If `reverse` is False: apply all renamings in turns (if multiple match, it's the responsibility of the mappings to make sure
    # they are coherent).
    # Else, first apply the `WeightConverter` if any
    first_iterable = weight_renamings if not reverse else weight_converters
    for transform in first_iterable:
        renamed_key, source_pattern = transform.rename_source_key(renamed_key)
        # Only break after match is we are using the `WeightConverter`s here, i.e. `reverse` is True
        if reverse and source_pattern is not None:
            converter_source_pattern = source_pattern
            break

    # 2. If `reverse` is False: apply renaming through weight conversions on the key if we have any WeightConverter (here we stop after
    # the first match, as we assume only 1 converter can match any source key).
    # Else, apply all the `WeightRenaming`s
    second_iterable = weight_converters if not reverse else weight_renamings
    for transform in second_iterable:
        renamed_key, source_pattern = transform.rename_source_key(renamed_key)
        # Only break after match is we are using the `WeightConverter`s here, i.e. `reverse` is False
        if not reverse and source_pattern is not None:
            converter_source_pattern = source_pattern
            break

    # 3. check if we need to add or remove base_model_prefix if necessary (only during loading, not saving)
    if base_model_prefix is not None and meta_state_dict is not None:
        if (
            renamed_key.startswith(base_model_prefix)
            and meta_state_dict.get(re.sub(f"^{base_model_prefix}.", "", renamed_key, count=1)) is not None
        ):
            renamed_key = re.sub(f"^{base_model_prefix}.", "", renamed_key, count=1)
        elif meta_state_dict.get(f"{base_model_prefix}.{renamed_key}") is not None:
            renamed_key = f"{base_model_prefix}.{renamed_key}"

    return renamed_key, converter_source_pattern


def _add_unmatched_checkpoint_key(
    key: str,
    model: Model,
    loading_info: LoadStateDictInfo,
) -> None:
    """Classify a checkpoint key that does not match the current model state dict.

    During pipeline-parallel loading, each stage receives keys for the entire model even though its local state dict
    contains only that stage's parameters. A key owned by another stage is therefore recorded as intentionally skipped
    instead of unexpected. Without the key, it would be considered unexpected.
    """
    stage = getattr(model, "_pp_stage", None)
    if stage is None:
        loading_info.unexpected_keys.add(key)
        return

    base_model = getattr(model, model.base_model_prefix)
    owner_rank = stage.find_rank_for_key(key, len(base_model.layers), model.base_model_prefix)
    owned_by_another_stage = owner_rank is not None and owner_rank != stage.pp_rank

    if owned_by_another_stage:
        loading_info.skipped_pp_keys.add(key)
    else:
        loading_info.unexpected_keys.add(key)


def convert_and_load_state_dict_in_model(
    model: PreTrainedModel,
    state_dict: dict[str, Any],
    load_config: LoadStateDictConfig,
    disk_offload_index: dict | None = None,
):
    base_model_prefix = model.base_model_prefix
    device_map = load_config.device_map or {"": "cpu"}
    dtype = load_config.dtype
    disk_offload_folder = load_config.disk_offload_folder
    offload_buffers = load_config.offload_buffers
    dtype_plan = load_config.dtype_plan or {}
    weight_mapping = load_config.weight_mapping or []
    meta_model_state_dict = model.state_dict()
    model_buffers = {k for k, _ in model.named_buffers()}

    # We start from all missing keys, and we will remove/add them from the proper containers as loading advances
    loading_info = LoadStateDictInfo(
        missing_keys=set(meta_model_state_dict.keys()),
        unexpected_keys=set(),
        mismatched_keys=set(),
        conversion_errors={},
        error_msgs=[],
        skipped_pp_keys=set(),
    )

    # We use threading by default, if not explicitly deactivated via env variable. If we have to offload,
    # we cannot use it either to control the memory as we are under memory constraints, so we need to be sequential.
    # When doing on-the-fly quantization, we also use sync loading to avoid worker threads loading full-precision
    # tensors to GPU faster than the main thread can quantize them, which would cause a large memory spike.
    if (
        #is_env_variable_true("HF_DEACTIVATE_ASYNC_LOAD") or
        "disk" in device_map.values()
    ):
        thread_pool = None
    else:
        thread_pool = ThreadPoolExecutor(max_workers=GLOBAL_WORKERS)

    renamings = [
        entry
        for entry in weight_mapping
        if isinstance(entry, WeightRenaming)
    ]
    converters = [
        entry
        for entry in weight_mapping
        if isinstance(entry, WeightConverter)
    ]
    param_name_to_load: dict[str, WeightRenaming | WeightConverter] = {}

    if dtype_plan != {}:
        dtype_policy_alt, dtype_policy_by_group_name, _ = build_glob_alternation(list(dtype_plan.keys()))

    pattern_to_converter = {
        k: converter
        for converter in converters
        for k in converter.source_patterns
    }
    print(f"Pattern to convert - {pattern_to_converter}")

    state_dict = sorted(
        state_dict.items(),
        key=lambda kv: dot_natural_key(kv[0])
    )
    for original_key, tensor in state_dict:
        # 1. Rename the key according to all renaming and weight conversion patterns.
        renamed_key, source_pattern = rename_source_key(
            original_key, renamings, converters, base_model_prefix, meta_model_state_dict
        )
        print("="*50)
        print(">>", renamed_key, original_key, source_pattern)
        print("="*50)
        if renamed_key not in meta_model_state_dict and original_key in meta_model_state_dict:
            # Key should probably not have been renamed but we might need the `prefix` to be added.
            renamed_key, source_pattern = rename_source_key(
                original_key, [], [], base_model_prefix=base_model_prefix, meta_state_dict=meta_model_state_dict
            )

        # 2. finally, collect the tensor into the proper converter
        if renamed_key in meta_model_state_dict:
            empty_param = meta_model_state_dict.get(renamed_key)
            # If we enter here, we have a WeightConverter operation to perform
            if source_pattern is not None:
                new_converter = deepcopy(pattern_to_converter[source_pattern])
                # each target key gets its own converter instance
                mapping = param_name_to_load.setdefault(renamed_key, new_converter)
            # Otherwise, only potential renaming
            else:
                mapping = param_name_to_load.setdefault(renamed_key, WeightRenaming(original_key, renamed_key))
                source_pattern = original_key

            # 3. Handle dtype casting

            _dtype = dtype
            if dtype_plan != {} and dtype_policy_alt.search(renamed_key):
                matched_dtype_pattern = dtype_policy_alt.search(renamed_key)
                if matched_dtype_pattern is not None:
                    _dtype = dtype_plan[dtype_policy_by_group_name[matched_dtype_pattern.lastgroup]]
            elif empty_param is not None and empty_param.dtype != _dtype:
                _dtype = empty_param.dtype  # usually correct when initializing

            # Per-expert sharding (EP) needs `tensor_idx` = the expert index so the
            # distributed op selects whole experts. The signal is a `MergeModulelist`
            # in the chain; it isn't always `operations[0]` (e.g. an FP8 quantizer
            # prepends a scale-decode op), so scan the whole chain rather than just the head.
            tensor_idx = (
                len(mapping.collected_tensors.get(source_pattern, []))
                if isinstance(mapping, WeightConverter)
                and any(isinstance(op, MergeModulelist) for op in mapping.operations)
                else None
            )

            # 4. Handle DTensor sharding or device_map placement
            param_device = get_device(device_map, renamed_key, valid_torch_device=True)
            sharding_op = None
            if is_dtensor(empty_param):
                sharding_op = DtensorShardOperation(empty_param)

            # Some parameters are so large (qwen4_exp ple_embedding is about ~95 GiB) that we cannot afford to perform the Operations
            # directly on the device, as it will completely blow up the memory during the ops memory spike. So defer to "cpu", then
            # accelerate will take care of putting back on correct device after loading
            # Note that we only do it with `device_map` but not with `tp_plan`, as tp will perform local sharding before, so memory
            # spike during conversion ops should be fine
            if sharding_op is None and isinstance(mapping, WeightConverter) and mapping.force_cpu:
                param_device = "cpu"

            future_or_tensor = spawn_materialize(
                thread_pool,
                tensor,
                param_device,
                _dtype,
                sharding_op=sharding_op,
                tensor_idx=tensor_idx,
            )

            mapping.add_tensor(renamed_key, original_key, source_pattern, future_or_tensor)
        elif source_pattern is not None:  # add all target keys as unexpected
            mapping = pattern_to_converter[source_pattern]
            for k in mapping.target_patterns:
                _add_unmatched_checkpoint_key(
                    renamed_key.replace(mapping.target_patterns[0], k),
                    model,
                    loading_info,
                )
        else:
            _add_unmatched_checkpoint_key(renamed_key, model, loading_info)
    try:
        for first_param_name, mapping in tqdm(param_name_to_load.items(), desc="Loading weights"):
            try:
                realized_value = mapping.convert(
                    first_param_name,
                    model=model,
                    config=model.config,
                    loading_info=loading_info,
                )
                for target_name, param in realized_value.items():
                    param = param[0] if isinstance(param, list) else param
                    param_device = get_device(device_map, target_name)
                    # Exception for params that are so huge that they need conversions on cpu no matter what - we put them back on
                    # the correct device now (if the device_map managed to make them fit on any device, as usually they will stay on
                    # cpu and this is a no-op)
                    if getattr(mapping, "force_cpu", False) and param_device != "disk":
                        param = param.to(param_device)
                    # Offloading support
                    if param_device == "disk" and (target_name not in model_buffers or offload_buffers):
                        disk_offload_index = offload_and_maybe_resave_param(
                            target_name, param, loading_info, disk_offload_folder, disk_offload_index, mapping
                        )
                    else:
                        set_param_for_module(
                            model,
                            target_name,
                            param,
                            loading_info,
                            hf_quantizer,
                        )

                # Cleanup all the tensors that were gathered before next iteration
                del realized_value

            except SkipParameters:
                continue

    # Close the pool, independently of whether the code was interrupted or finished successfully
    finally:
        if thread_pool is not None:
            # `cancel_futures=True` in case the program was interrupted, to avoid wasting time on exit
            thread_pool.shutdown(wait=False, cancel_futures=True)

    # Keep the current weight conversion mapping for later saving (in case it was coming directly from the user), but
    # only if it was used, i.e. it matched any weight from the checkpoints
    model_specific_conversions = [conversion for conversion in weight_mapping if conversion.was_used()]
    model._weight_conversions = model_specific_conversions

    return loading_info, disk_offload_index


class Model(nn.Module):
    
    # General model properties
    config_class = ModelConfig
    base_model_prefix: str = ""

    # Device-map related properties
    _no_split_modules: set[str] | list[str] | None = None
    _skip_keys_device_placement: set[str] | list[str] | None = None

    # Specific dtype upcasting
    # `_keep_in_fp32_modules` will upcast to fp32 only if the requested dtype is fp16
    # `_keep_in_fp32_modules_strict` will upcast to fp32 independently if the requested dtype is fp16 or bf16
    _keep_in_fp32_modules: set[str] | list[str] | None = None
    _keep_in_fp32_modules_strict: set[str] | list[str] | None = None

     # Loading-specific properties
    # A dictionary `{"target": "source"}` of checkpoint keys that are potentially tied to one another
    _tied_weights_keys: dict[str, str] = None
    # A list of `re` patterns describing keys to ignore if they are missing from checkpoints to avoid warnings
    _keys_to_ignore_on_load_missing: set[str] | list[str] | None = None
    # A list of `re` patterns describing keys to ignore if they are unexpected in the checkpoints to avoid warnings
    _keys_to_ignore_on_load_unexpected: set[str] | list[str] | None = None
    # A list of keys to ignore when saving the model
    _keys_to_ignore_on_save: set[str] | list[str] | None = None

    # Advanced functionalities support
    supports_gradient_checkpointing: bool = False

    # Distributed Mixin
    _device_mesh = None
    _tp_plan: dict[str, str] | None = None
    _ep_plan: dict[str, str] | None = None
    _tp_size = None
    _fsdp_size = None
    _pp_plan: dict[str, tuple[str, str]] | None = None
    _fsdp_plan: dict[str, str] | None = None

    @property
    def base_model(self) -> nn.Module:
        """
        `torch.nn.Module`: The main body of the model.
        """
        return getattr(self, self.base_model_prefix, self)

    def __init__(self, config: ModelConfig):
        super().__init__()

        if not isinstance(config, self.config_class):
            raise TypeError(
                f"config must be an instance of "
                f"{self.config_class.__name__}"
            )

        self.config = config

    def _backward_compatibility_gradient_checkpointing(self):
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()
            # Remove the attribute now that is has been consumed, so it's no saved in the config.
            delattr(self.config, "gradient_checkpointing")

    def init_parallel_plans(self) -> None:
        """Copy class-level plans onto the instance and merge config/children contributions."""
        model_cls = type(self)
        self._tp_plan = dict(getattr(model_cls, "_tp_plan", None) or {})
        self._ep_plan = dict(getattr(model_cls, "_ep_plan", None) or {})
        self._pp_plan = dict(getattr(model_cls, "_pp_plan", None) or {})
        self._fsdp_plan = dict(getattr(model_cls, "_fsdp_plan", None) or {})

        if self.base_model is self:
            self._pp_plan.update(self.config.base_model_pp_plan or {})
            self._tp_plan.update(self.config.base_model_tp_plan or {})
            self._ep_plan.update(self.config.base_model_ep_plan or {})
            self._fsdp_plan.update(self.config.base_model_fsdp_plan or {})

        for name, module in self.named_children():
            if plan := getattr(module, "_ep_plan", None):
                self._ep_plan.update({f"{name}.{k}": v for k, v in plan.copy().items()})
            if plan := getattr(module, "_tp_plan", None):
                self._tp_plan.update({f"{name}.{k}": v for k, v in plan.copy().items()})
            if plan := getattr(module, "_pp_plan", None):
                self._pp_plan.update({f"{name}.{k}": v for k, v in plan.copy().items()})
            if plan := getattr(module, "_fsdp_plan", None):
                self._fsdp_plan.update({f"{name}.{k}": v for k, v in plan.copy().items()})

    def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict:
        if all_submodels:
            expanded_tied_weights = {}
            for prefix, submodule in self.named_modules(remove_duplicate=False):
                if isinstance(submodule, Model):
                    # Will dynamically check the config if it has changed
                    submodel_tied_weights = submodule.get_expanded_tied_weights_keys(all_submodels=False)
                    if prefix != "":
                        submodel_tied_weights = {
                            f"{prefix}.{k}": f"{prefix}.{v}" for k, v in submodel_tied_weights.items()
                        }
                    expanded_tied_weights.update(submodel_tied_weights)
            return expanded_tied_weights

        tied_mapping = self._tied_weights_keys
        # If the config does not specify any tying, return empty dict
        # NOTE: not all modules have `tie_word_embeddings` attr, for example vision-only
        # modules do not have any word embeddings!
        tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)
        if not tie_word_embeddings:
            return {}
        # If None, return empty dict
        elif tied_mapping is None:
            return {}
        # Short-cut for the most common cases: if the tied weights mapping only contains already expanded params,
        # return it directly (the regex matches names containing only letters, numbers, dots, and underscores to make
        # sure it does not contain a regex pattern, and finishing by "bias" or "weight" to make sure it's not a module)
        common_case_regex = re.compile(r"^[A-Za-z0-9_\.]+(weight)|(bias)$")
        if all(common_case_regex.match(k) for k in tied_mapping.keys() | tied_mapping.values()):
            return tied_mapping.copy()

        # We need to expand the regex patterns or the modules into proper parameters
        expanded_tied_weights = {}
        all_param_names = {k for k, _ in self.named_parameters(remove_duplicate=False)} | {
            k for k, _ in self.named_buffers(remove_duplicate=False)
        }
        for target_name, source_name in tied_mapping.items():
            target_name = "^" + target_name
            source_name = "^" + source_name

            source_params = sorted(filter(lambda x: re.search(source_name, x), all_param_names))
            target_params = sorted(filter(lambda x: re.search(target_name, x), all_param_names))
            if (
                not len(source_params) > 0
                or not len(target_params) > 0
                or len(target_params) % len(source_params) != 0
            ):
                raise ValueError(
                    f"There is an issue with your definition of `tie_weights_keys` for {source_name}:{target_name}. "
                    f"We found {source_params} to tie into {target_params}"
                )
            # we cycle source as it should be dispatch in many target if regex
            for target_n, source_n in zip(target_params, cycle(source_params)):
                # If the source is already registered as a target, use the original corresponding source. This should never
                # happen in general, but some models such as `d_fine` have complicated regex patterns, so it end up being
                # the case for simplicity of the regexes. Fix it silently here
                if source_n in expanded_tied_weights.keys():
                    # Use original source instead of having keys both as source and targets
                    expanded_tied_weights[target_n] = expanded_tied_weights[source_n]
                # Usual case, everything is already correct
                else:
                    expanded_tied_weights[target_n] = source_n

        return expanded_tied_weights

    @torch.no_grad()
    def _init_weights(self, module):
        """
        Initialize the weights. This is quite general on purpose, in the spirit of what we usually do. For more complex
        initialization scheme, it should be overridden by the derived `PreTrainedModel` class. In case a model adds an explicit
        `nn.Parameter`, this method should also be overridden in order to initialize it correctly.
        """
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range or 0.02
        elif hasattr(self.config, "init_std"):
            std = self.config.init_std
        elif hasattr(self.config, "initializer_factor"):
            std = self.config.initializer_factor
        else:
            # 0.02 is the standard default value across the library
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            if getattr(module, "weight", None) is not None:
                normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    xavier_uniform_(param)
                elif "bias" in name:
                    constant_(param, 0.0)
        elif isinstance(module, nn.Embedding):
            normal_(module.weight, mean=0.0, std=std)
            # Here we need the check explicitly, as we slice the weight in the `zeros_` call, so it looses the flag
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.MultiheadAttention):
            # This uses torch's original init
            module._reset_parameters()
        # We cannot use `isinstance` on the RMSNorms or LayerNorms, as they usually are custom modules which change names
        # between modelings (because they are prefixed with the model name)
        elif (
            isinstance(module, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            or "LayerNorm" in module.__class__.__name__
            or "RMSNorm" in module.__class__.__name__
        ):
            # Norms can exist without weights (in which case they are None from torch primitives)
            if getattr(module, "weight", None) is not None:
                ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                zeros_(module.bias)
            # And the potential buffers for the BatchNorms
            if getattr(module, "running_mean", None) is not None:
                zeros_(module.running_mean)
                ones_(module.running_var)
                zeros_(module.num_batches_tracked)
        # This matches all the usual RotaryEmbeddings modules
        elif "RotaryEmbedding" in module.__class__.__name__ and hasattr(module, "original_inv_freq"):
            # Default and vision axial rope are defined in modeling files, only one can be defined at a time!
            rope_init_fn_with_self = {
                "axial": getattr(module, "compute_axial_rope_parameters", None),
                "default": getattr(module, "compute_default_rope_parameters", None),
                **ROPE_INIT_FUNCTIONS,
            }
            rope_fn = rope_init_fn_with_self[module.rope_type]
            buffer_value, _ = rope_fn(module.config)
            copy_(module.inv_freq, buffer_value)
            copy_(module.original_inv_freq, buffer_value)

    def _initialize_weights(self, module, is_custom_code: bool = False):
        """
        Initialize the weights if they are not already initialized.
        """
        if getattr(module, "_is_hf_initialized", False):
            return

        # This check is for remote code that does NOT use either `torch.init` or `transformers.initialization` in `_init_weights`
        # which allow to check the flag directly on param. As they don't and write the params in-place, params would be reinitialized
        # otherwise
        if (
            is_custom_code
            and all(getattr(param, "_is_hf_initialized", False) for param in module.parameters(recurse=False))
            and all(
                getattr(buffer, "_is_hf_initialized", False)
                for buffer in module.buffers(recurse=False)
                if buffer is not None
            )
        ):
            module._is_hf_initialized = True
            return

        self._init_weights(module)
        module._is_hf_initialized = True

    @torch.no_grad()
    @guard_torch_init_functions()
    def initialize_weights(self):
        """
        This is equivalent to calling `self.apply(self._initialize_weights)`, but correctly handles composite models.
        This function dynamically dispatches the correct `init_weights` function to the modules as we advance in the
        module graph along the recursion. It can handle an arbitrary number of sub-models. Without it, every composite
        model would have to recurse a second time on all sub-models explicitly in the outer-most `_init_weights`, which
        is extremely error prone and inefficient.
        """
        if not hasattr(torch.nn.Module, "smart_apply"):
            # This function is equivalent to `torch.nn.Module.apply`, except that it dynamically adjust the function
            # to apply as we go down the graph
            def smart_apply(module: nn.Module, fn: t.Callable[[nn.Module, bool], None], is_custom_code: bool):
                for child in module.children():
                    # We found a sub-model: recursively dispatch its own init function now!
                    if isinstance(child, Model):
                        smart_apply(child, child._initialize_weights, is_custom_code)
                    else:
                        smart_apply(child, fn, is_custom_code)
                fn(module, is_custom_code)
                return module

            setattr(torch.nn.Module, "smart_apply", smart_apply)

        # Let the magic happen with this simple call
        smart_apply_fn = getattr(self, "smart_apply")
        # `getattr(self, ...)` returns a bound method, so `self` is already provided as the receiver.
        smart_apply_fn(self._initialize_weights, self.is_custom_code())

    def tie_weights(self, missing_keys: set[str] | None = None, recompute_mapping: bool = True):
        """
        Tie the model weights. If `recompute_mapping=False` (default when called internally), it will rely on the
        `model.all_tied_weights_keys` attribute, containing the `{target: source}` mapping for the tied params.
        If `recompute_mapping=True`, it will re-check all internal submodels and their config to determine the params
        that need to be tied. This is the default when `model.tie_weights()` is called on its own, outside of
        `__init__`, and `from_pretrained`, in case the config values were changed somewhere.

        Note that during `from_pretrained`, tying is *symmetric*: if the mapping says "tie target -> source" but
        `source` is missing in the checkpoint while `target` exists, we *swap* source and target so we can still
        tie everything to the parameter that actually exists.
        """
        # In this case, the keys stored in `all_tied_weights_keys` are already correct
        if not recompute_mapping:
            tied_keys = self.all_tied_weights_keys
        else:
            tied_keys = self.get_expanded_tied_weights_keys(all_submodels=True)

        tied_keys = list(tied_keys.items())
        for i, (target_param_name, source_param_name) in enumerate(tied_keys):
            # This is `from_pretrained` -> let's check symmetrically in case the source key is not present
            if missing_keys is not None:
                remove_from_missing = True
                source_is_there = source_param_name not in missing_keys
                target_is_there = target_param_name not in missing_keys
                # Both are already present -> it means the config is wrong and do not reflect the actual
                # checkpoint -> let's raise a warning and NOT tie them
                if source_is_there and target_is_there:
                    source_param = self.get_parameter(source_param_name)
                    target_param = self.get_parameter(target_param_name)

                    # Skip check if both are disk offloaded. Tied tensors always
                    # share the same offload device as per `infer_auto_device_map`
                    if source_param.device.type == "meta" and target_param.device.type == "meta":
                        continue

                    # If both are present, check if the weights are exactly similar, and only tie in this case
                    # This check is important, as torch `.bin` checkpoints always contain both keys, referencing the same storage
                    if not torch.equal(source_param, target_param):
                        logger.warning(
                            f"The tied weights mapping and config for this model specifies to tie {source_param_name} to "
                            f"{target_param_name}, but both are present in the checkpoints with different values, so we will NOT "
                            "tie them. You should update the config with `tie_word_embeddings=False` to silence this warning."
                        )
                        # Remove from internal attribute to correctly reflect actual tied weights
                        self.all_tied_weights_keys.pop(target_param_name)
                        # Skip to next iteration
                        continue
                # We're missing the source but we have the target -> we swap them, tying the parameter that exists
                elif not source_is_there and target_is_there:
                    target_param_name, source_param_name = source_param_name, target_param_name
                # Both are missing -> check other keys in case more than 2 keys are tied to the same weight
                elif not source_is_there and not target_is_there:
                    for target_backup, source_backup in tied_keys[i + 1 :]:
                        # In case of more than 2 keys tied to the same weight, they are guaranteed to all have
                        # the same source thanks to `get_expanded_tied_weights_keys` so this check is enough
                        if source_backup == source_param_name:
                            target_backup_is_there = target_backup not in missing_keys
                            # If the target is present, we found the correct weight to tie into (we know the source is missing)
                            # Note here that we do not tie the missing source right now as well, as it will be done anyway when
                            # the pair (target_backup, source_backup) becomes the main pair (target_param_name, source_param_name)
                            if target_backup_is_there:
                                source_param_name = target_backup
                                break
                    # If we did not break from the loop, it was impossible to find a source key -> let's raise
                    else:
                        # TODO Cyril: here ideally we want to raise instead of warning, but will break our CI as we have
                        # tests loading model from empty dicts to perform init checks - since we don't raise, add a flag
                        # to NOT remove from missing keys as it's actually still missing
                        remove_from_missing = False
                        logger.warning(
                            f"This checkpoint seem corrupted. The tied weights mapping for this model specifies to tie "
                            f"{source_param_name} to {target_param_name}, but both are absent from the checkpoint, "
                            "and we could not find another related tied weight for those keys"
                        )

            # Perform the actual tying
            source_param = self.get_parameter_or_buffer(source_param_name)
            if "." in target_param_name:
                parent_name, name = target_param_name.rsplit(".", 1)
                parent = self.get_submodule(parent_name)
            else:
                name = target_param_name
                parent = self
            # Tie the weights
            setattr(parent, name, source_param)
            self._adjust_bias(parent, source_param)
            # Remove from missing if necessary
            if missing_keys is not None and remove_from_missing:
                missing_keys.discard(target_param_name)

    def _adjust_bias(self, output_embeddings, input_embeddings):
        if getattr(output_embeddings, "bias", None) is not None and hasattr(output_embeddings, "weight"):
            weight_shape = output_embeddings.weight.shape
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (0, weight_shape[0] - output_embeddings.bias.shape[0]),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    @classmethod
    def is_remote_code(cls) -> bool:
        """Return whether the current model is custom code, i.e. code loaded from the hub, or class that we just registered
        via `register_for_auto_class`."""
        return cls._auto_class is not None

    @classmethod
    def is_custom_code(cls) -> bool:
        """Return whether the current model is custom code, i.e. either code loaded from the hub, or defined in any user-specific
        module/session."""
        return cls.is_remote_code() or not cls.__module__.startswith("transformers.")

    def get_parameter_or_buffer(self, target: str):
        """
        Return the parameter or buffer given by `target` if it exists, otherwise throw an error. This combines
        `get_parameter()` and `get_buffer()` in a single handy function. If the target is an `_extra_state` attribute,
        it will return the extra state provided by the module. Note that it only work if `target` is a leaf of the model.
        """
        try:
            return self.get_parameter(target)
        except AttributeError:
            pass
        try:
            return self.get_buffer(target)
        except AttributeError:
            pass
        module, param_name = get_module_from_name(self, target)
        if (
            param_name == "_extra_state"
            and getattr(module.__class__, "get_extra_state", torch.nn.Module.get_extra_state)
            is not torch.nn.Module.get_extra_state
        ):
            return module.get_extra_state()

        raise AttributeError(f"`{target}` is neither a parameter, buffer, nor extra state.")

    def init_weights(self):
        """
        Initialize and tie the weights if needed. If using a custom `PreTrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """
        # If we are initializing on meta device, there is no point in trying to run inits
        if get_torch_context_manager_or_global_device() != torch.device("meta"):
            # Initialize weights
            self.initialize_weights()
        # Tie weights needs to be called here, but it can use the pre-computed `all_tied_weights_keys`
        self.tie_weights(recompute_mapping=False)

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        It is also used to obtain all correct static properties (parallelism plans, tied_weights_keys, _keep_in_fp32_modules, etc)
        correctly in the case of composite models (that is, the top level model should know about those properties from its children).
        """
        # Attach the different parallel plans and tied weight keys to the top-most model, so that everything is
        # easily available.
        self.init_parallel_plans()
        # Current submodel should register its tied weights
        self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=False)
        # Current submodel should register its `_keep_in_fp32_modules`
        self._keep_in_fp32_modules = set(self._keep_in_fp32_modules or [])
        self._keep_in_fp32_modules_strict = set(self._keep_in_fp32_modules_strict or [])
        # Current submodel must register its `_no_split_modules`/`_skip_keys_device_placement` as well for device_map
        self._no_split_modules = set(self._no_split_modules or [])
        self._skip_keys_device_placement = set(self._skip_keys_device_placement or [])
        # Current submodel must register the `_keys_to_ignore_on_load_unexpected/missing`
        self._keys_to_ignore_on_load_unexpected = set(self._keys_to_ignore_on_load_unexpected or [])
        self._keys_to_ignore_on_load_missing = set(self._keys_to_ignore_on_load_missing or [])
        self._keys_to_ignore_on_save = set(self._keys_to_ignore_on_save or [])

        # Iterate over children only: as the final model is created, this is enough to gather the properties from all submodels.
        # This works because the way the `__init__` and `post_init` are called on all submodules is depth-first in the graph
        for name, module in self.named_children():
            # Always attach the keys of the children (if the children's config says to NOT tie, then it's empty)
            if tied_keys := getattr(module, "all_tied_weights_keys", None):
                self.all_tied_weights_keys.update({f"{name}.{k}": f"{name}.{v}" for k, v in tied_keys.copy().items()})
            # Record keep_in_fp_32 modules from the children as well
            if keep_fp32 := getattr(module, "_keep_in_fp32_modules", None):
                self._keep_in_fp32_modules.update(keep_fp32)
            if keep_fp32_strict := getattr(module, "_keep_in_fp32_modules_strict", None):
                self._keep_in_fp32_modules_strict.update(keep_fp32_strict)
            # Record `_no_split_modules`/`_skip_keys_device_placement` from the children
            if no_split := getattr(module, "_no_split_modules", None):
                self._no_split_modules.update(no_split)
            if skip_keys := getattr(module, "_skip_keys_device_placement", None):
                self._skip_keys_device_placement.update(skip_keys)
            # Record `_keys_to_ignore_on_load_unexpected/missing` from the children - note that we do not add the name (prefix)
            # of the current module to the child's key, as this is matched by regex anyway and adding prefix could break the regex
            if ignore_unexpected := getattr(module, "_keys_to_ignore_on_load_unexpected", None):
                self._keys_to_ignore_on_load_unexpected.update(ignore_unexpected)
            if ignore_missing := getattr(module, "_keys_to_ignore_on_load_missing", None):
                self._keys_to_ignore_on_load_missing.update(ignore_missing)
            # This one is matched exactly, not by regex, so we need to add the prefix
            if ignore_save := getattr(module, "_keys_to_ignore_on_save", None):
                self._keys_to_ignore_on_save.update({f"{name}.{k}" for k in ignore_save})

        # Maybe initialize the weights and tie the keys
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()


    def _get_dtype_plan(self, dtype: torch.dtype) -> dict:
        """Create the dtype_plan describing modules/parameters that should use the `keep_in_fp32` flag."""
        dtype_plan = {}

        # The _keep_in_fp32_modules flag is only used to avoid bf16 -> fp16 casting precision issues. It was introduced
        # in case of force loading a model that should stay in bf16 in fp16
        # See https://github.com/huggingface/transformers/issues/20287 for details.
        if (
            self._keep_in_fp32_modules is not None and
            dtype == torch.float16
        ):
            dtype_plan.update(
                dict.fromkeys(
                    self._keep_in_fp32_modules,
                    torch.float32
                )
            )

        # The _keep_in_fp32_modules_strict was introduced to always force upcast to fp32, for both fp16 and bf16
        if (self._keep_in_fp32_modules_strict is not None and
            dtype in (torch.float16, torch.bfloat16)
        ):
            dtype_plan.update(
                dict.fromkeys(
                    self._keep_in_fp32_modules_strict,
                    torch.float32
                )
            )

        return dtype_plan
    
    @classmethod
    def get_init_context(
        cls,
        dtype: torch.dtype,
        allow_all_kernels: bool | None
    ):
        # Need to instantiate with correct dtype
        init_contexts = [
            local_torch_dtype(dtype, cls.__name__),
            #init.no_tie_weights(),
            #apply_patches()
        ]
        # Needed as we cannot forward the `allow_all_kernels` arg in the model's __init__
        if allow_all_kernels:
            init_contexts.append(allow_all_hub_kernels())
        else:
            # meta_device_safe_creation_ops patches torch.linspace to default to CPU
            # so that custom models calling .item() during __init__ (e.g. drop-path
            # schedules) don't crash on meta tensors.
            init_contexts.extend([
                torch.device("meta"),
                #init.meta_device_safe_creation_ops()
            ])

        return init_contexts
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        ignore_mismatched_sizes: bool = False,
        use_safetensors: bool | None = None,
        weights_only: bool = True,
        disable_mmap: bool | None = None,
        **kwargs,
    ):
        dtype = kwargs.pop("dtype", None)
        torch_dtype = kwargs.pop("torch_dtype", None)  # kept for BC
        device_map = kwargs.pop("device_map", None)
        
        adapter_kwargs = (kwargs.pop("adapter_kwargs", {}) or {}).copy()
        adapter_name = kwargs.pop("adapter_name", "default")

        device_mesh = kwargs.pop("device_mesh", None)
        
        allow_all_kernels = kwargs.pop("allow_all_kernels", False)
        use_kernels = kwargs.pop("use_kernels", False)
        kernel_config = kwargs.pop("kernel_config", None)
        
        # For BC on torch_dtype argument
        if torch_dtype is not None:
            dtype = dtype if dtype is not None else torch_dtype
        if dtype is None:
            dtype = "auto"
            
        model_kwargs = {}

        device_map = check_and_set_device_map(device_map)  # warn, error and fix the device map
        
        # Get Config
        config = (
            cls.config_class
            .from_pretrained(
                pretrained_model_name_or_path
            )
        )

        # Get tensor file
        files = _resolve_checkpoint_files(
            pretrained_model_name_or_path
        )

        config, dtype = _get_dtype(
            dtype, files, config, weights_only
        )
        config.name_or_path = pretrained_model_name_or_path

        model_init_context = cls.get_init_context(
            dtype, allow_all_kernels
        )
        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        with ContextManagers(model_init_context):
            model = cls(config, *model_args, **model_kwargs)
            #patch_output_recorders(model)
            
        # # Create the dtype_plan to potentially use the `keep_in_fp32` flags (this needs to be called on the already
        # # instantiated model, as the flags can be modified by instances sometimes)
        # dtype_plan = model._get_dtype_plan(dtype)

        # # Obtain the weight conversion mapping for this model if any are registered and apply to all submodels recursively
        # weight_conversions = get_model_conversion_mapping(
        #     model
        # )
        # print(weight_conversions)

        # # Prepare the full device map
        # if device_map is not None:
        #     device_map = _get_device_map(model, device_map, max_memory, hf_quantizer)

        # # Finalize model weight initialization
        # offload_folder = kwargs.pop("offload_folder", None)
        # offload_buffers = kwargs.pop("offload_buffers", False)
        
        # load_config = LoadStateDictConfig(
        #     pretrained_model_name_or_path=pretrained_model_name_or_path,
        #     ignore_mismatched_sizes=ignore_mismatched_sizes,
        #     device_map=device_map,
        #     disk_offload_folder=offload_folder,
        #     offload_buffers=offload_buffers,
        #     dtype=dtype,
        #     dtype_plan=dtype_plan,
        #     device_mesh=device_mesh,
        #     weights_only=weights_only,
        #     weight_mapping=weight_conversions,
        #     use_safetensors=use_safetensors,
        #     disable_mmap=disable_mmap,
        # )
        # loading_info, disk_offload_index = cls._load_pretrained_model(
        #     model, files, load_config
        # )
        # loading_info = cls._finalize_model_loading(model, load_config, loading_info)
        # model.eval()  # Set model in evaluation mode to deactivate Dropout modules by default
        
        # return model
        
        state_dict = load_state_dict(files)

        
        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=False,
        )

        if missing:
            raise RuntimeError(
                f"Missing model weights: {missing}"
            )

        if unexpected:
            raise RuntimeError(
                f"Unexpected model weights: {unexpected}"
            )
        model.eval()
        return model

    @staticmethod
    def _load_pretrained_model(
        model: "Model",
        files: list[Path] | None,
        load_config: LoadStateDictConfig,
        expected_keys: list[str] | None = None,
    ) -> tuple[LoadStateDictInfo, dict]:
        """Perform the actual loading of some checkpoints into a `model`, by reading them from disk and dispatching them accordingly."""

        # Model's definition arriving here is final (TP hooks added, quantized layers replaces)
        expected_keys = list(model.state_dict().keys()) if expected_keys is None else expected_keys

        # This offload index if for params explicitly on the "disk" in the device_map
        disk_offload_index = None
        # Prepare parameters offloading if needed
        if load_config.device_map is not None and "disk" in load_config.device_map.values():
            raise NotImplementedError

        error_msgs = []

        all_pointer = set()
        if (
            files is not None and
            str(files[0]).endswith(".safetensors")
        ):
            from safetensors import safe_open

            merged_state_dict = {}
            for file in files:
                # if load_config.disable_mmap or _is_on_hf_mount(file):
                #     with open(file, "rb") as _fh:
                #         merged_state_dict.update(_safe_load_bytes(_fh.read()))
                #     continue
                is_mps = load_config.device_map is not None and any(
                    (d.type if isinstance(d, torch.device) else d) == "mps"
                    for d in load_config.device_map.values()
                )
                # Use pread on MPS (mmap incompatible) and Windows (mmap reserves
                # copy-on-write commit charge for the entire file, exhausting memory
                # for large multi-shard checkpoints).
                if is_mps:
                    backend, device = "pread", "mps"
                elif sys.platform == "win32":
                    backend, device = "pread", "cpu"
                else:
                    backend, device = "mmap", "cpu"
                file_pointer = safe_open(file, framework="pt", device=device, backend=backend)
                all_pointer.add(file_pointer)
                for k in file_pointer.keys():
                    merged_state_dict[k] = file_pointer.get_slice(k)  # don't materialize yet
        # Checkpoints are .bin
        elif files is not None:
            raise NotImplementedError(".bin pytorch")
            merged_state_dict = {}
            for ckpt_file in checkpoint_files:
                merged_state_dict.update(load_state_dict(ckpt_file, disable_mmap=load_config.disable_mmap))
        else:
            raise ValueError("Neither a state dict nor checkpoint files were found.")

        loading_info, disk_offload_index = convert_and_load_state_dict_in_model(
            model=model,
            state_dict=merged_state_dict,
            load_config=load_config,
            disk_offload_index=disk_offload_index,
        )

        # finally close all opened file pointers
        for k in all_pointer:
            k.__exit__(None, None, None)

        return loading_info, disk_offload_index

    @staticmethod
    def _finalize_model_loading(
        model,
        load_config: LoadStateDictConfig,
        loading_info: LoadStateDictInfo
    ) -> LoadStateDictInfo:
        """Perform all post processing operations after having loaded some checkpoints into a model, such as moving
        missing keys from meta device to their expected device, reinitializing missing weights according to proper
        distributions, tying the weights and logging the loading report."""
        try:
            # Marks tied weights as `_is_hf_initialized` to avoid initializing them (it's very important for efficiency)
            model.mark_tied_weights_as_initialized(loading_info)

            # Move missing (and potentially mismatched) keys and non-persistent buffers back to their expected device from
            # meta device (because they were not moved when loading the weights as they were not in the loaded state dict)
            model._move_missing_keys_from_meta_to_device(
                loading_info.missing_and_mismatched(),
                load_config.device_map,
                load_config.device_mesh,
            )

            # Correctly initialize the missing (and potentially mismatched) keys (all parameters without the `_is_hf_initialized` flag)
            model._initialize_missing_keys(load_config.is_quantized)

            # Tie the weights
            model.tie_weights(missing_keys=loading_info.missing_keys, recompute_mapping=False)

            # Adjust missing and unexpected keys
            model._adjust_missing_and_unexpected_keys(loading_info)
        finally:
            log_state_dict_report(
                model=model,
                pretrained_model_name_or_path=load_config.pretrained_model_name_or_path,
                ignore_mismatched_sizes=load_config.ignore_mismatched_sizes,
                loading_info=loading_info,
                logger=logger,
            )

        return loading_info

    def _move_missing_keys_from_meta_to_device(
        self,
        missing_keys: list[str],
        device_map: dict | None,
        device_mesh: "DeviceMeshLike | None",
        hf_quantizer: HfQuantizer | None = None,
    ) -> None:
        """Move missing params/buffers off meta to their target device.

        Loaded weights are handled earlier in `convert_and_load_state_dict_in_model`
        via `DtensorShardOperation` and `set_param_for_module`. This only
        materializes keys that were not loaded (or mismatched) so
        `_initialize_missing_keys` can run proper init on them.
        """
        is_quantized = hf_quantizer is not None

        # In this case we need to move everything back
        # if is_fsdp_enabled() and not is_local_dist_rank_0() and not is_quantized:
        #     for key, param in self.named_parameters():
        #         value = torch.zeros_like(param, device="cpu")
        #         _load_parameter_into_model(self, key, value)
        #     for key, buffer in self.named_buffers():
        #         value = torch.zeros_like(buffer, device="cpu")
        #         _load_parameter_into_model(self, key, value)
        #     return

        # The tied weight keys are in the "missing" usually, but they should not be moved (they will be tied anyway)
        # This is especially important because if they are moved, they will lose the `_is_hf_initialized` flag, and they
        # will be re-initialized for nothing (which can be quite long)
        for key in missing_keys - self.all_tied_weights_keys.keys():
            param = self.get_parameter_or_buffer(key)
            param_device = get_device(device_map, key, valid_torch_device=True)
            value = torch.empty_like(param, device=param_device)
            # For TP, we may need to shard the param
            if is_dtensor(param):
                local = torch.empty(param._local_tensor.shape, dtype=param.dtype, device=param_device)
                value = torch.nn.Parameter(
                    _dtensor_from_local_like(local, param),
                    requires_grad=param.requires_grad,
                )
            _load_parameter_into_model(self, key, value)
        # We need to move back non-persistent buffers as well, as they are not part of loaded weights anyway
        for key, buffer in self.named_non_persistent_buffers():
            buffer_device = get_device(device_map, key, valid_torch_device=True)
            value = torch.empty_like(buffer, device=buffer_device)
            _load_parameter_into_model(self, key, value)

    def _initialize_missing_keys(self, is_quantized: bool) -> None:
        """
        Initialize the missing keys (keys that are part of the model parameters, but were NOT found in the loaded state dicts), according to
        `_initialize_weights`. Indeed, since the corresponding weights are missing from the state dict, they will not be replaced and need to
        be initialized correctly (i.e. weight initialization distribution).

        Also marks non-missing params/buffers with `_is_hf_initialized` and propagates this flag to modules,
        so that `_initialize_weights` can skip fully-initialized modules entirely.
        """
        if is_fsdp_enabled() and not is_local_dist_rank_0():
            # Handle FSDP edge case when using cpu ram efficient loading to ensure it is marked as initialized
            # since it will get its weights broadcasted from rank0
            # We actually need to do that only because we want to re-initialize non-persistent buffers with correct values.
            # Everything else in the state_dict will be gathered from rank0, so we don't need re-initialization.
            # We could simply early return after buffer inits if we had a way to init only the non-persistent buffers
            for key in self.state_dict():
                try:
                    param_or_buffer = self.get_parameter_or_buffer(key)
                    param_or_buffer._is_hf_initialized = True
                except AttributeError:
                    pass  # may happen when handling pre-quantized weights
            self._is_hf_initialized = True

        self.initialize_weights()

    def _adjust_missing_and_unexpected_keys(self, loading_info: LoadStateDictInfo) -> None:
        """Adjust the `missing_keys` and `unexpected_keys` based on current model's exception rules, to avoid
        raising unneeded warnings/errors. This is performed in-place.
        """
        # Old checkpoints may have keys for rotary_emb.inv_freq for each layer, however we moved this buffer to the main model
        # (so the buffer name has changed). Remove them in such a case. This is another exception that was not added to
        # `_keys_to_ignore_on_load_unexpected` as it touches many models -> we add it manually to the existing patterns
        has_inv_freq_buffers = any(buffer.endswith("rotary_emb.inv_freq") for buffer, _ in self.named_buffers())
        additional_unexpected_patterns = {r"rotary_emb\.inv_freq"} if has_inv_freq_buffers else set()
        # Same idea for `position_ids`: used to be a persistent buffer, now `persistent=False` in most models.
        has_position_ids_buffers = any(buffer.endswith("position_ids") for buffer, _ in self.named_buffers())
        if has_position_ids_buffers:
            additional_unexpected_patterns.add(r"(^|\.)position_ids$")

        missing_patterns = self._keys_to_ignore_on_load_missing or set()
        unexpected_patterns = (self._keys_to_ignore_on_load_unexpected or set()) | additional_unexpected_patterns
        ignore_missing_regex, ignore_unexpected_regex = None, None
        if len(missing_patterns) > 0:
            ignore_missing_regex = re.compile("|".join(rf"({pattern})" for pattern in missing_patterns))
        if len(unexpected_patterns) > 0:
            ignore_unexpected_regex = re.compile("|".join(rf"({pattern})" for pattern in unexpected_patterns))

        # Clean-up missing keys
        if ignore_missing_regex is not None:
            loading_info.missing_keys = {
                key for key in loading_info.missing_keys if ignore_missing_regex.search(key) is None
            }

        # Clean-up unexpected keys
        if ignore_unexpected_regex is not None:
            loading_info.unexpected_keys = {
                key for key in loading_info.unexpected_keys if ignore_unexpected_regex.search(key) is None
            }

    def mark_tied_weights_as_initialized(self, loading_info):
        """Adds the `_is_hf_initialized` flag on parameters that will be tied, in order to avoid initializing them
        later as they will be tied (overwritten) anyway.
        This is very important as most embeddings are tied, and they are huge params (vocabularies are often 256k), so
        running inits on them is very costly."""
        for tied_param in getattr(self, "all_tied_weights_keys", {}).keys():
            param = self.get_parameter(tied_param)
            setattr(param, "_is_hf_initialized", True)

        # Some custom code models define module tying (not parameter tying) in their __init__. When modules themselves are shared,
        # weights inside both modules appear in the `state_dict` but only one will appear in the safetensors checkpoints
        # as they are inherently tied because the 2 modules are the same object. In this case, once we load a parameter
        # inside one of the 2 modules, the other will also automatically be loaded and will have the `_is_hf_initialized`
        # flag (because we call `setattr` with the loaded param on the module, which is the same object), but its counterpart
        # will still appear as a missing key as we never get it out of the set (because it appears in the state_dict as well).
        # So we remove it now - otherwise it's considered missing and will be wrongly reinitialized
        # Note: this is never an issue in main Transformers, as we never do module-tying, only parameter-tying, and we know
        # which params are supposed to be tied to which other params
        # if self.is_custom_code():
        #     # Remove those that are already initialized, but appear as missing due to module tying (only if they are not known
        #     # tied weights, i.e. we did not explicitly mark them as initialized just above)
        #     loading_info.missing_keys = {
        #         key
        #         for key in loading_info.missing_keys
        #         if key in self.all_tied_weights_keys
        #         or not getattr(self.get_parameter_or_buffer(key), "_is_hf_initialized", False)
        #     }
        
class Wav2Vec2Config(ModelConfig):
    do_stable_layer_norm: bool = False
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    hidden_size: int = 768
    attention_dropout: float | int = 0.1

class GradientCheckpointingLayer(nn.Module):
    gradient_checkpointing = False
    # Layers that only read the KV cache can set this to keep it under gradient checkpointing (the recompute reads the
    # same states). Writers must leave it `False`, otherwise the cache is updated a second time on the backward replay.
    _can_checkpoint_with_cache = False

class Wav2Vec2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Wav2Vec2Config | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

class Wav2Vec2EncoderLayer(GradientCheckpointingLayer):
     def __init__(self, config):
        super().__init__()
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            config=config,
        )

class Wav2Vec2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            Wav2Vec2EncoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        self.gradient_checkpointing = False

class Wav2Vec2PreTrainedModel(Model):
    config: Wav2Vec2Config
    base_model_prefix = "wav2vec2"
    main_input_name = "input_values"
    input_modalities = "audio"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True


class Wav2Vec2Model(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.config = config

        if config.do_stable_layer_norm:
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config)
        else:
            self.encoder = Wav2Vec2Encoder(config)

        # Needed to be called here
        self.post_init()

class UniqueModel(Wav2Vec2PreTrainedModel):
    config_class = Wav2Vec2Config
    # Try and mock mms
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)

def _resolve_checkpoint_files(
    pretrained_model_name_or_path: str | Path,
) -> list[Path]:
    # Safetensors, single file.
    try:
        return [
            ensure_file(
                pretrained_model_name_or_path,
                SAFE_WEIGHTS_NAME,
            )
        ]
    except FileNotFoundError:
        pass

    # PyTorch, single file.
    try:
        return [
            ensure_file(
                pretrained_model_name_or_path,
                WEIGHTS_NAME,
            )
        ]
    except FileNotFoundError:
        pass

    raise FileNotFoundError(
        "Could not find model weights. Expected one of: "
        f"{SAFE_WEIGHTS_NAME!r}, {WEIGHTS_NAME!r}"
    )
