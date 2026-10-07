
import json
import logging
import os
from collections.abc import Callable
from contextlib import nullcontext
from functools import wraps
from types import ModuleType
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import torch


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})


# vendored from distutils.util
def strtobool(val) -> int:
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values are 'n', 'no', 'f', 'false', 'off', and '0'.
    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in {"y", "yes", "t", "true", "on", "1"}:
        return 1
    if val in {"n", "no", "f", "false", "off", "0"}:
        return 0
    raise ValueError(f"invalid truth value {val!r}")


def is_env_variable_true(env_variable: str) -> bool:
    """Detect whether `env_variable` has been set to a true value in the environment"""
    return os.getenv(env_variable, "false").lower() in ("true", "1", "y", "yes", "on")


def is_env_variable_false(env_variable: str) -> bool:
    """Detect whether `env_variable` has been set to a false value in the environment"""
    return os.getenv(env_variable, "true").lower() in ("false", "0", "n", "no", "off")


def maybe_import_error(message: str, *, raise_error: bool) -> bool:
    """Report an unmet dependency precondition: raise `ImportError(message)` when `raise_error`, else
    return `False`. Lets an `is_*_available` / `is_*_loadable` check read as a flat
    `if unmet: return maybe_import_error(msg, ...)` — a bool for callers probing availability, the specific
    error for callers that want to fail loudly (`raise_error=True`)."""
    if raise_error:
        raise ImportError(message)
    return False


def resolve_internal_import(module: ModuleType | None, chained_path: str) -> Callable | ModuleType | None:
    """
    Check if a given `module` has an internal import path as defined by the `chained_path`.
    This can either be the full path (not exposed in `__init__`) OR the last part of the chain (exposed in `__init__`).

    This is an important helper function for kernels based modules to apply the import from the module
    itself, i.e. stay compatible with original libraries in certain cases.

    Example:
        Module: `mamba_ssm`
        Chained Path: `ops.triton.selective_state_update.selective_state_update`
        Resulting import attempt at:
            - `mamba_ssm.selective_state_update`
            - `mamba_ssm.ops.triton.selective_state_update.selective_state_update`
    """
    if not module:
        return None

    if final_module := getattr(module, chained_path.split(".")[-1], None):
        return final_module

    final_module = module
    for path in chained_path.split("."):
        final_module = getattr(final_module, path, None)
        if not final_module:
            return None
    return final_module


def can_return_tuple(func):
    """
    Decorator to wrap model method, to call output.to_tuple() if return_dict=False passed as a kwarg or
    return_dict=False is set in the config.

    Note:
        output.to_tuple() convert output to tuple skipping all `None` values.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return_dict = self.config.return_dict if hasattr(self, "config") else True
        return_dict_passed = kwargs.pop("return_dict", return_dict)
        if return_dict_passed is not None:
            return_dict = return_dict_passed
        output = func(self, *args, **kwargs)
        if not return_dict and not isinstance(output, tuple):
            output = output.to_tuple()
        return output

    return wrapper


def merge_with_config_defaults(func):
    """
    Decorator using config field (if they exist) as default value for some args and kwargs. Precedence is always
    given to the args/kwargs that are explicitly passed.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        args_with_config_defaults = [
            "use_cache",
            "vision_feature_layer",
            "vision_feature_select_strategy",
            "vision_aspect_ratio",
        ]
        for arg_name in args_with_config_defaults:
            arg_index = None
            if arg_name in func.__code__.co_varnames:
                arg_index = func.__code__.co_varnames.index(arg_name) - 1  # -1 for self

            if arg_index is not None and len(args) > arg_index and args[arg_index] is not None:
                arg_value = args[arg_index]
            elif kwargs.get(arg_name) is not None:
                arg_value = kwargs[arg_name]
            else:
                arg_value = getattr(self.config, arg_name, None)

            if arg_value is not None:
                # Arg-specific handling
                if arg_name == "use_cache":
                    if getattr(self, "gradient_checkpointing", False) and self.training and arg_value:
                        logger.warning_once(
                            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                        )
                        arg_value = False
                elif arg_name == "vision_feature_select_strategy":
                    valid_strategies = ["default", "full"]
                    if arg_value not in valid_strategies:
                        raise ValueError(
                            f"`Unexpected select feature strategy: {arg_value}. Please select from {valid_strategies}."
                        )

                if arg_index is not None and len(args) > arg_index:
                    args = list(args)
                    args[arg_index] = arg_value
                    args = tuple(args)
                else:
                    kwargs[arg_name] = arg_value

        # Maybe temporarily overwrite config value to create the correct mask - kwarg takes precedence
        is_causal = kwargs.get("is_causal", getattr(self.config, "is_causal", None))
        if is_causal is not None:
            is_causal_in_config = hasattr(self.config, "is_causal")
            if is_causal_in_config:
                is_causal_original_value = self.config.is_causal
            # Set it to both config and kwargs (it's needed in both, and can come from only 1 of the sources)
            self.config.is_causal = is_causal
            kwargs["is_causal"] = is_causal

        # Call the original forward with the updated kwargs/config
        try:
            if kwargs.get("debug_io", False):
                from ..model_debugging_utils import model_addition_debugger_context

                with model_addition_debugger_context(
                    self, kwargs.get("debug_io_dir", "model_debug"), kwargs.get("prune_layers")
                ):
                    output = func(self, *args, **kwargs)
            else:
                output = func(self, *args, **kwargs)
        # Restore original config value
        finally:
            if is_causal is not None:
                if is_causal_in_config:
                    self.config.is_causal = is_causal_original_value
                else:
                    del self.config.is_causal

        return output

    return wrapper


def safe_load_json_file(json_file: str):
    "A helper to load safe config files and raise a proper error message if it wasn't serialized correctly"
    try:
        with open(json_file, encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)
    except json.JSONDecodeError:
        raise OSError(f"It looks like the config file at '{json_file}' is not a valid JSON file.")
    return config_dict


def maybe_autocast(
    device_type: str,
    dtype: "torch.dtype | None" = None,
    enabled: bool = True,
    cache_enabled: bool | None = None,
):
    """
    Context manager that only autocasts if:

    - `autocast` is already enabled in this context
    - Or this call to `maybe_autocast` has `enabled=True`

    This prevents `autocast` being added to the graph when it is effectively a no-op.
    Which makes graph splitting in `torch.compile` more flexible as it removes the
    requirement that partition IDs be monotonically increasing.
    """
    from ._torch import is_torch_available
    if not is_torch_available():
        raise ImportError("`maybe_autocast` requires PyTorch to be installed.")

    import torch

    if device_type == "meta":
        return nullcontext()
    if torch.is_autocast_enabled(device_type) or enabled:
        return torch.autocast(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
    else:
        return nullcontext()
