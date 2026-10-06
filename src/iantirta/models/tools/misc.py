
import os
from collections.abc import Callable
from types import ModuleType


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
