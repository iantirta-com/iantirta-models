
from ._torch import _is_torch_available
from ._mlx import _is_mlx_available


def is_numpy_array(x) -> bool:
    """
    Tests if `x` is a numpy array or not.
    """
    import numpy as np

    return isinstance(x, np.ndarray)


def is_torch_tensor(x) -> bool:
    """
    Tests if `x` is a torch tensor or not.
    Safe to call even if torch is not installed.
    """
    if not _is_torch_available:
        return False

    import torch

    return isinstance(x, torch.Tensor)


def is_mlx_array(x) -> bool:
    """
    Tests if `x` is a mlx array or not.
    Safe to call even when mlx is not installed.
    """
    if not _is_mlx_available:
        return False

    import mlx.core as mx

    return isinstance(x, mx.array)


def is_torch_fx_proxy(x) -> bool:
    try:
        import torch.fx

        return isinstance(x, torch.fx.Proxy)
    except Exception:
        return False


def _get_tensor_type(x) -> str | None:
    """
    Tries to guess the framework of an object `x`
    from its repr (brittle but will help
    in `is_tensor` to try the frameworks in a smart order,
    without the need to import the frameworks).
    """
    if (type_str := str(type(x))).startswith("<class 'torch."):
        return "pt"
    elif type_str.startswith("<class 'numpy."):
        return "np"
    elif type_str.startswith("<class 'mlx."):
        return "mlx"


def _get_frameworks_test_func(x):
    """
    Returns an (ordered since we are in Python 3.7+)
    dictionary framework to test function,
    which places the framework we can guess
    from the repr first, then Numpy, then the others.
    """
    framework_to_test = {
        "pt": is_torch_tensor,
        "np": is_numpy_array,
        "mlx": is_mlx_array,
    }
    preferred_framework = _get_tensor_type(x)
    # We will test this one first, then numpy, then the others.
    frameworks = (
        []
        if preferred_framework is None
        else [preferred_framework]
    )

    if preferred_framework != "np":
        frameworks.append("np")

    frameworks.extend([
        f
        for f in framework_to_test
        if f not in [preferred_framework, "np"]
    ])

    return {
        f: framework_to_test[f]
        for f in frameworks
    }


def is_tensor(x) -> bool:
    """
    Tests if `x` is a `torch.Tensor`, `np.ndarray` or `mlx.array`
    in the order defined by `infer_framework_from_repr`
    """
    # This gives us a smart order to test the
    # frameworks with the corresponding tests.
    to_test_func = _get_frameworks_test_func(x)
    for test_func in to_test_func.values():
        if test_func(x):
            return True

    # Tracers
    if is_torch_fx_proxy(x):
        return True

    return False
