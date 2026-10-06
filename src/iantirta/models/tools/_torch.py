
import logging
import os
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import packaging.version
from packaging import version

from ._deps import _is_package_available, _make_compile_constant

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}

# Try to run a native pytorch job in an environment with TorchXLA installed by setting this value to 0.
USE_TORCH_XLA = os.environ.get("USE_TORCH_XLA", "1").upper()
TORCHAO_MIN_VERSION = "0.15.0"


@lru_cache
@_make_compile_constant
def is_torch_available() -> bool:
    try:
        is_available, torch_version = _is_package_available(
            "torch", return_version=True
        )
        parsed_version = version.parse(torch_version)
        if is_available and parsed_version < version.parse("2.5.0"):
            logger.warning_once(
                "Disabling PyTorch because PyTorch >= 2.5 "
                f"is required but found {torch_version}"
            )
        return is_available and (
            version.parse(torch_version) >= version.parse("2.5.0")
        )
    except packaging.version.InvalidVersion:
        return False


@lru_cache
@_make_compile_constant
def get_torch_version() -> str:
    _, torch_version = _is_package_available("torch", return_version=True)
    return torch_version


@lru_cache
@_make_compile_constant
def is_torch_greater_or_equal(library_version: str, accept_dev: bool = False) -> bool:
    """
    Accepts a library version and returns True if the current version of the library is greater than or equal to the
    given version. If `accept_dev` is True, it will also accept development versions (e.g. 2.7.0.dev20250320 matches
    2.7.0).
    """
    if not is_torch_available():
        return False

    if accept_dev:
        return version.parse(version.parse(get_torch_version()).base_version) >= version.parse(library_version)
    else:
        return version.parse(get_torch_version()) >= version.parse(library_version)


@lru_cache
@_make_compile_constant
def is_torch_less_or_equal(library_version: str, accept_dev: bool = False) -> bool:
    """
    Accepts a library version and returns True if the current version of the library is less than or equal to the
    given version. If `accept_dev` is True, it will also accept development versions (e.g. 2.7.0.dev20250320 matches
    2.7.0).
    """
    if not is_torch_available():
        return False

    if accept_dev:
        return version.parse(version.parse(get_torch_version()).base_version) <= version.parse(library_version)
    else:
        return version.parse(get_torch_version()) <= version.parse(library_version)

@lru_cache
@_make_compile_constant
def is_torch_flex_attn_available() -> bool:
    return is_torch_available() and version.parse(get_torch_version()) >= version.parse("2.5.0")


@lru_cache
@_make_compile_constant
def is_torch_cuda_available() -> bool:
    if is_torch_available():
        import torch

        return torch.cuda.is_available()
    return False


@lru_cache
@_make_compile_constant
def is_torch_xpu_available(check_device: bool = False) -> bool:
    """
    Checks if XPU acceleration is available via stock PyTorch (>=2.6) and
    potentially if a XPU is in the environment.
    """
    if not is_torch_available():
        return False

    torch_version = version.parse(get_torch_version())
    if torch_version.major == 2 and torch_version.minor < 6:
        return False

    import torch

    if check_device:
        try:
            # Will raise a RuntimeError if no XPU is found
            _ = torch.xpu.device_count()
            return torch.xpu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "xpu") and torch.xpu.is_available()


@lru_cache
@_make_compile_constant
def is_torch_npu_available(check_device=False) -> bool:
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if not is_torch_available() or not _is_package_available("torch_npu")[0]:
        return False

    import torch
    import torch_npu  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no NPU is found
            if hasattr(torch, "npu"):
                _ = torch.npu.device_count()
                return torch.npu.is_available()
            return False
        except RuntimeError:
            return False
    return hasattr(torch, "npu") and torch.npu.is_available()


@lru_cache
@_make_compile_constant
def is_torch_mlu_available() -> bool:
    """
    Checks if `mlu` is available via an `cndev-based` check which won't trigger the drivers and leave mlu
    uninitialized.
    """
    if not is_torch_available() or not _is_package_available("torch_mlu")[0]:
        return False

    import torch
    import torch_mlu  # noqa: F401

    pytorch_cndev_based_mlu_check_previous_value = os.environ.get("PYTORCH_CNDEV_BASED_MLU_CHECK")
    try:
        os.environ["PYTORCH_CNDEV_BASED_MLU_CHECK"] = str(1)
        available = torch.mlu.is_available() if hasattr(torch, "mlu") else False
    finally:
        if pytorch_cndev_based_mlu_check_previous_value:
            os.environ["PYTORCH_CNDEV_BASED_MLU_CHECK"] = pytorch_cndev_based_mlu_check_previous_value
        else:
            os.environ.pop("PYTORCH_CNDEV_BASED_MLU_CHECK", None)

    return available


@lru_cache
@_make_compile_constant
def is_torch_musa_available(check_device=False) -> bool:
    "Checks if `torch_musa` is installed and potentially if a MUSA is in the environment"
    if not is_torch_available() or not _is_package_available("torch_musa")[0]:
        return False

    import torch
    import torch_musa  # noqa: F401

    torch_musa_min_version = "0.33.0"
    accelerate_available, accelerate_version = _is_package_available("accelerate", return_version=True)
    if accelerate_available and version.parse(accelerate_version) < version.parse(torch_musa_min_version):
        return False

    if check_device:
        try:
            # Will raise a RuntimeError if no MUSA is found
            if hasattr(torch, "musa"):
                _ = torch.musa.device_count()
                return torch.musa.is_available()
            return False
        except RuntimeError:
            return False
    return hasattr(torch, "musa") and torch.musa.is_available()


@lru_cache
@_make_compile_constant
def is_torch_xla_available(check_is_tpu=False, check_is_gpu=False) -> bool:
    """
    Check if `torch_xla` is available. To train a native pytorch job in an environment with torch xla installed, set
    the USE_TORCH_XLA to false.
    """
    assert not (check_is_tpu and check_is_gpu), "The check_is_tpu and check_is_gpu cannot both be true."

    torch_xla_available = USE_TORCH_XLA in ENV_VARS_TRUE_VALUES and _is_package_available("torch_xla")[0]
    if not torch_xla_available:
        return False

    import torch_xla

    if check_is_gpu:
        return torch_xla.runtime.device_type() in ["GPU", "CUDA"]
    elif check_is_tpu:
        return torch_xla.runtime.device_type() == "TPU"

    return True


@lru_cache
@_make_compile_constant
def is_torch_hpu_available() -> bool:
    "Checks if `torch.hpu` is available and potentially if a HPU is in the environment"
    if (
        not is_torch_available()
        or not _is_package_available("habana_frameworks")[0]
        or not _is_package_available("habana_frameworks.torch")[0]
    ):
        return False

    torch_hpu_min_accelerate_version = "1.5.0"
    accelerate_available, accelerate_version = _is_package_available("accelerate", return_version=True)
    if accelerate_available and version.parse(accelerate_version) < version.parse(torch_hpu_min_accelerate_version):
        return False

    import torch

    if os.environ.get("PT_HPU_LAZY_MODE", "1") == "1":
        # import habana_frameworks.torch in case of lazy mode to patch torch with torch.hpu
        import habana_frameworks.torch  # noqa: F401

    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        return False

    # We patch torch.gather for int64 tensors to avoid a bug on Gaudi
    # Graph compile failed with synStatus 26 [Generic failure]
    # This can be removed once bug is fixed but for now we need it.
    original_gather = torch.gather

    def patched_gather(input: torch.Tensor, dim: int, index: torch.LongTensor) -> torch.Tensor:
        if input.dtype == torch.int64 and input.device.type == "hpu":
            return original_gather(input.to(torch.int32), dim, index).to(torch.int64)
        else:
            return original_gather(input, dim, index)

    torch.gather = patched_gather
    torch.Tensor.gather = patched_gather

    original_take_along_dim = torch.take_along_dim

    def patched_take_along_dim(input: torch.Tensor, indices: torch.LongTensor, dim: int | None = None) -> torch.Tensor:
        if input.dtype == torch.int64 and input.device.type == "hpu":
            return original_take_along_dim(input.to(torch.int32), indices, dim).to(torch.int64)
        else:
            return original_take_along_dim(input, indices, dim)

    torch.take_along_dim = patched_take_along_dim

    original_cholesky = torch.linalg.cholesky

    def safe_cholesky(A, *args, **kwargs):
        output = original_cholesky(A, *args, **kwargs)

        if torch.isnan(output).any():
            jitter_value = 1e-9
            diag_jitter = torch.eye(A.size(-1), dtype=A.dtype, device=A.device) * jitter_value
            output = original_cholesky(A + diag_jitter, *args, **kwargs)

        return output

    torch.linalg.cholesky = safe_cholesky

    original_scatter = torch.scatter

    def patched_scatter(
        input: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        if input.device.type == "hpu" and input is src:
            return original_scatter(input, dim, index, src.clone(), *args, **kwargs)
        else:
            return original_scatter(input, dim, index, src, *args, **kwargs)

    torch.scatter = patched_scatter
    torch.Tensor.scatter = patched_scatter

    # IlyasMoutawwakil: we patch torch.compile to use the HPU backend by default
    # https://github.com/huggingface/transformers/pull/38790#discussion_r2157043944
    # This is necessary for cases where torch.compile is used as a decorator (defaulting to inductor)
    # https://github.com/huggingface/transformers/blob/af6120b3eb2470b994c21421bb6eaa76576128b0/src/transformers/models/modernbert/modeling_modernbert.py#L204
    original_compile = torch.compile

    def hpu_backend_compile(*args, **kwargs):
        if kwargs.get("backend") not in ["hpu_backend", "eager"]:
            logger.warning(
                f"Calling torch.compile with backend={kwargs.get('backend')} on a Gaudi device is not supported. "
                "We will override the backend with 'hpu_backend' to avoid errors."
            )
            kwargs["backend"] = "hpu_backend"

        return original_compile(*args, **kwargs)

    torch.compile = hpu_backend_compile

    return True


@lru_cache
@_make_compile_constant
def is_torchaudio_available() -> bool:
    return is_torch_available() and _is_package_available("torchaudio")[0]


@lru_cache
@_make_compile_constant
def is_torchcodec_available() -> bool:
    return _is_package_available("torchcodec")[0]


@lru_cache
@_make_compile_constant
def is_torchvision_available() -> bool:
    from ._vision import is_vision_available
    
    return is_vision_available() and is_torch_available() and _is_package_available("torchvision")[0]


@lru_cache
@_make_compile_constant
def is_torchao_available(min_version: str = TORCHAO_MIN_VERSION) -> bool:
    if not is_torch_available():
        return False
    is_available, torchao_version = _is_package_available("torchao", return_version=True)
    return is_available and version.parse(torchao_version) >= version.parse(min_version)


@lru_cache
@_make_compile_constant
def is_torch_distributed_available() -> bool:
    if not is_torch_available():
        return False
    import torch

    return torch.distributed.is_available()


@lru_cache
def is_rocm_platform() -> bool:
    if is_torch_available():
        import torch

        return torch.version.hip is not None
    return False


def is_torch_fx_proxy(x) -> bool:
    try:
        import torch.fx

        return isinstance(x, torch.fx.Proxy)
    except Exception:  # noqa: BLE001
        return False


def is_fake_tensor(x) -> bool:
    try:
        import torch

        return isinstance(x, torch._subclasses.FakeTensor)
    except Exception:  # noqa: BLE001
        return False


def is_torchdynamo_compiling() -> bool:
    # Importing torch._dynamo causes issues with PyTorch profiler (https://github.com/pytorch/pytorch/issues/130622)
    # hence rather relying on `torch.compiler.is_compiling()` when possible (torch>=2.3)
    try:
        import torch

        return torch.compiler.is_compiling()
    except Exception:  # noqa: BLE001
        return False


def is_torchdynamo_exporting() -> bool:
    try:
        import torch

        return torch.compiler.is_exporting()
    except Exception:  # noqa: BLE001
        return False


def is_jit_tracing() -> bool:
    try:
        import torch

        return torch.jit.is_tracing()
    except Exception:  # noqa: BLE001
        return False


# Deliberately not `@_make_compile_constant`: the answer flips during CUDA graph capture, so inlining
# it at trace time would bake a transient into the graph.
def is_cuda_stream_capturing() -> bool:
    try:
        import torch

        return torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


def is_tracing(tensor=None) -> bool:
    """Checks whether we are tracing a graph with dynamo (compile or export), torch.jit, torch.fx, jax.jit (with torchax) or
    CUDA stream capturing or FakeTensor"""

    # Note that `is_torchdynamo_compiling` checks both compiling and exporting (the export check is stricter and
    # only checks export)
    _is_tracing = is_torchdynamo_compiling() or is_jit_tracing() or is_cuda_stream_capturing()
    if tensor is not None:
        _is_tracing |= is_torch_fx_proxy(tensor)
        _is_tracing |= is_fake_tensor(tensor)
        _is_tracing |= is_jax_jitting(tensor)

    return _is_tracing


def is_jax_jitting(x):
    """returns True if we are inside of `jax.jit` context, False otherwise.

    When a torch model is being compiled with `jax.jit` using torchax,
    the tensor that goes through the model would be an instance of
    `torchax.tensor.Tensor`, which is a tensor subclass. This tensor has
    a `jax` method to return the inner Jax array
    (https://github.com/google/torchax/blob/13ce870a1d9adb2430333c27bb623469e3aea34e/torchax/tensor.py#L134).
    Here we use ducktyping to detect if the inner jax array is a jax Tracer
    then we are in tracing context. (See more at: https://github.com/jax-ml/jax/discussions/9241)

    Args:
      x: torch.Tensor

    Returns:
      bool: whether we are inside of jax jit tracing.
    """

    if not hasattr(x, "jax"):
        return False
    try:
        import jax

        return isinstance(x.jax(), jax.core.Tracer)
    except Exception:  # noqa: BLE001
        return False


def check_torch_load_is_safe() -> None:
    if not is_torch_greater_or_equal("2.6"):
        raise ValueError(
            "Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users "
            "to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply "
            "when loading files with safetensors."
            "\nSee the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434"
        )


def torch_compilable_check(cond: Any, msg: str | Callable[[], str], error_type: type[Exception] = ValueError) -> None:
    """
    Combines the functionalities of `torch._check`, `torch._check_with` and `torch._check_tensor_all_with` to provide a
    unified way to perform checks that are compatible with TorchDynamo (torch.compile & torch.export).

    The advantage of using `torch._check(cond, msg, error_type)` over `if cond: raise error_type(msg)` is that the former
    works as a truthfulness hint for TorchDynamo, instead of failing with a data-dependent control flow error during compilation.

    All checks using this method can be disabled in production environments by setting `TRANSFORMERS_DISABLE_TORCH_CHECK=1`.

    Args:
        cond (`bool`, `torch.Tensor` or `Callable[[], bool | torch.Tensor]`): The condition to check.
        msg (`str` or `Callable[[], str]`): The error message to display if the condition is not met.
        error_type (`type[Exception]`, *optional*, defaults to `ValueError`): The type of error to raise if the condition is not met.

    Raises:
        error_type: If the condition is not met.
    """
    if os.getenv("TRANSFORMERS_DISABLE_TORCH_CHECK", "0") == "1":
        return

    import torch

    # When tracing, msg may be an f-string with tensor values that dynamo can't trace
    # (callable/isinstance on it breaks). Check compilation first and use torch._check
    # without msg (it only serves as a compiler hint in that case).
    if is_tracing():
        if isinstance(cond, torch.Tensor):
            torch._check_tensor_all(cond)
        else:
            torch._check(cond)
        return

    if not callable(msg):
        # torch._check requires msg to be a callable but we want to keep the API simple for users
        def msg_callable():
            return msg
    else:
        msg_callable = msg

    if callable(cond):
        cond = cond()

    # These checks are also compiler hints for TorchDynamo telling
    # it that the condition is expected to be True during compilation
    if isinstance(cond, torch.Tensor):
        torch._check_tensor_all_with(error_type, cond, msg_callable)
    else:
        torch._check_with(error_type, cond, msg_callable)


def is_torch_device(x) -> bool:
    """
    Tests if `x` is a torch device or not. Safe to call even if torch is not installed.
    """
    if not is_torch_available():
        return False

    import torch

    return isinstance(x, torch.device)


def is_torch_dtype(x) -> bool:
    """
    Tests if `x` is a torch dtype or not. Safe to call even if torch is not installed.
    """
    if not is_torch_available():
        return False

    import torch

    if isinstance(x, str):
        if hasattr(torch, x):
            x = getattr(torch, x)
        else:
            return False
    return isinstance(x, torch.dtype)


def get_device_type(device: "torch.device | str | None" = None) -> str:
    """Type of a device (the current accelerator by default, else cpu), with AMD GPUs reported as rocm."""
    import torch

    if device is None:
        device = torch.accelerator.current_accelerator() or torch.device("cpu")
    device_type = torch.device(device).type if isinstance(device, str) else device.type
    return "rocm" if device_type == "cuda" and is_rocm_platform() else device_type
