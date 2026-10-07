
import logging
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Thread
from typing import TYPE_CHECKING, TypeVar
from zipfile import is_zipfile

import torch
from safetensors import safe_open
from safetensors.torch import load as _safe_load_bytes
from torch import nn

from ...distributed.tensor_parallel import _get_parameter_tp_plan
from ...distributed.utils import (
    _get_torch_distributed_world_size,
    _is_torch_distributed_initialized,
)
from ...loading.safetensor_conversion import auto_conversion
from ...quantizers import HfQuantizer
from ...quantizers.quantizers_utils import get_module_from_name
from ...remote._types import DownloadKwargs
from ...remote.files import cached_file, get_checkpoint_shard_files, has_file
from ...tools._torch import check_torch_load_is_safe
from ...tools.misc import is_env_variable_true
from ..config import PreTrainedConfig

if TYPE_CHECKING:
    from iantirta.models.tools.types import DeviceMeshLike

    from ...loading.core import WeightConverter, WeightRenaming
    from .pretrained import PreTrainedModel


logger = logging.getLogger("iantirta.models.core.model")


WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"


MULTIPLE_CHOICE_DUMMY_INPUTS = [
    [[0, 1, 0, 1], [1, 0, 0, 1]]
] * 2  # Needs to have 0s and 1s only since XLM uses it for langs too.
DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]


# XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0").upper()
# XLA_DOWNCAST_BF16 = os.environ.get("XLA_DOWNCAST_BF16", "0").upper()
SpecificPreTrainedModelType = TypeVar("SpecificPreTrainedModelType", bound="PreTrainedModel")
_is_quantized = False
_is_ds_init_called = False


@dataclass(frozen=True)
class LoadStateDictConfig:
    """
    Config for loading weights. This allows bundling arguments that are just
    passed around.
    """

    pretrained_model_name_or_path: str | None = None
    download_kwargs: DownloadKwargs | None = field(default_factory=DownloadKwargs)
    use_safetensors: bool | None = None
    ignore_mismatched_sizes: bool = False
    sharded_metadata: dict | None = None
    device_map: dict | None = None
    disk_offload_folder: str | None = None
    offload_buffers: bool = False
    dtype: torch.dtype | None = None
    dtype_plan: dict = field(default_factory=dict)
    hf_quantizer: HfQuantizer | None = None
    device_mesh: "DeviceMeshLike | None" = None
    weights_only: bool = True
    weight_mapping: list["WeightConverter | WeightRenaming"] | None = None
    disable_mmap: bool | None = None

    @property
    def is_quantized(self) -> bool:
        return self.hf_quantizer is not None


@contextmanager
def set_quantized_state():
    global _is_quantized
    _is_quantized = True
    try:
        yield
    finally:
        _is_quantized = False


# Skip recursive calls to deepspeed.zero.Init to avoid pinning errors.
# This issue occurs with ZeRO stage 3 when using NVMe offloading.
# For more details, refer to issue #34429.
@contextmanager
def set_zero3_state():
    global _is_ds_init_called
    _is_ds_init_called = True
    try:
        yield
    finally:
        _is_ds_init_called = False


@contextmanager
def local_torch_dtype(dtype: torch.dtype, model_class_name: str | None = None):
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


def get_state_dict_dtype(state_dict):
    """
    Returns the first found floating dtype in `state_dict` if there is one, otherwise returns the first dtype.
    """
    for t in state_dict.values():
        # We cannot instantiate a whole model under float4/8_xxx dtypes (torch does not allow setting them as default dtype)
        if t.is_floating_point() and "float8_" not in str(t.dtype) and "float4_" not in str(t.dtype):
            return t.dtype

    # if no floating dtype was found return whatever the first dtype is
    if len(state_dict) == 0:
        return torch.float32
    return next(iter(state_dict.values())).dtype


str_to_torch_dtype = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "U16": torch.uint16,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "U32": torch.uint32,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "U64": torch.uint64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def _is_on_hf_mount(path: "str | os.PathLike") -> bool:
    """True if `path` lives on an hf-mount FUSE filesystem (device string 'hf-mount').

    hf-mount's mmap + readahead interaction deadlocks under parallel page-faults,
    so callers should load the file into memory instead. Linux-only; returns False
    on other platforms.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        real = os.path.realpath(os.fspath(path))
        with open("/proc/mounts", encoding="utf-8") as fh:
            entries = sorted(
                ((p[0], p[1]) for p in (l.split() for l in fh) if len(p) >= 2),
                key=lambda e: len(e[1]),
                reverse=True,
            )
        for dev, mp in entries:
            if real == mp or real.startswith(mp.rstrip("/") + "/"):
                return dev == "hf-mount"
    except (OSError, ValueError):
        pass
    return False


def load_state_dict(
    checkpoint_file: str | os.PathLike,
    map_location: str | torch.device = "cpu",
    weights_only: bool = True,
    disable_mmap: bool | None = None,
) -> dict[str, torch.Tensor]:
    """
    Reads a `safetensor` or a `.bin` checkpoint file. We load the checkpoint on "cpu" by default.

    When `disable_mmap` is True, safetensors files are read fully into memory instead of
    being memory-mapped. When `disable_mmap` is None (default), it is auto-detected to True
    on hf-mount FUSE filesystems (see `_is_on_hf_mount`).
    """
    checkpoint_path = os.fspath(checkpoint_file)
    if disable_mmap is None:
        disable_mmap = _is_on_hf_mount(checkpoint_path)
    # Use safetensors if possible
    if checkpoint_path.endswith(".safetensors"):
        if disable_mmap and map_location != "meta":
            with open(checkpoint_path, "rb") as _fh:
                state_dict = _safe_load_bytes(_fh.read())
            if map_location != "cpu":
                state_dict = {k: v.to(map_location) for k, v in state_dict.items()}
            return state_dict
        with safe_open(checkpoint_path, framework="pt") as f:
            state_dict = {}
            for k in f:
                if map_location == "meta":
                    _slice = f.get_slice(k)
                    k_dtype = _slice.get_dtype()
                    if k_dtype in str_to_torch_dtype:
                        dtype = str_to_torch_dtype[k_dtype]
                    else:
                        raise ValueError(f"Cannot load safetensors of unknown dtype {k_dtype}")
                    state_dict[k] = torch.empty(size=_slice.get_shape(), dtype=dtype, device="meta")
                else:
                    state_dict[k] = f.get_tensor(k).to(map_location)
            return state_dict

    # Fallback to torch.load (if weights_only was explicitly False, do not check safety as this is known to be unsafe)
    if weights_only:
        check_torch_load_is_safe()
    extra_args = {}
    # mmap can only be used with files serialized with zipfile-based format.
    if map_location != "meta" and is_zipfile(checkpoint_path):
        extra_args = {"mmap": True}

    return torch.load(checkpoint_path, map_location=map_location, weights_only=weights_only, **extra_args)


def _load_parameter_into_model(model: "PreTrainedModel", param_name: str, tensor: torch.Tensor):
    """Cast a single parameter or buffer `param_name` into the `model`, with value `tensor`."""
    parent, param_type = get_module_from_name(model, param_name)
    if param_type in parent._parameters and not isinstance(tensor, nn.Parameter):
        tensor = nn.Parameter(tensor, requires_grad=tensor.is_floating_point())
    # We need to use setattr here, as we set non-persistent buffers as well with this function (`load_state_dict`
    # does not allow to do it)
    setattr(parent, param_type, tensor)


def _add_variant(weights_name: str, variant: str | None = None) -> str:
    if variant is not None:
        path, name = weights_name.rsplit(".", 1)
        weights_name = f"{path}.{variant}.{name}"
    return weights_name


def _get_resolved_checkpoint_files(
    pretrained_model_name_or_path: str | os.PathLike | None,
    variant: str | None,
    gguf_file: str | None,
    use_safetensors: bool | None,
    user_agent: dict | None,
    is_remote_code: bool,  # Because we can't determine this inside this function, we need it to be passed in
    transformers_explicit_filename: str | None = None,
    download_kwargs: DownloadKwargs | None = None,
    tqdm_class: type | None = None,
) -> tuple[list[str] | None, dict | None]:
    """Get all the checkpoint filenames based on `pretrained_model_name_or_path`, and optional metadata if the
    checkpoints are sharded.
    This function will download the data if necessary.
    """
    download_kwargs = download_kwargs or DownloadKwargs()
    cache_dir = download_kwargs.get("cache_dir")
    force_download = download_kwargs.get("force_download", False)
    proxies = download_kwargs.get("proxies")
    local_files_only = download_kwargs.get("local_files_only", False)
    token = download_kwargs.get("token")
    revision = download_kwargs.get("revision") or "main"
    subfolder = download_kwargs.get("subfolder", "")
    if transformers_explicit_filename is not None:  # noqa: SIM102
        if not transformers_explicit_filename.endswith(".safetensors") and not transformers_explicit_filename.endswith(  # noqa: SIM102
            ".safetensors.index.json"
        ):
            if transformers_explicit_filename != "adapter_model.bin":
                raise ValueError(
                    "The transformers file in the config seems to be incorrect: it is neither a safetensors file "
                    "(*.safetensors) nor a safetensors index file (*.safetensors.index.json): "
                    f"{transformers_explicit_filename}"
                )

    is_sharded = False

    if pretrained_model_name_or_path is not None and gguf_file is None:
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        # If the file is a local folder (but not in the HF_HOME cache, even if it's technically local)
        if is_local:
            if transformers_explicit_filename is not None:
                # If the filename is explicitly defined, load this by default
                base_dir = os.path.join(pretrained_model_name_or_path, subfolder)
                archive_file = os.path.join(base_dir, transformers_explicit_filename)
                # Just a small check to make sure `transformers_explicit_filename` does not escape the base_dir, i.e. it does not
                # contain `..` for example
                try:
                    absolute_base_dir = os.path.abspath(base_dir)
                    absolute_archive_file = os.path.abspath(archive_file)
                    contained = os.path.commonpath([absolute_base_dir, absolute_archive_file]) == absolute_base_dir
                except ValueError:
                    contained = False
                if not contained:
                    raise ValueError(
                        f"`transformers_weights` must reference a file inside the model directory, got {transformers_explicit_filename}"
                    )
                is_sharded = transformers_explicit_filename.endswith(".safetensors.index.json")
            elif use_safetensors is not False and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))
            ):
                # Load from a safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant)
                )
            elif use_safetensors is not False and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
            ):
                # Load from a sharded safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                )
                is_sharded = True
            elif not use_safetensors and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant))
            ):
                # Load from a PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant)
                )
            elif not use_safetensors and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))
            ):
                # Load from a sharded PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant)
                )
                is_sharded = True
            elif use_safetensors:
                raise OSError(
                    f"Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)} found in directory"
                    f" {pretrained_model_name_or_path}."
                )
            else:
                raise OSError(
                    f"Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)}, or {_add_variant(WEIGHTS_NAME, variant)},"
                    f" found in directory {pretrained_model_name_or_path}."
                )
        elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
            archive_file = pretrained_model_name_or_path
            is_local = True
        else:
            # set correct filename
            if transformers_explicit_filename is not None:
                filename = transformers_explicit_filename
                is_sharded = transformers_explicit_filename.endswith(".safetensors.index.json")
            elif use_safetensors is not False:
                filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
            else:
                filename = _add_variant(WEIGHTS_NAME, variant)

            # Prepare set of kwargs for hub functions
            has_file_kwargs = {
                "revision": revision,
                "proxies": proxies,
                "token": token,
                "cache_dir": cache_dir,
                "local_files_only": local_files_only,
            }
            cached_file_kwargs = {
                "force_download": force_download,
                "user_agent": user_agent,
                "subfolder": subfolder,
                "_raise_exceptions_for_gated_repo": False,
                "_raise_exceptions_for_missing_entries": False,
                "tqdm_class": tqdm_class,
                **has_file_kwargs,
            }
            can_auto_convert = (
                # If we are in a CI environment or in a pytest run, we prevent the conversion
                not is_env_variable_true("DISABLE_SAFETENSORS_CONVERSION")
                and not is_remote_code  # converter bot does not work on remote code
                and subfolder == ""  # converter bot does not work on subfolders
            )

            try:
                # Load from URL or cache if already cached
                # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                # result when internet is up, the repo and revision exist, but the file does not.
                resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

                # Try safetensors files first if not already found
                if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True
                    elif use_safetensors:
                        if revision == "main" and can_auto_convert:
                            resolved_archive_file, revision, is_sharded = auto_conversion(
                                pretrained_model_name_or_path, **cached_file_kwargs
                            )
                        cached_file_kwargs["revision"] = revision
                        if resolved_archive_file is None:
                            raise OSError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} "
                                "and thus cannot be loaded with `safetensors`. Please do not set `use_safetensors=True`."
                            )
                    else:
                        # This repo has no safetensors file of any kind, we switch to PyTorch.
                        filename = _add_variant(WEIGHTS_NAME, variant)
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path, filename, **cached_file_kwargs
                        )

                # Then try `.bin` files
                if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        _add_variant(WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True

                # If we have a match, but it's `.bin` format, try to launch safetensors conversion for next time
                if resolved_archive_file is not None:
                    safe_weights_name = SAFE_WEIGHTS_INDEX_NAME if is_sharded else SAFE_WEIGHTS_NAME
                    if (
                        filename in [WEIGHTS_NAME, WEIGHTS_INDEX_NAME]
                        and not has_file(pretrained_model_name_or_path, safe_weights_name, **has_file_kwargs)
                        and can_auto_convert
                    ):
                        Thread(
                            target=auto_conversion,
                            args=(pretrained_model_name_or_path,),
                            kwargs={"ignore_errors_during_conversion": True, **cached_file_kwargs},
                            name="Thread-auto_conversion",
                        ).start()

                # If no match, raise appropriare errors
                else:
                    # Otherwise, no PyTorch file was found
                    if variant is not None and has_file(
                        pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs
                    ):
                        raise OSError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file without the variant"
                            f" {variant}. Use `variant=None` to load this model from those weights."
                        )
                    else:
                        raise OSError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_NAME, variant)}."
                        )

            except OSError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                # to the original exception.
                raise
            except Exception as e:
                # For any other exception, we throw a generic error.
                raise OSError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)}."
                ) from e

        if is_local:
            logger.info(f"loading weights file {archive_file}")
            resolved_archive_file = archive_file
        else:
            logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")

    elif gguf_file:
        # Case 1: the GGUF file is present locally
        if os.path.isfile(gguf_file):
            resolved_archive_file = gguf_file
        # Case 2: The GGUF path is a location on the Hub
        # Load from URL or cache if already cached
        else:
            cached_file_kwargs = {
                "cache_dir": cache_dir,
                "force_download": force_download,
                "proxies": proxies,
                "local_files_only": local_files_only,
                "token": token,
                "user_agent": user_agent,
                "revision": revision,
                "subfolder": subfolder,
                "_raise_exceptions_for_gated_repo": False,
                "_raise_exceptions_for_missing_entries": False,
            }

            resolved_archive_file = cached_file(pretrained_model_name_or_path, gguf_file, **cached_file_kwargs)

    # We now download and resolve all checkpoint files if the checkpoint is sharded
    sharded_metadata = None
    if is_sharded:
        checkpoint_files, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_archive_file,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            user_agent=user_agent,
            revision=revision,
            subfolder=subfolder,
            tqdm_class=tqdm_class,
        )
    else:
        checkpoint_files = [resolved_archive_file] if pretrained_model_name_or_path is not None else None

    return checkpoint_files, sharded_metadata


def _get_dtype(
    dtype: str | torch.dtype | dict | None,
    checkpoint_files: list[str] | None,
    config: PreTrainedConfig,
    sharded_metadata: dict | None,
    state_dict: dict | None,
    weights_only: bool,
    hf_quantizer: HfQuantizer | None = None,
) -> tuple[PreTrainedConfig, torch.dtype]:
    """Find the correct `dtype` to use based on provided arguments. Also update the `config` based on the
    inferred dtype. We do the following:
    1. If dtype is "auto", we try to read the config, else auto-detect dtype from the loaded state_dict, by checking
    its first weights entry that is of a floating type - we assume all floating dtype weights are of the same dtype
    2. Else, use the dtype provided as a dict or str
    """
    is_sharded = sharded_metadata is not None

    if dtype is not None:
        if isinstance(dtype, str):
            if dtype == "auto":
                if hasattr(config, "dtype") and config.dtype is not None:
                    dtype = config.dtype
                    logger.info(f"Will use dtype={dtype} as defined in model's config object")
                else:
                    if is_sharded and "dtype" in sharded_metadata:
                        dtype = sharded_metadata["dtype"]
                    elif state_dict is not None:
                        dtype = get_state_dict_dtype(state_dict)
                    elif checkpoint_files is not None and checkpoint_files[0].endswith(".gguf"):
                        dtype = None
                    else:
                        state_dict = load_state_dict(
                            checkpoint_files[0], map_location="meta", weights_only=weights_only
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

    if hf_quantizer is not None:
        main_dtype = hf_quantizer.update_dtype(main_dtype)

    # Set it on the config and subconfigs
    config.dtype = main_dtype
    for sub_config_key in config.sub_configs:
        if (sub_config := getattr(config, sub_config_key)) is not None:
            sub_config.dtype = main_dtype

    return config, main_dtype


def is_accelerator_device(device: str | int | torch.device) -> bool:
    """Check if the device is an accelerator. We need to function, as device_map can be "disk" as well, which is not
    a proper `torch.device`.
    """
    if device == "disk":
        return False
    else:
        return torch.device(device).type not in ["meta", "cpu"]


def get_total_byte_count(
    model: "PreTrainedModel", accelerator_device_map: dict, hf_quantizer: HfQuantizer | None = None
):
    """
    This utility function calculates the total bytes count needed to load the model on each device.
    This is useful for caching_allocator_warmup as we want to know how much cache we need to pre-allocate.
    """

    total_byte_count = defaultdict(lambda: 0)
    tied_param_names = model.all_tied_weights_keys.keys()
    tp_plan = model.tp_plan if _is_torch_distributed_initialized() else []

    for param_name, device in accelerator_device_map.items():
        # Skip if the parameter has already been accounted for (tied weights)
        if param_name in tied_param_names:
            continue

        param = model.get_parameter_or_buffer(param_name)

        if hf_quantizer is not None:
            dtype_size = hf_quantizer.param_element_size(model, param_name, param)
        else:
            dtype_size = param.element_size()

        param_byte_count = param.numel() * dtype_size

        if len(tp_plan) > 0:
            is_part_of_plan = _get_parameter_tp_plan(param_name, tp_plan, is_weight=True) is not None
            param_byte_count //= _get_torch_distributed_world_size() if is_part_of_plan else 1

        total_byte_count[device] += param_byte_count
    return total_byte_count


def caching_allocator_warmup(model: "PreTrainedModel", expanded_device_map: dict, hf_quantizer: HfQuantizer | None):
    """This function warm-ups the caching allocator based on the size of the model tensors that will reside on each
    device. It allows to have one large call to Malloc, instead of recursively calling it later when loading
    the model, which is actually the loading speed bottleneck.
    Calling this function allows to cut the model loading time by a very large margin.

    A few facts related to loading speed (taking into account the use of this function):
    - When loading a model the first time, it is usually slower than the subsequent times, because the OS is very likely
    to cache the different state dicts (if enough resources/RAM are available)
    - Trying to force the OS to cache the files in advance (by e.g. accessing a small portion of them) is really hard,
    and not a good idea in general as this is low level OS optimizations that depend on resource usage anyway
    - As of 18/03/2025, loading a Llama 70B model with TP takes ~1 min without file cache, and ~13s with full file cache.
    The baseline, i.e. only loading the tensor shards on device and adjusting dtype (i.e. copying them) is ~5s with full cache.
    These numbers are reported for TP on 4 H100 GPUs.
    - It is useless to pre-allocate more than the model size in this function (i.e. using an `allocation_factor` > 1) as
    cudaMalloc is not a bottleneck at all anymore
    - Loading speed bottleneck is now almost only tensor copy (i.e. changing the dtype) and moving the tensors to the devices.
    However, we cannot really improve on those aspects obviously, as the data needs to be moved/copied in the end.
    """
    # Remove disk, cpu and meta devices, and cast to proper torch.device
    accelerator_device_map = {
        param: torch.device(device) for param, device in expanded_device_map.items() if is_accelerator_device(device)
    }
    if not accelerator_device_map:
        return

    total_byte_count = get_total_byte_count(model, accelerator_device_map, hf_quantizer)

    # This will kick off the caching allocator to avoid having to Malloc afterwards
    for device, byte_count in total_byte_count.items():
        if device.type in ["cuda", "xpu"]:
            accelerator_module = getattr(torch, device.type)
            index = device.index if device.index is not None else accelerator_module.current_device()
            try:
                free_device_memory, total_device_memory = accelerator_module.mem_get_info(index)
            except (RuntimeError, NotImplementedError, AttributeError) as e:
                # Some backends cannot report free memory (e.g. Intel XPU under WSL2, where the Level Zero Sysman
                # interface is not exposed). Warmup is a best-effort optimization, so skip it for this device
                # instead of failing the whole model load.
                logger.warning_once(
                    f"Skipping caching allocator warmup for {device}: could not query device memory ({e})"
                )
                continue
            unused_memory = accelerator_module.memory_reserved(index) - accelerator_module.memory_allocated(index)
            # If we have reserved but unused memory, we can lower the allocation we want to make, but only if it's still
            # higher than the unused memory. This is because otherwise torch will use that unused memory when performing
            # our own allocation, thus not allocating any new memory from the GPU. For example if byte_count=6 GiB,
            # unused_memory=4 GiB, then we cannot allocate only 2 GiB as this would *likely* (may not be exact, due to
            # fragmentation issues) simply use the pool of 4 GiB unused memory that is available. In those cases, it's better
            # to allocate more than the technically only 2 GiB required
            if byte_count - unused_memory > unused_memory:
                byte_count = byte_count - unused_memory
            # Minimum amount that will trigger new gpu allocation, even if it's technically "too much" compared to what we need
            elif byte_count - unused_memory > 1.5 * 1024**3:
                # Nothing we can do here, the memory will need to fill itself as we load params, but we cannot reallocate
                # from gpu until the unused memory is not filled
                if unused_memory + 1 > free_device_memory:
                    byte_count = 0
                # We allocate the minimum amount that will force new gpu allocation, even if it's technically "too much"
                else:
                    byte_count = unused_memory + 1
            # If we only need to reallocate less than 1.5 GiB of what is already allocated, then don't allocate more
            else:
                byte_count = 0
            # Allow up to (max device memory - 1.2 GiB) in resource-constrained hardware configurations. Trying to reserve more
            # than that amount might sometimes lead to unnecessary cuda/xpu OOM, if the last parameter to be loaded on the device is large,
            # and the remaining reserved memory portion is smaller than the param size -> torch will then try to fully re-allocate all
            # the param size, instead of using the remaining reserved part, and allocating only the difference, which can lead
            # to OOM. See https://github.com/huggingface/transformers/issues/37436#issuecomment-2808982161 for more details.
            # Note that we use an absolute value instead of device proportion here, as a 8GiB device could still allocate too much
            # if using e.g. 90% of device size, while a 140GiB device would allocate too little
            byte_count = min(byte_count, total_device_memory - 1.2 * 1024**3)
        elif device.type == "mps":
            # Skip warmup on MPS: there is a limit of the maximum size a single buffer can have on MPS,
            # which from testing seems to be about 2/3 of the total device memory (tested on apple silicon).
            # This causes the warmup function to return a `RuntimeError: Invalid buffer size: XX.XX GiB`.
            # NOTE: not tested on intel macs
            continue
        elif device.type == "neuron":
            # Skip warmup on Neuron (AWS Trainium/Inferentia): it provides no benefit as there is no reusable memory pool
            continue
        # We divide by 2 here as we allocate in fp16
        _ = torch.empty(int(byte_count // 2), dtype=torch.float16, device=device, requires_grad=False)
