# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of huggingface_hub, improved by iantirta.com

from __future__ import annotations

import logging
import copy

from typing import Any, Self
from pathlib import Path
from dataclasses import dataclass, field

import torch
from torch import nn

from ..config import PreTrainedConfig
from ..config.mixin import PretrainedOptions
from ..exceptions import YetToImplement

logger = logging.getLogger(__name__)


@dataclass
class ModelPretrainedOptions(PretrainedOptions):
    config: PreTrainedConfig | str | Path | None = None

    code_revision: str | None = None

    adapter_kwargs: dict | None = None

    ignore_mismatched_sizes: bool = False
    use_safetensors: bool | None = None
    weights_only: bool = True
    fusion_config: dict[str, bool | dict[str, Any]] | None = None
    disable_mmap: bool | None = None

    state_dict: dict | None = None

    tqdm_class: type | None = None
    output_loading_info: bool = False

    dtype: str | torch.dtype | None = None
    torch_dtype: str | torch.dtype | None = None  # kept for BC
    device_map: str | dict | None = None
    max_memory: float | None = None
    offload_folder: str | None = None
    offload_buffers: bool = False

    quantization_config = None

    variant: str | None = None
    adapter_kwargs: dict = field(default_factory=dict)
    adapter_name: str = "default"

    generation_config = None

    distributed_config = None
    device_mesh = None

    tp_plan = None
    tp_size = None

    trust_remote_code: bool | None = None

    allow_all_kernels: bool = False
    use_kernels: bool = False
    kernel_config = None
    key_mapping: dict | None = None

    attn_implementation: str | None = None
    experts_implementation: str | None = None


class ModelMixin(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        *inputs,
        **kwargs
    ):
        super().__init__()
        if not isinstance(config,PreTrainedConfig):
            raise TypeError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    @classmethod
    def from_pretrained(
        cls: type[Self],
        pretrained_model_name_or_path: str | Path,
        *model_args,
        options: ModelPretrainedOptions | dict | None = None,
        **kwargs
    ) -> type[Self]:
        if not isinstance(options, ModelPretrainedOptions):
            if options and isinstance(options, dict):
                options = ModelPretrainedOptions.from_dict(options)
            elif kwargs and isinstance(kwargs, dict):
                options = ModelPretrainedOptions.from_dict(kwargs)

        if options.torch_dtype is not None:
            raise YetToImplement("torch dtype is deprecated")
        if options.dtype is None:
            options.dtype = "auto"

        has_standalone_tp_args = (
            options.tp_plan is not None
            or options.tp_size is not None
        )
        if has_standalone_tp_args:
            raise YetToImplement('tp plan or tp size is not supported.')

        if options.gguf_file is not None:
            raise YetToImplement("gguf file is not supported.")

        if options.distributed_config is not None:
            raise YetToImplement("distributed config is not supported.")

        if options.adapter_kwargs is not None:
            raise YetToImplement("adapter kwargs is not supported.")

        # resolve device map here...

        if options.attn_implementation is not None:
            raise YetToImplement("setting attn implementation is not supported.")

        if options.experts_implementation is not None:
            raise YetToImplement("setting experts implementation is not supported.")

        if options.kernel_config is not None:
            raise YetToImplement("kernel config is not supported.")

        if options.use_kernels:
            raise YetToImplement("use kernel is not supported.")

        if not isinstance(options.config, PreTrainedConfig):
            raise YetToImplement("loading config from model is not supported.")

        options.config = copy.deepcopy(options.config)

        # Quantization here...

        checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            options=options,
            is_remote_code=cls.is_remote_code(),
            transformers_explicit_filename=getattr(
                options.config,
                "transformers_weights",
                None
            ),
        )

        # dtype mapping here...

        if options.fusion_config is not None:
            raise YetToImplement("fusion Config is not supoorted")

        fusion_config = getattr(options.config, "fusion_config", None)
        if fusion_config is not None:
            raise YetToImplement("fusion Config is not supoorted")

        #model_init_context = cls.get_init_context(dtype, is_quantized, _is_ds_init_called, allow_all_kernels)

        #config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        # with ContextManagers(model_init_context):
        #     model = cls(
        #         config,
        #         *model_args,
        #         **asdict(options)
        #     )
        #     patch_output_recorders(model)
        

def _add_variant(weights_name: str, variant: str | None = None) -> str:
    if variant is not None:
        path, name = weights_name.rsplit(".", 1)
        weights_name = f"{path}.{variant}.{name}"
    return weights_name


def _get_resolved_checkpoint_files(
    pretrained_model_name_or_path: str | Path | None,
    options: ModelPretrainedOptions,
    is_remote_code: bool,  # Because we can't determine this inside this function, we need it to be passed in
    transformers_explicit_filename: str | None = None,
) -> tuple[list[Path] | None, dict | None]:
    if options.variant is not None:
        raise YetToImplement("using variant is not supprted.")
    variant = options.variant

    if transformers_explicit_filename is not None:
        raise YetToImplement("passing explicit filename is not supported.")

    is_sharded = False
    if (
        pretrained_model_name_or_path is not None
            and options.gguf_file is None
    ):
        if is_local := Path(pretrained_model_name_or_path).is_dir():
            raise YetToImplement("passing a local directory is not supported.")
        elif (Path(options.subfolder) / pretrained_model_name_or_path).is_file():
            archive_file = Path(pretrained_model_name_or_path)
            is_local = True
        else:
            if transformers_explicit_filename is not None:
                raise YetToImplement("passing explicit filename is not supported.")
            elif use_safetensors is not False:
                filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
            else:
                filename = _add_variant(WEIGHTS_NAME, variant)

            cached_file_kwargs = {
                **asdict(options),
                "_raise_exceptions_for_gated_repo": False,
                "_raise_exceptions_for_missing_entries": False,
            }
            
            try:
                resolved_archive_file = cached_file(
                    pretrained_model_name_or_path,
                    filename, 
                    **cached_file_kwargs,
                )
                if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True
                    elif options.use_safetensors:
                        raise YetToImplement("auto conversion safetensors is not supported.")
                    else:
                        filename = _add_variant(WEIGHTS_NAME, variant)
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            filename,
                            **cached_file_kwargs
                        )

                # Then try `.bin` files
                if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        _add_variant(WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True

                # If we have a match, but it's `.bin` format, try to launch safetensors conversion for next time
                if resolved_archive_file is not None:
                    safe_weights_name = (
                        SAFE_WEIGHTS_INDEX_NAME
                        if is_sharded
                        else SAFE_WEIGHTS_NAME
                    )
                    if (
                        filename in [WEIGHTS_NAME, WEIGHTS_INDEX_NAME]
                    ):
                        if resolved_archive_file.endswith(".safetensors"):
                            raise YetToImplement("auto conversion safetensors is not supported.")
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

    elif options.gguf_file:
        raise YetToImplement("gguf file is not supported.")

    # We now download and resolve all checkpoint files if the checkpoint is sharded
    sharded_metadata = None
    if is_sharded:
        raise YetToImplement("shared file weights is not supported.")
        checkpoint_files, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_archive_file,
            **asdict(options)
        )
    else:
        checkpoint_files = (
            [resolved_archive_file]
            if pretrained_model_name_or_path is not None
            else None
        )

    return checkpoint_files, sharded_metadata
