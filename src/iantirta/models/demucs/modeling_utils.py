import os
from typing import Any
from iantirta.models.vendor.transformers.modeling_utils import (
    PreTrainedModel, SpecificPreTrainedModelType,
)
from iantirta.models.vendor.transformers.configuration_utils import (
    PreTrainedConfig
)
import yaml
from huggingface_hub.dataclasses import strict

DEFAULT_NAMESPACE = "adefossez"


def hf_repo_name(name: str) -> str:
    """Map a demucs model name to its HuggingFace repository name,
    e.g. `htdemucs_ft` -> `HTDemucs-ft`, `mdx_extra_q` -> `Demucs-mdx_extra_q`."""
    if name == 'htdemucs':
        return 'HTDemucs'
    elif name.startswith('htdemucs_'):
        return 'HTDemucs-' + name[len('htdemucs_'):]
    else:
        return 'Demucs-' + name


@strict
class DemucsConfig(PreTrainedConfig):
    pass

class DemucsPreTrainedModel(PreTrainedModel):
    config: DemucsConfig
    
    # @classmethod
    # def from_pretrained(
    #     cls: type[SpecificPreTrainedModelType],
    #     pretrained_model_name_or_path: str | os.PathLike | None,
    #     *model_args,
    #     config: PreTrainedConfig | str | os.PathLike | None = None,
    #     cache_dir: str | os.PathLike | None = None,
    #     ignore_mismatched_sizes: bool = False,
    #     force_download: bool = False,
    #     local_files_only: bool = False,
    #     token: str | bool | None = None,
    #     revision: str = "main",
    #     use_safetensors: bool | None = None,
    #     weights_only: bool = True,
    #     fusion_config: dict[str, bool | dict[str, Any]] | None = None,
    #     disable_mmap: bool | None = None,
    #     **kwargs,
    # ) -> SpecificPreTrainedModelType:
    #     namespace = DEFAULT_NAMESPACE
    #     if "/" in pretrained_model_name_or_path:
    #         namespace, name = pretrained_model_name_or_path.split("/", 1)
    #     pretrained_model_name_or_path = f"{namespace}/{hf_repo_name(name)}"
    #     return super().from_pretrained(
    #         pretrained_model_name_or_path,
    #     )

    @classmethod
    def _dict_from_json_file(cls, json_file: str | os.PathLike):
        if json_file.endswith(".json"):
            return super()._dict_from_json_file(json_file)
        else:
            try:
                with open(json_file) as file:
                    return yaml.safe_load(file)
            except:
                raise
