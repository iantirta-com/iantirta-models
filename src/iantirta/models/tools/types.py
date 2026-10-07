from __future__ import annotations

from enum import Enum
from os import PathLike
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

DeviceMeshLike: TypeAlias = Any  # PyTorch stubs do not model torch.distributed.device_mesh consistently yet.

if TYPE_CHECKING:
    import torch
class GenerativePreTrainedModel(Protocol):
    """Protocol for the model interface that GenerationMixin expects.

    GenerationMixin is designed to be mixed into PreTrainedModel subclasses. This Protocol documents the
    attributes and methods the mixin relies on from its host class. It is *not* used at runtime — its
    purpose is to help the ``ty`` type checker resolve ``self.<attr>`` accesses inside the mixin.
    """

    config: Any  # PretrainedConfig — kept as Any to avoid circular imports
    device: torch.device
    dtype: torch.dtype
    main_input_name: str
    base_model_prefix: str
    _is_stateful: bool
    hf_quantizer: Any
    encoder: Any
    hf_device_map: dict[str, Any]
    _previous_max_cache_length: int

    generation_config: Any  # GenerationConfig

    def __getattr__(self, name: str) -> Any: ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def can_generate(self) -> bool: ...
    def get_encoder(self) -> Any: ...
    def get_output_embeddings(self) -> Any: ...
    def get_input_embeddings(self) -> Any: ...
    def set_output_embeddings(self, value: Any) -> None: ...
    def set_input_embeddings(self, value: Any) -> None: ...
    def get_compiled_call(self, compile_config: Any) -> Any: ...
    def set_experts_implementation(self, *args: Any, **kwargs: Any) -> Any: ...
    def _supports_logits_to_keep(self) -> bool: ...


class StringValuedEnumLike(Protocol):
    value: str


class PeftConfigLike(Protocol):
    peft_type: StringValuedEnumLike
    is_prompt_learning: bool
    base_model_name_or_path: str | PathLike[str] | None
    inference_mode: bool

    def save_pretrained(self, save_directory: str | PathLike[str], **kwargs: Any) -> None: ...


class WhisperGenerationConfigLike(Protocol):
    """Protocol for Whisper-specific generation config fields accessed in generation internals."""

    no_timestamps_token_id: int


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the `padding` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in an
    IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TensorType(ExplicitEnum):
    """
    Possible values for the `return_tensors` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for
    tab-completion in an IDE.
    """

    PYTORCH = "pt"
    NUMPY = "np"
    MLX = "mlx"
