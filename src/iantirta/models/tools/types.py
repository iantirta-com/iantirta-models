from enum import Enum
from os import PathLike
from typing import Any, Protocol, TypeAlias

DeviceMeshLike: TypeAlias = Any  # PyTorch stubs do not model torch.distributed.device_mesh consistently yet.


class StringValuedEnumLike(Protocol):
    value: str


class PeftConfigLike(Protocol):
    peft_type: StringValuedEnumLike
    is_prompt_learning: bool
    base_model_name_or_path: str | PathLike[str] | None
    inference_mode: bool

    def save_pretrained(self, save_directory: str | PathLike[str], **kwargs: Any) -> None: ...


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
