import functools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TypedDict, TypeVar, Union

import numpy as np

from ...tools._deps import direct_iantirta_models_import
from ...tools._torch import is_torch_available
from ...tools.type_validators import (
    device_validator,
    image_size_validator,
    padding_validator,
    positive_any_number,
    positive_int,
    resampling_validator,
    tensor_type_validator,
    truncation_validator,
    video_metadata_validator,
)
from ...tools.types import TensorType
from ..tokenizer.base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from .image.utils import (
    ChannelDimension,
    is_vision_available,
)
from .video.utils import VideoMetadataType

if is_torch_available():
    import torch

if is_vision_available():
    from .image.utils import PILImageResampling

if TYPE_CHECKING:
    from .mixin import ProcessorMixin


PROCESSOR_NAME = "processor_config.json"
AUDIO_TOKENIZER_NAME = "audio_tokenizer_config.json"


# type hinting: specifying the type of processor class that inherits from ProcessorMixin
SpecificProcessorType = TypeVar("SpecificProcessorType", bound="ProcessorMixin")

# Dynamically import the module to grab the attribute classes of the processor from their names.
iantirta_models_module = direct_iantirta_models_import(Path(__file__).parent.parent.parent)


class _LazyAutoProcessorMapping(dict):
    """
    Lazy dictionary to avoid circular imports.
    The mapping names are only imported when accessed.
    """

    _MAPPING_NAMES = {  # noqa: RUF012
        "image_processor": ("iantirta.models.core.auto.image_processing_auto", "AutoImageProcessor"),
        "video_processor": ("iantirta.models.core.auto.video_processing_auto", "AutoVideoProcessor"),
        "feature_extractor": ("iantirta.models.core.auto.feature_extraction_auto", "AutoFeatureExtractor"),
        "audio_processor": ("iantirta.models.core.auto.feature_extraction_auto", "AutoFeatureExtractor"),
        "tokenizer": ("iantirta.models.core.auto.tokenization_auto", "AutoTokenizer"),
    }

    def __getitem__(self, key):
        if key not in self._MAPPING_NAMES:
            raise KeyError(key)
        module_name, attr_name = self._MAPPING_NAMES[key]
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)

    def __contains__(self, key):
        return key in self._MAPPING_NAMES

    def keys(self):
        return self._MAPPING_NAMES.keys()


MODALITY_TO_AUTOPROCESSOR_MAPPING = _LazyAutoProcessorMapping()

MODALITY_TO_BASE_CLASS_MAPPING = {
    "audio_tokenizer": (
        "HiggsAudioV2TokenizerModel",
        "DacModel",
    ),  # TODO: @eustlb, to be replaced with PreTrainedAudioTokenizerBase
    "audio_processor": "FeatureExtractionMixin",
    "tokenizer": ("PreTrainedTokenizerBase", "MistralCommonBackend"),
    "feature_extractor": "FeatureExtractionMixin",
    "image_processor": "ImageProcessingMixin",
    "video_processor": "BaseVideoProcessor",
}


def _get_modality_for_attribute(attribute_name: str) -> str:
    """
    Get the canonical modality type for a given attribute name.

    For example:
    - "image_processor" -> "image_processor"
    - "encoder_image_processor" -> "image_processor"
    - "text_tokenizer" -> "tokenizer"
    - "my_feature_extractor" -> "feature_extractor"
    """
    for modality in MODALITY_TO_AUTOPROCESSOR_MAPPING:
        if modality in attribute_name:
            return modality
    raise ValueError(
        f"Cannot determine modality for attribute '{attribute_name}'. "
        f"Attribute name must contain one of: {list(MODALITY_TO_AUTOPROCESSOR_MAPPING.keys())}"
    )


class TextKwargs(TypedDict, total=False):
    """
    Keyword arguments for text processing. For extended documentation, check out tokenization_utils_base methods and
    docstrings associated.

    Attributes:
        add_special_tokens (`bool`, *optional*)
            Whether or not to add special tokens when encoding the sequences.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*)
            Activates and controls padding.
        truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*):
            Activates and controls truncation.
        max_length (`int`, *optional*):
            Controls the maximum length to use by one of the truncation/padding parameters.
        stride (`int`, *optional*):
            If set, the overflowing tokens will contain some tokens from the end of the truncated sequence.
        is_split_into_words (`bool`, *optional*):
            Whether or not the input is already pre-tokenized.
        pad_to_multiple_of (`int`, *optional*):
            If set, will pad the sequence to a multiple of the provided value.
        return_token_type_ids (`bool`, *optional*):
            Whether to return token type IDs.
        return_attention_mask (`bool`, *optional*):
            Whether to return the attention mask.
        return_overflowing_tokens (`bool`, *optional*):
            Whether or not to return overflowing token sequences.
        return_special_tokens_mask (`bool`, *optional*):
            Whether or not to return special tokens mask information.
        return_offsets_mapping (`bool`, *optional*):
            Whether or not to return `(char_start, char_end)` for each token.
        return_length (`bool`, *optional*):
            Whether or not to return the lengths of the encoded inputs.
        verbose (`bool`, *optional*):
            Whether or not to print more information and warnings.
        padding_side (`str`, *optional*):
            The side on which padding will be applied.
        return_mm_token_type_ids (`bool`, *optional*):
            Whether to return multimodal token type ids indicating mm placeholder token positions.
        return_text_replacement_offsets (`bool`, *optional*):
            Whether to return character offsets for each mm placeholder and its replacement.
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
            If set, will return tensors of a particular framework. Acceptable values are:
            - `'pt'`: Return PyTorch `torch.Tensor` objects.
            - `'np'`: Return NumPy `np.ndarray` objects.
    """

    text_pair: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None
    text_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None
    text_pair_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None
    add_special_tokens: bool | None
    padding: Annotated[bool | str | PaddingStrategy | None, padding_validator()]
    truncation: Annotated[bool | str | TruncationStrategy | None, truncation_validator()]
    max_length: Annotated[int | None, positive_int()]
    stride: Annotated[int | None, positive_int()]
    is_split_into_words: bool | None
    pad_to_multiple_of: Annotated[int | None, positive_int()]
    return_token_type_ids: bool | None
    return_attention_mask: bool | None
    return_overflowing_tokens: bool | None
    return_special_tokens_mask: bool | None
    return_offsets_mapping: bool | None
    return_length: bool | None
    verbose: bool | None
    padding_side: Literal["left", "right"] | None
    return_mm_token_type_ids: bool | None
    return_text_replacement_offsets: bool | None
    return_tensors: Annotated[str | TensorType | None, tensor_type_validator()]


class ImagesKwargs(TypedDict, total=False):
    """
    Keyword arguments for image processing. For extended documentation, check the appropriate ImageProcessor
    class methods and docstrings.

    Attributes:
        do_convert_rgb (`bool`):
            Whether to convert the image to RGB format.
        do_resize (`bool`, *optional*):
            Whether to resize the image.
        size (`dict[str, int]`, *optional*):
            Resize using one of the supported size dictionaries. Pixel-area bounds use
            `{"min_pixels": int, "max_pixels": int}`.
        default_to_square (`bool`, *optional*, defaults to `self.default_to_square`):
            Whether to default to a square when resizing, if size is an int.
        crop_size (`dict[str, int]`, *optional*):
            Desired output size when applying center-cropping.
        resample (`PILImageResampling`, *optional*):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*):
            Whether to normalize the image.
        image_mean (`float` or `list[float] or tuple[float, float, float]`, *optional*):
            Mean to use if normalizing the image.
        image_std (`float` or `list[float] or tuple[float, float, float]`, *optional*):
            Standard deviation to use if normalizing the image.
        do_pad (`bool`, *optional*):
            Whether to pad the images in the batch.
        pad_size (`dict[str, int]`, *optional*):
            The size `{"height": int, "width" int}` to pad the images to.
        do_center_crop (`bool`, *optional*):
            Whether to center crop the image.
        data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the output image.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image.
        device (`Union[str, torch.Tensor]`, *optional*):
            The device to use for processing (e.g. "cpu", "cuda"), only relevant for torchvision backend.
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
            If set, will return tensors of a particular framework. Acceptable values are:
            - `'pt'`: Return PyTorch `torch.Tensor` objects.
            - `'np'`: Return NumPy `np.ndarray` objects.
        disable_grouping (`bool`, *optional*):
            Whether to group images by shapes when processing or not, only relevant for torchvision backend.
        image_seq_length (`int`, *optional*):
            The number of image tokens to be used for each image in the input.
            Added for backward compatibility but this should be set as a processor attribute in future models.
    """

    do_convert_rgb: bool | None
    do_resize: bool | None
    size: Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, image_size_validator()]
    default_to_square: bool | None
    crop_size: Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, image_size_validator()]
    resample: Annotated[Union["PILImageResampling", int] | None, resampling_validator()]
    do_rescale: bool | None
    rescale_factor: float | None
    do_normalize: bool | None
    image_mean: float | list[float] | tuple[float, ...] | None
    image_std: float | list[float] | tuple[float, ...] | None
    do_pad: bool | None
    pad_size: Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, image_size_validator()]
    do_center_crop: bool | None
    data_format: str | ChannelDimension | None
    input_data_format: str | ChannelDimension | None
    device: Annotated[Union[str, "torch.device"] | None, device_validator()]
    return_tensors: Annotated[str | TensorType | None, tensor_type_validator()]
    disable_grouping: bool | None
    image_seq_length: int | None


class VideosKwargs(TypedDict, total=False):
    """
    Keyword arguments for video processing.

    Attributes:
        do_convert_rgb (`bool`):
            Whether to convert the video to RGB format.
        do_resize (`bool`):
            Whether to resize the video.
        size (`dict[str, int]`, *optional*):
            Resize the shorter side of the input to `size["shortest_edge"]`.
        default_to_square (`bool`, *optional*, defaults to `self.default_to_square`):
            Whether to default to a square when resizing, if size is an int.
        resample (`PILImageResampling`, *optional*):
            Resampling filter to use if resizing the video.
        do_rescale (`bool`, *optional*):
            Whether to rescale the video by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*):
            Scale factor to use if rescaling the video.
        do_normalize (`bool`, *optional*):
            Whether to normalize the video.
        image_mean (`float` or `list[float] or tuple[float, float, float]`, *optional*):
            Mean to use if normalizing the video.
        image_std (`float` or `list[float] or tuple[float, float, float]`, *optional*):
            Standard deviation to use if normalizing the video.
        do_center_crop (`bool`, *optional*):
            Whether to center crop the video.
        do_pad (`bool`, *optional*):
            Whether to pad the images in the batch.
        do_sample_frames (`bool`, *optional*):
            Whether to sample frames from the video before processing or to process the whole video.
        video_metadata (`Union[VideoMetadata, dict]`, *optional*):
            Metadata of the video containing information about total duration, fps and total number of frames.
        num_frames (`int`, *optional*):
            Maximum number of frames to sample when `do_sample_frames=True`.
        fps (`int` or `float`, *optional*):
            Target frames to sample per second when `do_sample_frames=True`.
        crop_size (`dict[str, int]`, *optional*):
            Desired output size when applying center-cropping.
        data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the output video.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input video.
        device (`Union[str, torch.Tensor]`, *optional*):
            The device to use for processing (e.g. "cpu", "cuda"), only relevant for fast image processing.
        return_metadata (`bool`, *optional*):
            Whether to return video metadata or not.
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
            If set, will return tensors of a particular framework. Acceptable values are:
            - `'pt'`: Return PyTorch `torch.Tensor` objects.
            - `'np'`: Return NumPy `np.ndarray` objects.
    """

    do_convert_rgb: bool | None
    do_resize: bool | None
    size: Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, image_size_validator()]
    default_to_square: bool | None
    resample: Annotated[Union["PILImageResampling", int] | None, resampling_validator()]
    do_rescale: bool | None
    rescale_factor: float | None
    do_normalize: bool | None
    image_mean: float | list[float] | tuple[float, ...] | None
    image_std: float | list[float] | tuple[float, ...] | None
    do_center_crop: bool | None
    do_pad: bool | None
    crop_size: Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, image_size_validator()]
    data_format: str | ChannelDimension | None
    input_data_format: str | ChannelDimension | None
    device: Annotated[Union[str, "torch.device"] | None, device_validator()]
    do_sample_frames: bool | None
    video_metadata: Annotated[VideoMetadataType | None, video_metadata_validator()]
    fps: Annotated[int | float | None, positive_any_number()]
    num_frames: Annotated[int | None, positive_int()]
    return_metadata: bool | None
    return_tensors: Annotated[str | TensorType | None, tensor_type_validator()]


class AudioKwargs(TypedDict, total=False):
    """
    Keyword arguments for audio processing.

    Attributes:
        sampling_rate (`int`, *optional*):
            The sampling rate at which the `raw_speech` input was sampled.
        raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
            The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
            values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
            stereo, i.e. single float per timestep.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding
            index) among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        truncation (`bool`, *optional*):
            Activates truncation to cut input sequences longer than *max_length* to *max_length*.
        pad_to_multiple_of (`int`, *optional*):
            If set, will pad the sequence to a multiple of the provided value.
        return_attention_mask (`bool`, *optional*):
            Whether or not [`~ASTFeatureExtractor.__call__`] should return `attention_mask`.
        device (`str` or `torch.device`, *optional*):
            The device to compute the audio features on (e.g. "cpu", "cuda"), only relevant for feature
            extractors that compute them with torch.
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
            If set, will return tensors of a particular framework. Acceptable values are:
            - `'pt'`: Return PyTorch `torch.Tensor` objects.
            - `'np'`: Return NumPy `np.ndarray` objects.
        load_audio_backend (`str`, *optional*):
            Backend used by [`~audio_utils.load_audio`] to decode/resample audio referenced by URL/path
            in `apply_chat_template`. One of `"auto"`, `"torchcodec"`, `"librosa"`, `"torchaudio"`.
    """

    sampling_rate: Annotated[int | None, positive_int()]
    raw_speech: Union["np.ndarray", list[float], list["np.ndarray"], list[list[float]]] | None
    padding: Annotated[bool | str | PaddingStrategy | None, padding_validator()]
    max_length: Annotated[int | None, positive_int()]
    truncation: Annotated[bool | str | TruncationStrategy | None, truncation_validator()]
    pad_to_multiple_of: Annotated[int | None, positive_int()]
    return_attention_mask: bool | None
    device: Annotated[Union[str, "torch.device"] | None, device_validator()]
    return_tensors: Annotated[str | TensorType | None, tensor_type_validator()]
    load_audio_backend: str | None


class ProcessingKwargs(TypedDict, total=False):
    """
    Base class for kwargs passing to processors.
    In case a model has specific kwargs that are not present in the base class or default values for existing keys,
    it should have its own `ModelProcessorKwargs` class that inherits from `ProcessingKwargs` to provide:
        1) Additional typed keys and that this model requires to process inputs.
        2) Default values for existing keys under a `_defaults` attribute.
    New keys have to be defined as follows to ensure type hinting is done correctly.

    ```python
    # adding a new image kwarg for this model
    class ModelImagesKwargs(ImagesKwargs, total=False):
        new_image_kwarg: Optional[bool]

    class ModelProcessorKwargs(ProcessingKwargs, total=False):
        images_kwargs: ModelImagesKwargs
        _defaults = {
            "images_kwargs: {
                "new_image_kwarg": False,
            }
            "text_kwargs": {
                "padding": "max_length",
            },
        }

    ```

    For Python 3.8 compatibility, when inheriting from this class and overriding one of the kwargs,
    you need to manually update the __annotations__ dictionary. This can be done as follows:

    ```python
    class CustomProcessorKwargs(ProcessingKwargs, total=False):
        images_kwargs: CustomImagesKwargs

    CustomProcessorKwargs.__annotations__["images_kwargs"] = CustomImagesKwargs  # python 3.8 compatibility
    ```

    """

    _defaults = {}  # noqa: RUF012

    text_kwargs: TextKwargs = {  # noqa: RUF012
        **TextKwargs.__annotations__,
    }
    images_kwargs: ImagesKwargs = {  # noqa: RUF012
        **ImagesKwargs.__annotations__,
    }
    videos_kwargs: VideosKwargs = {  # noqa: RUF012
        **VideosKwargs.__annotations__,
    }
    audio_kwargs: AudioKwargs = {  # noqa: RUF012
        **AudioKwargs.__annotations__,
    }


class TokenizerChatTemplateKwargs(TypedDict, total=False):
    """
    NOTE: `TokenizerChatTemplateKwargs` is deprecated and will be removed in future versions
    Keyword arguments for tokenizer's `apply_chat_template`, when it is called from within a processor.

    tools (`list[Dict]`, *optional*):
        A list of tools (callable functions) that will be accessible to the model. If the template does not
        support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
        giving the name, description and argument types for the tool. See our
        [chat templating guide](https://huggingface.co/docs/iantirta.models/main/en/chat_templating#automated-function-conversion-for-tool-use)
        for more information.
    documents (`list[dict[str, str]]`, *optional*):
        A list of dicts representing documents that will be accessible to the model if it is performing RAG
        (retrieval-augmented generation). If the template does not support RAG, this argument will have no
        effect. We recommend that each document should be a dict containing "title" and "text" keys. Please
        see the RAG section of the [chat templating guide](https://huggingface.co/docs/iantirta.models/main/en/chat_templating#arguments-for-RAG)
        for examples of passing documents with chat templates.
    add_generation_prompt (bool, *optional*):
        If this is set, a prompt with the token(s) that indicate
        the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model.
        Note that this argument will be passed to the chat template, and so it must be supported in the
        template for this argument to have any effect.
    continue_final_message (bool or str, *optional*):
        If this is set, the chat will be formatted so that the final
        message in the chat is open-ended, without any EOS tokens. The model will continue this message
        rather than starting a new one. This allows you to "prefill" part of
        the model's response for it. If a string is passed, it will be used as the key for the field to continue
        (e.g. "reasoning_content"). Cannot be used at the same time as `add_generation_prompt`.

    return_assistant_tokens_mask (`bool`, defaults to `False`):
        Whether to return a mask of the assistant generated tokens. For tokens generated by the assistant,
        the mask will contain 1. For user and system tokens, the mask will contain 0.
        This functionality is only available for chat templates that support it via the `{% generation %}` keyword.
    reasoning_effort (`str`, *optional*):
        The reasoning effort level to use for the model's response. Supported values depend on the model
        (e.g. `"none"`, "low"`, `"medium"`, `"high"`). If the template does not support reasoning effort,
        this argument will have no effect.
    """

    tools: list[dict] | None = None
    documents: list[dict[str, str]] | None = None
    add_generation_prompt: bool | None = False
    continue_final_message: bool | str | None = False
    return_assistant_tokens_mask: bool | None = False
    reasoning_effort: str | None = None


class ProcessorChatTemplateKwargs(TokenizerChatTemplateKwargs, total=False):
    """
    NOTE: `ProcessorChatTemplateKwargs` is deprecated and will be removed in future versions

    Keyword arguments for processor's `apply_chat_template`.

    tokenize (`bool`, *optional*, defaults to `False`):
        Whether to tokenize the output or not.
    return_dict (`bool`, defaults to `False`):
        Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
    load_audio_from_video (`bool`, *optional*, defaults to `False`):
        Whether to use the audio track of input video. If `True` the audio track will be loaded and passed to the
        processor. This flag has no effect if the model doesn't support audio modality.
    """

    tokenize: bool | None = False
    return_dict: bool | None = False
    load_audio_from_video: bool | None = False


class AllKwargsForChatTemplate(TypedDict, total=False):
    "NOTE: `AllKwargsForChatTemplate` is deprecated and will be removed in future versions"

    processor_kwargs: ProcessingKwargs
    template_kwargs: ProcessorChatTemplateKwargs


@dataclass
class MultiModalData:
    """
    Dataclass that holds extra useful data for processing
    multimodal data. Processors currently cannot return keys,
    unless it is used in model's forward. Thus we have helper
    methods that calculate and return useful data from processing
    input multimodals (images/videos).
    Note that this dataclass is aimed to be used only in vLLM
    and we might change its API in the future.
    """

    num_image_tokens: list[int] | None = None
    num_video_tokens: list[int] | None = None
    num_audio_tokens: list[int] | None = None
    num_image_patches: list[int] | None = None

    def __contains__(self, key):
        return hasattr(self, key) and getattr(self, key) is not None

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")


@functools.lru_cache(maxsize=8)
def _merge_typed_dict(preprocessor_typed_dict: type, modality_typed_dict: type) -> type:
    return TypedDict(
        "merged_typed_dict",
        {**preprocessor_typed_dict.__annotations__, **modality_typed_dict.__annotations__},
        total=False,
    )


def prepare_prompt_input(
    inputs: str | list[str] | None,
    batch_size: int,
    input_name: str = "inputs",
) -> list[str | None]:
    """
    Normalize a string, list of strings, or ``None`` into a list of length ``batch_size``.

    Args:
        inputs (`str`, `list[str]`, or `None`):
            The input to normalize. A single string is broadcast to all batch items; a list must
            match ``batch_size`` exactly; ``None`` produces a list of ``None`` values.
        batch_size (`int`):
            Expected length of the output list.
        input_name (`str`, *optional*, defaults to `"inputs"`):
            Name used in error messages to identify the argument.

    Returns:
        `list[str | None]`: A list of length ``batch_size``.
    """
    if inputs is None:
        return [None] * batch_size
    if isinstance(inputs, str):
        return [inputs] * batch_size
    if isinstance(inputs, (list, tuple)):
        if len(inputs) != batch_size:
            raise ValueError(
                f"Received {len(inputs)} {input_name} for {batch_size} audio sample(s); counts must match."
            )
        return list(inputs)
    raise TypeError(f"`{input_name}` must be a string, a sequence of strings, or `None`.")
