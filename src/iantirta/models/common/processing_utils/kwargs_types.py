
from typing import Annotated, Literal, TypedDict, TypeVar, Union

from iantirta.models.tools.kwargs_types import Unpack


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

    _defaults = {}

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
