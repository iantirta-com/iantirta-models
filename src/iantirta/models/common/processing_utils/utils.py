

class _LazyAutoProcessorMapping(dict):
    """
    Lazy dictionary to avoid circular imports.
    The mapping names are only imported when accessed.
    """

    _MAPPING_NAMES = {  # noqa: RUF012
        "image_processor": ("iantirta.models.common.auto.image_processing_auto", "AutoImageProcessor"),
        "video_processor": ("iantirta.models.common.auto.video_processing_auto", "AutoVideoProcessor"),
        "feature_extractor": ("iantirta.models.common.auto.feature_extraction_auto", "AutoFeatureExtractor"),
        "audio_processor": ("iantirta.models.common.auto.feature_extraction_auto", "AutoFeatureExtractor"),
        "tokenizer": ("iantirta.models.common.auto.tokenization_auto", "AutoTokenizer"),
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
