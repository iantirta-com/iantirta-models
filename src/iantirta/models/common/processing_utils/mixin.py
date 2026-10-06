

import bisect
import copy
import functools
import inspect
import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from typing_extensions import Self

from iantirta.models.remote.files import cached_file
from iantirta.models.tools._torch import is_torch_available
from iantirta.models.tools._vision import is_vision_available
from iantirta.models.tools.types import TensorType
from iantirta.models.vendor.huggingface_hub.dataclasses import validate_typed_dict

from ..feature_extraction_utils import BatchFeature
from ..input_utils.audio_utils import load_audio, make_list_of_audio
from ..input_utils.image_utils import (
    ChannelDimension,
    ImageInput,
    make_flat_list_of_images,
)
from ..input_utils.video_utils import VideoInput, make_batched_videos
from ..tokenization_utils import (
    CHAT_TEMPLATE_DIR,
    CHAT_TEMPLATE_FILE,
    AudioInput,
    PaddingStrategy,
    PreTokenizedInput,
    PreTrainedTokenizerBase,
    TextInput,
    TruncationStrategy,
)
from ..tokenization_utils.chat_utils.template import (
    _get_template_variables,
    render_jinja_template,
)
from ..tokenization_utils.pretrained import PreTrainedAudioTokenizerBase
from .kwargs_types import ProcessingKwargs, Unpack
from .utils import MODALITY_TO_AUTOPROCESSOR_MAPPING, MODALITY_TO_BASE_CLASS_MAPPING

logger = logging.getLogger(__name__)


AUDIO_TOKENIZER_NAME = "audio_tokenizer_config.json"
PROCESSOR_NAME = "processor_config.json"
LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE = "chat_template.json"


@functools.lru_cache(maxsize=8)
def _merge_typed_dict(preprocessor_typed_dict: type, modality_typed_dict: type) -> type:
    return TypedDict(
        "merged_typed_dict",
        {**preprocessor_typed_dict.__annotations__, **modality_typed_dict.__annotations__},
        total=False,
    )


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


class ProcessorMixin:
    """
    This is a mixin used to provide saving/loading functionality for all processor classes.
    """

    # Dynamically set sub-processor attributes. Not every processor has all of these;
    # they are populated via setattr in __init__ based on each subclass's `attributes`.
    tokenizer: Any
    feature_extractor: Any
    image_processor: Any
    video_processor: Any
    chat_template: str | dict[str, str] | None

    # Names need to be attr_class for attr in attributes
    _auto_class = None
    valid_processor_kwargs = ProcessingKwargs
    skip_tensor_conversion = ["video_metadata", "text_replacement_offsets"]  # noqa: RUF012

    # args have to match the attributes class attribute
    def __init__(self, *args, **kwargs):
        # First, extract chat template from kwargs. It can never be a positional arg
        self.chat_template = kwargs.pop("chat_template", None)

        # Check audio tokenizer for its class but do not treat it as attr to avoid saving weights
        if (audio_tokenizer := kwargs.pop("audio_tokenizer", None)) is not None:
            proper_class = self.check_argument_for_proper_class("audio_tokenizer", audio_tokenizer)
            if not (is_torch_available() and isinstance(audio_tokenizer, PreTrainedAudioTokenizerBase)):
                raise ValueError(
                    f"Tried to use `{proper_class}` for audio tokenization. However, this class is not"
                    " registered for audio tokenization."
                )
            self.audio_tokenizer = audio_tokenizer

        # Sanitize args and kwargs
        for key in kwargs:
            if key not in self.get_attributes():
                raise TypeError(f"Unexpected keyword argument {key}.")
        for arg, attribute_name in zip(args, self.get_attributes()):
            if attribute_name in kwargs:
                raise TypeError(f"Got multiple values for argument {attribute_name}.")
            else:
                kwargs[attribute_name] = arg

        if len(kwargs) != len(self.get_attributes()):
            raise ValueError(
                f"This processor requires {len(self.get_attributes())} arguments: {', '.join(self.get_attributes())}. Got "
                f"{len(args)} arguments instead."
            )

        # Check each arg is of the proper class (this will also catch a user initializing in the wrong order)
        for attribute_name, arg in kwargs.items():
            self.check_argument_for_proper_class(attribute_name, arg)
            setattr(self, attribute_name, arg)

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos: VideoInput | None = None,
        audio: AudioInput | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        images, text, videos, audio = self.prepare_inputs_layout(
            images=images, text=text, videos=videos, audio=audio, **kwargs
        )
        self.validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)

        merged_kwargs = self._merge_kwargs(
            self.valid_processor_kwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs if hasattr(self, "tokenizer") else {},
            **kwargs,
        )

        processed_images = processed_videos = processed_audio = {}
        images_replacements = videos_replacements = audio_replacements = []
        if images is not None and hasattr(self, "image_processor"):
            processed_images, images_replacements = self._process_images(images, **merged_kwargs["images_kwargs"])
        if videos is not None and hasattr(self, "video_processor"):
            processed_videos, videos_replacements = self._process_videos(videos, **merged_kwargs["videos_kwargs"])
        if audio is not None and self._audio_processor is not None:
            processed_audio, audio_replacements = self._process_audio(audio, **merged_kwargs["audio_kwargs"])

        text_inputs = {}
        return_tensors = merged_kwargs["text_kwargs"].get("return_tensors", None)
        if getattr(self, "tokenizer", None) is not None and text is not None:
            return_mm_token_type_ids = merged_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
            return_text_replacement_offsets = merged_kwargs["text_kwargs"].pop(
                "return_text_replacement_offsets", False
            )

            text, text_replacement_offsets = self.get_text_with_replacements(
                text,
                images_replacements,
                videos_replacements,
                audio_replacements,
            )
            text_inputs = self.tokenizer(text, **merged_kwargs["text_kwargs"])
            self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video", "audio"])

            if return_text_replacement_offsets:
                text_inputs["text_replacement_offsets"] = text_replacement_offsets

            if return_mm_token_type_ids:
                text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])

        # Pop unused keys from the inputs, e.g. inputs used only to compute number of image tokens
        data = {**text_inputs, **processed_images, **processed_videos, **processed_audio}
        data = {k: v for k, v in data.items() if k not in self.unused_input_names}

        if not kwargs.get("return_metadata"):
            data.pop("video_metadata", None)

        return BatchFeature(data, tensor_type=return_tensors, skip_tensor_conversion=self.skip_tensor_conversion)

    def prepare_inputs_layout(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos: VideoInput | None = None,
        audio: AudioInput | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        """
        Normalize and prefetch inputs before processing. Wraps text in a list for multimodal
        processors, fetches remote images and audio if URLs are provided, and ensures audio
        is properly batched. Returns the normalized `(images, text, videos, audio)` tuple.
        """
        # To support BC with models in pre-MLLM era, don't wrap text in list
        if self.all_special_multimodal_tokens and text is not None:
            if isinstance(text, str):
                text = [text]
            # avoid in-place updates on text
            text = list(text).copy()

        if audio is not None and self._audio_processor is not None:
            sampling_rate = kwargs.get("sampling_rate", self._audio_processor.sampling_rate)
            audio = self._audio_processor.fetch_audio(audio, sampling_rate=sampling_rate)
            audio = make_list_of_audio(audio)

        if images is not None and hasattr(self, "image_processor"):
            images = self.image_processor.fetch_images(images)

        return images, text, videos, audio

    def validate_inputs(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos: VideoInput | None = None,
        audio: AudioInput | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        """
        Validate that at least one input is provided and that no deprecated keyword arguments
        are used. Raises ``ValueError`` otherwise.

        Override when the processor needs additional validation on the input args.
        """
        if "audios" in kwargs and audio is None:
            raise ValueError("You passed keyword argument `audios` which is deprecated. Please use `audio` instead.")

        if images is None and text is None and videos is None and audio is None:
            raise ValueError(f"You need to provide at least one input to call {self.__class__.__name__}")

    # Simple preprocessing includes calling the `subprocessor` and optionally
    # building placeholder strings. Each processor can override and add their
    # own special pre/post processing on top, e.g. see `audioflamingo`
    def _process_images(self, images: ImageInput, **kwargs):
        processed_images = self.image_processor(images, **kwargs)

        image_replacements = []
        if getattr(self, "image_token", None) is not None:
            # Some processors use nested struct, we need to flatten back if needed
            images = make_flat_list_of_images(images)
            for idx in range(len(images)):
                replacement_text = self.replace_image_token(processed_images, image_idx=idx, **kwargs)
                image_replacements.append(replacement_text)
        return processed_images, image_replacements

    def _process_videos(self, videos: VideoInput, **kwargs):
        processed_videos = self.video_processor(videos, **kwargs)

        video_replacements = []
        if getattr(self, "video_token", None) is not None:
            videos = make_batched_videos(videos)
            for idx in range(len(videos)):
                replacement_text = self.replace_video_token(processed_videos, video_idx=idx, **kwargs)
                video_replacements.append(replacement_text)

        return processed_videos, video_replacements

    @property
    def _audio_processor(self):
        # TODO: To be replaced with `audio_processor`
        return getattr(self, "audio_processor", getattr(self, "feature_extractor", None))

    def _process_audio(self, audio: AudioInput, **kwargs):
        processed_audio = self._audio_processor(audio, **kwargs)

        audio_replacements = []
        if getattr(self, "audio_token", None) is not None:
            for idx in range(len(audio)):
                replacement_text = self.replace_audio_token(processed_audio, audio_idx=idx, **kwargs)
                audio_replacements.append(replacement_text)

        return processed_audio, audio_replacements

    # To be overridden by each model's processor if they need to add placeholder tokens
    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        raise NotImplementedError

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        raise NotImplementedError

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        raise NotImplementedError

    def get_text_with_replacements(
        self,
        text: list[str],
        images_replacements: list[str] = [],  # noqa: B006
        videos_replacements: list[str] = [],  # noqa: B006
        audio_replacements: list[str] = [],  # noqa: B006
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """
        Replace multimodal placeholder tokens in a batch of text strings with their
        expanded representations, and return the modified texts alongside offset metadata.

        This method is the core text-side preprocessing step for multimodal inputs. It
        scans each text in the batch for special tokens (image, video, audio) and replaces
        them in-order with the pre-computed replacement strings produced by
        `self.replace_image_token` / `self.replace_video_token` / `self.replace_audio_token`.
        Replacements are consumed from each modality's list sequentially, so the i-th
        occurrence of e.g. ``self.image_token`` is replaced by ``images_replacements[i]``.

        To add a new multimodal processor with placeholder tokens, you need to define a correct
        `self.image_token` which is the same token that is embedded in input text and also used as
        placeholder and repeated many times. Then you need to override `self.replace_image_token`
        to return the correct replacement string for a given image at index `i`. Same goes for all
        other supported modalities.

        Args:
            text (`list[str]`):
                Batch of raw text strings, each potentially containing multimodal
                placeholder tokens. Note that it will be modified in-place and returned.
            images_replacements (`list[str]`, *optional*, defaults to `[]`):
                Expanded replacement strings for each image, in the order they appear
                across the batch. Produced by `self._process_images`.
            videos_replacements (`list[str]`, *optional*, defaults to `[]`):
                Expanded replacement strings for each video. Produced by
                `self._process_videos`.
            audio_replacements (`list[str]`, *optional*, defaults to `[]`):
                Expanded replacement strings for each audio input. Produced by
                `self._process_audio`.

        Returns:
            `tuple[list[str], list[dict[str, Any]]]`: A tuple of:
                - The modified `text` batch with all placeholder tokens expanded.
                - `batch_replacement_offsets`: one entry per batch item, each being a
                list of dicts with keys:
                    - `"type"` (`str`): modality name — `"image"`, `"video"`, or `"audio"`
                    - `"span"` (`tuple[int, int]`): original `(start, end)` char offsets of the placeholder token
                    - `"new_span"` (`tuple[int, int]`): `(start, end)` offsets of placeholder in the expanded string
                    - `"text"` (`str`): the original placeholder token string that was matched
                    - `"replacement"` (`str`): the string it was replaced with
        """
        # Early exit if no special tokens found, nothing to replace or if text is not a list
        if not self.all_special_multimodal_tokens:
            return text, []

        # Use named regex so we can extract groups later and replace
        # TODO @raushan: vllm encodes text and mm-data separately causing errors when a placeholder
        # has no associated mm-data. Thus we can check if there are any `replacements` and skip otherwise
        # Plan: update all models and contrib to vllm, they might benefit largely from `replacement_offsets`
        token_groups = []
        if len(images_replacements) > 0 and (image_token := getattr(self, "image_token", None)) is not None:
            token_groups.append(f"(?P<image>{re.escape(image_token)})")
        if len(videos_replacements) > 0 and (video_token := getattr(self, "video_token", None)) is not None:
            token_groups.append(f"(?P<video>{re.escape(video_token)})")
        if len(audio_replacements) > 0 and (audio_token := getattr(self, "audio_token", None)) is not None:
            token_groups.append(f"(?P<audio>{re.escape(audio_token)})")

        regex_special_mm_tokens = "|".join(token_groups) or r"(?!)"
        replacements_iters = {
            "image": iter(images_replacements),
            "video": iter(videos_replacements),
            "audio": iter(audio_replacements),
        }
        batch_replacement_offsets = []
        for batch_idx in range(len(text)):
            last = 0
            offset = 0
            replacement_offsets = []
            expanded_sample = []
            for m in re.finditer(regex_special_mm_tokens, text[batch_idx]):
                start, end = m.span()
                expanded_sample.append(text[batch_idx][last:start])

                # adjust spans using running offset if one sample has several MM data associated
                start_with_offset = start + offset

                mm_type = m.lastgroup
                replacement_text = next(replacements_iters[mm_type])
                replacement_offsets.append(
                    {
                        "type": mm_type,
                        "span": (start, end),
                        "new_span": (start_with_offset, start_with_offset + len(replacement_text)),
                        "text": m.group(),
                        "replacement": replacement_text,
                    }
                )
                expanded_sample.append(replacement_text)
                # update the offsets and the last position
                offset += len(replacement_text) - (end - start)
                last = end

            expanded_sample.append(text[batch_idx][last:])
            text[batch_idx] = "".join(expanded_sample)
            batch_replacement_offsets.append(replacement_offsets)
        return text, batch_replacement_offsets

    def create_mm_token_type_ids(self, input_ids: list) -> list[list[int]]:
        """
        Build per-token modality type IDs for a batch of token_id sequences.

        Each position is assigned an integer indicating which modality it belongs to:
        ``0`` for regular text, ``1`` for image tokens, ``2`` for video tokens, and
        ``3`` for audio tokens. Membership is determined by comparing against
        ``self.image_token_ids``, ``self.video_token_ids``, and ``self.audio_token_ids``.

        Args:
            input_ids (`list[list[int]]`):
                Batch of token ID sequences. May be unpadded (variable length), so
                a plain Python list of lists is expected rather than a tensor or
                uniformly-shaped array.

        Returns:
            `list[list[int]]`: A list of the same structure as ``input_ids``, where each
            integer is the modality type ID for the corresponding token.
        """
        mm_token_type_ids = []
        for tokenizer_input in input_ids:
            # Convert tensor rows to a list so `np.array` avoids NumPy 2.0's `__array__` copy-keyword deprecation.
            if not isinstance(tokenizer_input, list):
                tokenizer_input = tokenizer_input.tolist()
            tokenizer_input = np.array(tokenizer_input)
            mm_token_types = np.zeros_like(tokenizer_input)
            mm_token_types[np.isin(tokenizer_input, self.image_token_ids)] = 1
            mm_token_types[np.isin(tokenizer_input, self.video_token_ids)] = 2
            mm_token_types[np.isin(tokenizer_input, self.audio_token_ids)] = 3
            mm_token_type_ids.append(mm_token_types.tolist())
        return mm_token_type_ids

    @property
    def all_special_multimodal_tokens(self) -> list[str]:
        special_mm_tokens = [
            getattr(self, f"{modality}_token")
            for modality in ["image", "video", "audio"]
            if getattr(self, f"{modality}_token", None) is not None
        ]
        return special_mm_tokens

    # Special ids used per each modality in multimodal models. Models need to
    # override if they use special BOI/EOI/row/col/etc tokens that have to be marked
    # These values are used to build `mm_token_type_ids`
    @property
    def image_token_ids(self) -> list[int | None]:
        if _image_token_ids := getattr(self, "_image_token_ids", None):
            return _image_token_ids
        return [getattr(self, "image_token_id", None)]

    @image_token_ids.setter
    def image_token_ids(self, value: list[int | None]):
        self._image_token_ids = value

    @property
    def video_token_ids(self) -> list[int | None]:
        if _video_token_ids := getattr(self, "_video_token_ids", None):
            return _video_token_ids
        return [getattr(self, "video_token_id", None)]

    @video_token_ids.setter
    def video_token_ids(self, value: list[int | None]):
        self._video_token_ids = value

    @property
    def audio_token_ids(self) -> list[int | None]:
        if _audio_token_ids := getattr(self, "_audio_token_ids", None):
            return _audio_token_ids
        return [getattr(self, "audio_token_id", None)]

    @audio_token_ids.setter
    def audio_token_ids(self, value: list[int | None]):
        self._audio_token_ids = value

    def check_argument_for_proper_class(self, argument_name, argument):
        """
        Checks the passed argument's class against the expected transformers class. In case of an unexpected
        mismatch between expected and actual class, an error is raise. Otherwise, the proper retrieved class
        is returned.
        """
        # If the exact attribute name is not in the mapping, use its canonical modality
        # (e.g., "encoder_tokenizer" -> "tokenizer")
        if argument_name not in MODALITY_TO_BASE_CLASS_MAPPING:
            argument_name = _get_modality_for_attribute(argument_name)
        class_name = MODALITY_TO_BASE_CLASS_MAPPING.get(argument_name)
        if isinstance(class_name, tuple):
            proper_class = tuple(self.get_possibly_dynamic_module(n) for n in class_name if n is not None)
        else:
            proper_class = self.get_possibly_dynamic_module(class_name)

        if not isinstance(argument, proper_class):
            raise TypeError(
                f"Received a {type(argument).__name__} for argument {argument_name}, but a {class_name} was expected."
            )

        return proper_class

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this processor instance.
        """
        # Exclude tokenizer attributes before deepcopying to avoid copying large vocab/token structures.
        tokenizer_attributes = set()
        for attribute in self.__class__.get_attributes():
            if attribute in self.__dict__:
                modality = _get_modality_for_attribute(attribute)
                if modality == "tokenizer":
                    tokenizer_attributes.add(attribute)

        dict_to_copy = {k: v for k, v in self.__dict__.items() if k not in tokenizer_attributes}
        output = copy.deepcopy(dict_to_copy)

        # Get the kwargs in `__init__`.
        sig = inspect.signature(self.__init__)
        # Only save the attributes that are presented in the kwargs of `__init__`.
        # or in the attributes
        attrs_to_save = list(sig.parameters) + self.__class__.get_attributes()
        # extra attributes to be kept
        attrs_to_save += ["auto_map"]

        if "chat_template" in output:
            del output["chat_template"]

        def cast_array_to_list(dictionary):
            """
            Numpy arrays are not serialiazable but can be in pre-processing dicts.
            This function casts arrays to list, recusring through the nested configs as well.
            """
            for key, value in dictionary.items():
                if isinstance(value, np.ndarray):
                    dictionary[key] = value.tolist()
                elif isinstance(value, dict):
                    dictionary[key] = cast_array_to_list(value)
            return dictionary

        # Special case, add `audio_tokenizer` dict which points to model weights and path
        if "audio_tokenizer" in output:
            audio_tokenizer_dict = {
                "audio_tokenizer_class": self.audio_tokenizer.__class__.__name__,
                "audio_tokenizer_name_or_path": self.audio_tokenizer.name_or_path,
            }
            output["audio_tokenizer"] = audio_tokenizer_dict

        # Serialize attributes as a dict
        output = {
            k: v.to_dict() # if isinstance(v, PushToHubMixin) else v
            for k, v in output.items()
            if (
                k in attrs_to_save  # keep all attributes that have to be serialized
                and v.__class__.__name__ != "BeamSearchDecoderCTC"  # remove attributes with that are objects
            )
        }
        output = cast_array_to_list(output)
        output["processor_class"] = self.__class__.__name__

        return output

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
        """
        dictionary = self.to_dict()

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: str | os.PathLike):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this processor instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        attributes_repr = [f"- {name}: {getattr(self, name)!r}" for name in self.get_attributes()]
        attributes_repr = "\n".join(attributes_repr)
        return f"{self.__class__.__name__}:\n{attributes_repr}\n\n{self.to_json_string()}"

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        raise NotImplementedError()

    @classmethod
    def get_processor_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        processor of type [`~processing_utils.ProcessingMixin`] using `from_args_and_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.

        Returns:
            `tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the processor object.
        """
        # holding a copy for optionally loading the audio tokenizer (if available). It keeps the revision requested by
        # the user, as the audio tokenizer usually lives in another repository.
        audio_tokenizer_kwargs = copy.deepcopy(kwargs)

        cache_dir = kwargs.pop("cache_dir", None)
        _ = kwargs.pop("force_download", False)
        _ = kwargs.pop("proxies", None)
        _ = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")

        # Resolve the revision once, so that the template listing and all the files below come from the same repo state
        # revision = resolve_revision(
        #     pretrained_model_name_or_path,
        #     revision,
        #     token=token,
        #     local_files_only=local_files_only,
        #     cache_dir=cache_dir,
        # )

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "processor", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            processor_file = os.path.join(pretrained_model_name_or_path, PROCESSOR_NAME)

        additional_chat_template_files = {}
        resolved_additional_chat_template_files = {}
        if os.path.isfile(pretrained_model_name_or_path):
            resolved_processor_file = pretrained_model_name_or_path
            # can't load chat-template and audio tokenizer when given a file as pretrained_model_name_or_path
            resolved_chat_template_file = None
            resolved_raw_chat_template_file = None
            resolved_audio_tokenizer_file = None
            is_local = True
        else:
            if is_local:
                template_dir = Path(pretrained_model_name_or_path, CHAT_TEMPLATE_DIR)
                if template_dir.is_dir():
                    for template_file in template_dir.glob("*.jinja"):
                        template_name = template_file.stem
                        additional_chat_template_files[template_name] = f"{CHAT_TEMPLATE_DIR}/{template_file.name}"
            else:
                try:
                    for template in list_repo_templates(
                        pretrained_model_name_or_path,
                        local_files_only=local_files_only,
                        revision=revision,
                        cache_dir=cache_dir,
                        token=token,
                    ):
                        template = template.removesuffix(".jinja")
                        additional_chat_template_files[template] = f"{CHAT_TEMPLATE_DIR}/{template}.jinja"
                except EntryNotFoundError:
                    pass  # No template dir means no template files
            processor_file = PROCESSOR_NAME

            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_processor_file = cached_file(
                    pretrained_model_name_or_path,
                    processor_file,
                    cache_dir=cache_dir,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )

                # chat_template.json is a legacy file used by the processor class
                # a raw chat_template.jinja is preferred in future
                resolved_chat_template_file = cached_file(
                    pretrained_model_name_or_path,
                    LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE,
                    cache_dir=cache_dir,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )

                resolved_raw_chat_template_file = cached_file(
                    pretrained_model_name_or_path,
                    CHAT_TEMPLATE_FILE,
                    cache_dir=cache_dir,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )

                resolved_additional_chat_template_files = {
                    template_name: cached_file(
                        pretrained_model_name_or_path,
                        template_file,
                        cache_dir=cache_dir,
                        user_agent=user_agent,
                        revision=revision,
                        subfolder=subfolder,
                        _raise_exceptions_for_missing_entries=False,
                    )
                    for template_name, template_file in additional_chat_template_files.items()
                }

                resolved_audio_tokenizer_file = cached_file(
                    pretrained_model_name_or_path,
                    AUDIO_TOKENIZER_NAME,
                    cache_dir=cache_dir,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )
            except OSError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:  # noqa: BLE001
                # For any other exception, we throw a generic error.
                raise OSError(
                    f"Can't load processor for '{pretrained_model_name_or_path}'. If you were trying to load"
                    " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a {PROCESSOR_NAME} file"
                )

        # Add chat template as kwarg before returning because most models don't have processor config
        if resolved_chat_template_file is not None:
            # This is the legacy path
            with open(resolved_chat_template_file, encoding="utf-8") as reader:
                chat_template_json = json.loads(reader.read())
                chat_templates = {"default": chat_template_json["chat_template"]}
                if resolved_additional_chat_template_files:
                    raise ValueError(
                        "Cannot load chat template due to conflicting files - this checkpoint combines "
                        "a legacy chat_template.json file with separate template files, which is not "
                        "supported. To resolve this error, replace the legacy chat_template.json file "
                        "with a modern chat_template.jinja file."
                    )
        else:
            chat_templates = {
                template_name: open(template_file, "r", encoding="utf-8").read()  # noqa: SIM115
                for template_name, template_file in resolved_additional_chat_template_files.items()
            }
            if resolved_raw_chat_template_file is not None:
                with open(resolved_raw_chat_template_file, "r", encoding="utf-8") as reader:
                    chat_templates["default"] = reader.read()
        if isinstance(chat_templates, dict) and "default" in chat_templates and len(chat_templates) == 1:
            chat_templates = chat_templates["default"]  # Flatten when we just have a single template/file

        # Existing processors on the Hub created before #27761 being merged don't have `processor_config.json` (if not
        # updated afterward), and we need to keep `from_pretrained` work. So here it fallbacks to the empty dict.
        # (`cached_file` called using `_raise_exceptions_for_missing_entries=False` to avoid exception)
        # However, for models added in the future, we won't get the expected error if this file is missing.
        if resolved_processor_file is None:
            # In any case we need to pass `chat_template` if it is available
            processor_dict = {}
        else:
            try:
                # Load processor dict
                with open(resolved_processor_file, encoding="utf-8") as reader:
                    text = reader.read()
                processor_dict = json.loads(text)

            except json.JSONDecodeError:
                raise OSError(
                    f"It looks like the config file at '{resolved_processor_file}' is not a valid JSON file."
                )

        if is_local:
            logger.info(f"loading configuration file {resolved_processor_file}")
        else:
            logger.info(f"loading configuration file {processor_file} from cache at {resolved_processor_file}")

        if processor_dict.get("chat_template") is not None:
            logger.warning_once(
                "Chat templates should be in a 'chat_template.jinja' file but found key='chat_template' "
                "in the processor's config. Make sure to move your template to its own file."
            )
        elif chat_templates:
            processor_dict["chat_template"] = chat_templates

        # Audio tokenizer needs to load the model checkpoint first, because the saved
        # json file contains only references to the model path and repo id
        if resolved_audio_tokenizer_file is not None or "audio_tokenizer" in processor_dict:
            if resolved_audio_tokenizer_file is not None:
                reader = open(resolved_audio_tokenizer_file, "r", encoding="utf-8")  # noqa: SIM115
                audio_tokenizer_dict = reader.read()
                audio_tokenizer_dict = json.loads(audio_tokenizer_dict)
            else:
                audio_tokenizer_dict = processor_dict["audio_tokenizer"]

            audio_tokenizer_class = cls.get_possibly_dynamic_module(audio_tokenizer_dict["audio_tokenizer_class"])
            audio_tokenizer_path = audio_tokenizer_dict["audio_tokenizer_name_or_path"]
            processor_dict["audio_tokenizer"] = audio_tokenizer_class.from_pretrained(
                audio_tokenizer_path, **audio_tokenizer_kwargs
            )

        return processor_dict, kwargs

    @classmethod
    def from_args_and_dict(cls, args, processor_dict: dict[str, Any], **kwargs):
        """
        Instantiates a type of [`~processing_utils.ProcessingMixin`] from a Python dictionary of parameters.

        Args:
            processor_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~processing_utils.ProcessingMixin.to_dict`] method.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the processor object.

        Returns:
            [`~processing_utils.ProcessingMixin`]: The processor object instantiated from those
            parameters.
        """
        processor_dict = processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # We have to pop up some unused (but specific) kwargs and then validate that it doesn't contain unused kwargs
        # If we don't pop, some specific kwargs will raise a warning or error
        for unused_kwarg in cls.get_attributes() + ["auto_map", "processor_class"]:
            processor_dict.pop(unused_kwarg, None)

        # override processor_dict with given kwargs
        processor_dict.update(kwargs)

        # check if there is an overlap between args and processor_dict
        accepted_args_and_kwargs = cls.__init__.__code__.co_varnames[: cls.__init__.__code__.co_argcount][1:]

        # validate both processor_dict and given kwargs
        unused_kwargs, valid_kwargs = cls.validate_init_kwargs(
            processor_config=processor_dict, valid_kwargs=accepted_args_and_kwargs
        )

        # update args that are already in processor_dict to avoid duplicate arguments
        args_to_update = {
            i: valid_kwargs.pop(arg)
            for i, arg in enumerate(accepted_args_and_kwargs)
            if (arg in valid_kwargs and i < len(args))
        }
        args = [args_to_update.get(i, arg) for i, arg in enumerate(args)]

        # instantiate processor with used (and valid) kwargs only
        processor = cls(*args, **valid_kwargs)

        logger.info(f"Processor {processor}")
        if return_unused_kwargs:
            return processor, unused_kwargs
        else:
            return processor

    def _merge_kwargs(
        self,
        ModelProcessorKwargs: ProcessingKwargs,
        tokenizer_init_kwargs: dict | None = None,
        **kwargs,
    ) -> dict[str, dict]:
        """
        Method to merge dictionaries of kwargs cleanly separated by modality within a Processor instance.
        The order of operations is as follows:
            1) kwargs passed as before have highest priority to preserve BC.
                ```python
                high_priority_kwargs = {"crop_size" = {"height": 222, "width": 222}, "padding" = "max_length"}
                processor(..., **high_priority_kwargs)
                ```
            2) kwargs passed as modality-specific kwargs have second priority. This is the recommended API.
                ```python
                processor(..., text_kwargs={"padding": "max_length"}, images_kwargs={"crop_size": {"height": 222, "width": 222}}})
                ```
            3) kwargs passed during instantiation of a modality processor have fourth priority.
                ```python
                tokenizer = tokenizer_class(..., {"padding": "max_length"})
                image_processor = image_processor_class(...)
                processor(tokenizer, image_processor) # will pass max_length unless overridden by kwargs at call
                ```
            4) defaults kwargs specified at processor level have lowest priority.
                ```python
                class MyProcessingKwargs(ProcessingKwargs, CommonKwargs, TextKwargs, ImagesKwargs, total=False):
                    _defaults = {
                        "text_kwargs": {
                            "padding": "max_length",
                            "max_length": 64,
                        },
                    }
                ```
        Args:
            ModelProcessorKwargs (`ProcessingKwargs`):
                Typed dictionary of kwargs specifically required by the model passed.
            tokenizer_init_kwargs (`Dict`, *optional*):
                Dictionary of kwargs the tokenizer was instantiated with and need to take precedence over defaults.

        Returns:
            output_kwargs (`Dict`):
                Dictionary of per-modality kwargs to be passed to each modality-specific processor.

        """
        # holding a copy to avoid mutating user-provided arguments
        # Use deepcopy to also copy nested dicts (like videos_kwargs) that will be modified via pop()
        kwargs = copy.deepcopy(kwargs)

        # Initialize dictionaries
        output_kwargs = {
            "text_kwargs": {},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
        }

        default_kwargs = {
            "text_kwargs": {},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
        }

        map_preprocessor_kwargs = {
            "text_kwargs": "tokenizer",
            "images_kwargs": "image_processor",
            "audio_kwargs": "feature_extractor",
            "videos_kwargs": "video_processor",
        }

        possible_modality_keywords = {"text", "audio", "videos", "images"}
        used_keys = set()

        # get defaults from set model processor kwargs if they exist
        for modality in default_kwargs:  # noqa: PLC0206
            default_kwargs[modality] = ModelProcessorKwargs._defaults.get(modality, {}).copy()
            # Some preprocessors define a set of accepted "valid_kwargs" (currently only vision).
            # In those cases, we don’t declare a `ModalityKwargs` attribute in the TypedDict.
            # Instead, we dynamically obtain the kwargs from the preprocessor and merge them
            # with the general kwargs set. This ensures consistency between preprocessor and
            # processor classes, and helps prevent accidental mismatches.
            modality_valid_kwargs = set(ModelProcessorKwargs.__annotations__[modality].__annotations__)
            if modality in map_preprocessor_kwargs:
                preprocessor = getattr(self, map_preprocessor_kwargs[modality], None)
                preprocessor_valid_kwargs = (
                    getattr(preprocessor, "valid_kwargs", None) if preprocessor is not None else None
                )
                modality_valid_kwargs.update(
                    set(preprocessor_valid_kwargs.__annotations__ if preprocessor_valid_kwargs is not None else [])
                )
            # update defaults with arguments from tokenizer init
            for modality_key in modality_valid_kwargs:
                # init with tokenizer init kwargs if necessary
                if tokenizer_init_kwargs is not None and modality_key in tokenizer_init_kwargs:
                    value = (
                        getattr(self.tokenizer, modality_key)
                        if hasattr(self.tokenizer, modality_key)
                        else tokenizer_init_kwargs[modality_key]
                    )
                    default_kwargs[modality][modality_key] = value
        # now defaults kwargs are updated with the tokenizers defaults.
        # pass defaults to output dictionary
        output_kwargs.update(default_kwargs)

        # For `common_kwargs` just update all modality-specific kwargs with same key/values
        common_kwargs = ModelProcessorKwargs._defaults.get("common_kwargs", {})
        common_kwargs.update(kwargs.get("common_kwargs", {}))
        if common_kwargs:
            for kwarg in output_kwargs.values():
                kwarg.update(common_kwargs)

        # update modality kwargs with passed kwargs
        non_modality_kwargs = set(kwargs) - set(output_kwargs)
        for modality, output_kwarg in output_kwargs.items():
            modality_valid_kwargs = set(ModelProcessorKwargs.__annotations__[modality].__annotations__)
            if modality in map_preprocessor_kwargs:
                preprocessor = getattr(self, map_preprocessor_kwargs[modality], None)
                preprocessor_valid_kwargs = (
                    getattr(preprocessor, "valid_kwargs", None) if preprocessor is not None else None
                )
                modality_valid_kwargs.update(
                    set(preprocessor_valid_kwargs.__annotations__ if preprocessor_valid_kwargs is not None else [])
                )
            for modality_key in modality_valid_kwargs:
                # check if we received a structured kwarg dict or not to handle it correctly
                if modality in kwargs:
                    kwarg_value = kwargs[modality].pop(modality_key, "__empty__")
                    # check if this key was passed as a flat kwarg.
                    if kwarg_value != "__empty__" and modality_key in non_modality_kwargs:
                        raise ValueError(
                            f"Keyword argument {modality_key} was passed two times:\n"
                            f"in a dictionary for {modality} and as a **kwarg."
                        )
                    # fall back to the flat kwarg when the modality dict is present but doesn't carry this key
                    if kwarg_value == "__empty__" and modality_key in non_modality_kwargs:
                        kwarg_value = kwargs[modality_key]
                elif modality_key in kwargs:
                    # we get a modality_key instead of popping it because modality-specific processors
                    # can have overlapping kwargs
                    kwarg_value = kwargs.get(modality_key, "__empty__")
                else:
                    kwarg_value = "__empty__"
                if not isinstance(kwarg_value, str) or kwarg_value != "__empty__":
                    output_kwarg[modality_key] = kwarg_value
                    used_keys.add(modality_key)

        # Determine if kwargs is a flat dictionary or contains nested dictionaries
        if any(key in default_kwargs for key in kwargs):
            # kwargs is dictionary-based, and some keys match modality names
            for modality, subdict in kwargs.items():
                if modality in default_kwargs:
                    for subkey, subvalue in subdict.items():
                        if subkey not in used_keys:
                            output_kwargs[modality][subkey] = subvalue
                            used_keys.add(subkey)
        else:
            # kwargs is a flat dictionary
            for key, kwarg in kwargs.items():
                if key not in used_keys and key not in possible_modality_keywords:
                    logger.warning_once(
                        f"Keyword argument `{key}` is not a valid argument for this processor and will be ignored."
                    )

        for key, typed_dict_obj in ModelProcessorKwargs.__annotations__.items():
            if key in map_preprocessor_kwargs:
                preprocessor = getattr(self, map_preprocessor_kwargs[key], None)
                if preprocessor is None or getattr(preprocessor, "valid_kwargs", None) is None:
                    continue
                preprocessor_typed_dict_obj = preprocessor.valid_kwargs
                typed_dict_obj = _merge_typed_dict(preprocessor_typed_dict_obj, typed_dict_obj)
            validate_typed_dict(typed_dict_obj, output_kwargs[key])
        return output_kwargs

    @classmethod
    def from_pretrained(
        cls: type[Self],
        pretrained_model_name_or_path: str | os.PathLike,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs,
    ) -> Self:
        r"""
        Instantiate a processor associated with a pretrained model.

        <Tip>

        This class method is simply calling the feature extractor
        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`], image processor
        [`~image_processing_utils.ImageProcessingMixin`] and the tokenizer
        [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`] methods. Please refer to the docstrings of the
        methods above for more information.

        </Tip>

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~SequenceFeatureExtractor.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            **kwargs
                Additional keyword arguments passed along to both
                [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`] and
                [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`].
        """
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        # Resolve the revision once, so the processor config and all its sub-processors come from the same repo state.
        # kwargs["revision"] = resolve_revision(
        #     pretrained_model_name_or_path,
        #     revision,
        #     token=token,
        #     local_files_only=local_files_only,
        #     cache_dir=cache_dir,
        # )
        kwargs["revision"] = revision

        if token is not None:
            kwargs["token"] = token

        # Get processor_dict first so we can use it to instantiate non-tokenizer sub-processors
        processor_dict, instantiation_kwargs = cls.get_processor_dict(pretrained_model_name_or_path, **kwargs)
        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, processor_dict, **kwargs)
        return cls.from_args_and_dict(args, processor_dict, **instantiation_kwargs)

    @classmethod
    def get_attributes(cls):
        args_in_init = inspect.signature(cls.__init__).parameters.keys()
        attributes = []
        for sub_processor_type in args_in_init:
            # don't treat audio_tokenizer as an attribute
            if sub_processor_type == "audio_tokenizer":
                continue
            if any(modality in sub_processor_type for modality in MODALITY_TO_AUTOPROCESSOR_MAPPING):
                attributes.append(sub_processor_type)

        # Legacy processors may not override `__init__` and instead expose modality
        # attributes via `<attribute>_class`. In that case, `args_in_init` only exposes
        # `*args`/`**kwargs`, so we need to infer the attributes from those class-level
        # hints to keep backward compatibility (e.g. dynamic processors stored on the Hub).
        if not attributes:
            for attribute_name, value in cls.__dict__.items():
                if value is None or attribute_name == "audio_tokenizer_class" or not attribute_name.endswith("_class"):
                    continue
                inferred_attribute = attribute_name[: -len("_class")]
                if inferred_attribute == "audio_tokenizer":
                    continue
                if any(modality in inferred_attribute for modality in MODALITY_TO_AUTOPROCESSOR_MAPPING):
                    attributes.append(inferred_attribute)

        return attributes

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoProcessor"):
        """
        Register this class with a given auto class. This should only be used for custom feature extractors as the ones
        in the library are already mapped with `AutoProcessor`.



        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoProcessor"`):
                The auto class to register this new feature extractor with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import iantirta.models.core.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    @classmethod
    def _load_tokenizer_from_pretrained(
        cls, sub_processor_type, pretrained_model_name_or_path, subfolder="", **kwargs
    ):
        auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING["tokenizer"]
        is_primary = sub_processor_type == "tokenizer"

        if is_primary:
            # Primary tokenizer: load from root
            tokenizer = auto_processor_class.from_pretrained(
                pretrained_model_name_or_path, subfolder=subfolder, **kwargs
            )
        else:
            # Additional tokenizer: load from subfolder (e.g., "decoder_tokenizer")
            tokenizer_subfolder = os.path.join(subfolder, sub_processor_type) if subfolder else sub_processor_type
            try:
                tokenizer = auto_processor_class.from_pretrained(
                    pretrained_model_name_or_path, subfolder=tokenizer_subfolder, **kwargs
                )
            except (OSError, ValueError):
                fallback_folder = "the root directory" if not subfolder else f"subfolder `{subfolder}`"
                logger.warning(
                    f"Could not load tokenizer from subfolder `{tokenizer_subfolder}`. "
                    f"Falling back to loading from {fallback_folder}. "
                    f"This behavior is deprecated and will be removed in a future version."
                )
                tokenizer = auto_processor_class.from_pretrained(
                    pretrained_model_name_or_path, subfolder=subfolder, **kwargs
                )
        return tokenizer

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, processor_dict=None, **kwargs):
        """
        Identify and instantiate the subcomponents of Processor classes, such as image processors, tokenizers,
        and feature extractors. This method inspects the processor's `__init__` signature to identify parameters
        that correspond to known modality types (image_processor, tokenizer, feature_extractor, etc.) or contain
        modality names in their attribute name.

        For tokenizers: Uses the appropriate Auto class (AutoTokenizer) to load via `.from_pretrained()`.
        Additional tokenizers (e.g., "decoder_tokenizer") are loaded from subfolders.

        For other sub-processors (image_processor, feature_extractor, etc.): Primary ones are loaded via
        Auto class. Additional ones are instantiated from the config stored in processor_config.json
        (passed as processor_dict).

        Args:
            pretrained_model_name_or_path: Path or model id to load from.
            processor_dict: Optional dict containing processor config (from processor_config.json).
                Required when loading additional non-tokenizer sub-processors.
        """
        args = []
        processor_dict = processor_dict if processor_dict is not None else {}
        # Remove subfolder from kwargs to avoid duplicate keyword arguments
        subfolder = kwargs.pop("subfolder", "")

        # get args from processor init signature
        sub_processors = cls.get_attributes()
        for sub_processor_type in sub_processors:
            modality = _get_modality_for_attribute(sub_processor_type)
            is_primary = sub_processor_type == modality

            if (
                "tokenizer" in sub_processor_type
            ):  # This is only necessary for the checkpoint in test_processing_mistral3.py which has no config.json and
                # the tokenizer_config.json references LlamaTokenizerFast. TODO: update the config on the hub.
                if "PixtralProcessor" in cls.__name__:
                    from .tokenization_utils_tokenizers import TokenizersBackend

                    tokenizer = TokenizersBackend.from_pretrained(
                        pretrained_model_name_or_path, subfolder=subfolder, **kwargs
                    )
                else:
                    tokenizer = cls._load_tokenizer_from_pretrained(
                        sub_processor_type, pretrained_model_name_or_path, subfolder=subfolder, **kwargs
                    )
                args.append(tokenizer)
            elif is_primary:
                # Primary non-tokenizer sub-processor: load via Auto class
                auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING[sub_processor_type]
                # For backward compatibility, check if sub-processor class name is hardcoded as an attribute of the processor class.
                if hasattr(cls, sub_processor_type + "_class"):
                    sub_processor_class_name = getattr(cls, sub_processor_type + "_class")
                    logger.warning_once(
                        f"`{cls.__name__}` defines `{sub_processor_type}_class = '{sub_processor_class_name}'`, "
                        f"which is deprecated. Register the correct mapping in `{auto_processor_class.__name__}` instead.",
                    )
                    auto_processor_class = cls.get_possibly_dynamic_module(sub_processor_class_name)
                sub_processor = auto_processor_class.from_pretrained(
                    pretrained_model_name_or_path, subfolder=subfolder, **kwargs
                )
                args.append(sub_processor)

            elif sub_processor_type in processor_dict:
                # Additional non-tokenizer sub-processor: instantiate from config in processor_dict
                sub_processor_config = processor_dict[sub_processor_type]
                if isinstance(sub_processor_config, dict):
                    # Determine the class to instantiate
                    # Image processors have 'image_processor_type', feature extractors have 'feature_extractor_type'
                    type_key = f"{modality}_type"
                    class_name = sub_processor_config.get(type_key)
                    if class_name is None:
                        raise ValueError(
                            f"Cannot instantiate {sub_processor_type}: missing '{type_key}' in config. "
                            f"Config keys: {list(sub_processor_config.keys())}"
                        )
                    processor_class = cls.get_possibly_dynamic_module(class_name)
                    sub_processor = processor_class(**sub_processor_config)
                    args.append(sub_processor)
                else:
                    raise ValueError(
                        f"Expected dict for {sub_processor_type} in processor_config.json, "
                        f"got {type(sub_processor_config)}"
                    )
            else:
                raise ValueError(
                    f"Cannot find config for {sub_processor_type} in processor_config.json. "
                    f"Available keys: {list(processor_dict.keys())}"
                )

        return args

    @staticmethod
    def get_possibly_dynamic_module(module_name):
        import iantirta.models.common

        if hasattr(iantirta.models.common, module_name):
            return getattr(iantirta.models.common, module_name)
        lookup_locations = [
            iantirta.models.common.IMAGE_PROCESSOR_MAPPING,
            iantirta.models.common.VIDEO_PROCESSOR_MAPPING,
            iantirta.models.common.TOKENIZER_MAPPING,
            iantirta.models.common.FEATURE_EXTRACTOR_MAPPING,
            iantirta.models.common.MODEL_FOR_AUDIO_TOKENIZATION_MAPPING,
        ]
        for lookup_location in lookup_locations:
            for custom_class in lookup_location._extra_content.values():
                if isinstance(custom_class, tuple):
                    for custom_subclass in custom_class:
                        if custom_subclass is not None and custom_subclass.__name__ == module_name:
                            return custom_subclass
                elif custom_class is not None and custom_class.__name__ == module_name:
                    return custom_class
        raise ValueError(
            f"Could not find module {module_name} in `transformers`. If this is a custom class, "
            f"it should be registered using the relevant `AutoClass.register()` function so that "
            f"other functions can find it!"
        )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        if not hasattr(self, "tokenizer"):
            raise ValueError(f"Cannot batch decode text: {self.__class__.__name__} has no tokenizer.")
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        if not hasattr(self, "tokenizer"):
            raise ValueError(f"Cannot decode text: {self.__class__.__name__} has no tokenizer.")
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def unused_input_names(self) -> list[str]:
        "Input names returned always by subprocessors but not used in model's `forward`"
        return []

    @property
    def model_input_names(self) -> list[str]:
        model_input_names = []
        for attribute_name in self.get_attributes():
            attribute = getattr(self, attribute_name, None)
            if attribute is not None:
                attr_input_names = attribute.model_input_names
                model_input_names.extend(attr_input_names)
        return [name for name in model_input_names if name not in self.unused_input_names]

    @staticmethod
    def validate_init_kwargs(processor_config, valid_kwargs):
        kwargs_from_config = set(processor_config.keys())
        valid_kwargs_set = set(valid_kwargs)

        unused_keys = kwargs_from_config - valid_kwargs_set
        valid_keys = kwargs_from_config & valid_kwargs_set

        unused_kwargs = {k: processor_config[k] for k in unused_keys} if unused_keys else {}
        valid_kwargs = {k: processor_config[k] for k in valid_keys} if valid_keys else {}

        return unused_kwargs, valid_kwargs

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        chat_template: str | None = None,
        tools: list[dict] | None = None,
        documents: list[dict[str, str]] | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool | str = False,
        return_assistant_tokens_mask: bool = False,
        tokenize: bool = False,
        return_tensors: str | TensorType | None = None,
        return_dict: bool = False,
        load_audio_from_video: bool = False,
        processor_kwargs: dict | None = None,
        **kwargs,
    ) -> str:
        """
        Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
        conversations to turn them into a single tokenizable string.

        The input is expected to be in the following format, where each message content is a list consisting of text and
        optionally image or video inputs. One can also provide an image, video, URL or local path which will be used to form
        `pixel_values` when `return_dict=True`. If not provided, one will get only the formatted text, optionally tokenized text.

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "Please describe this image in detail."},
                ],
            },
        ]

        Args:
            conversation (`Union[list[Dict, [str, str]], list[list[dict[str, str]]]]`):
                The conversation to format.
            chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the tokenizer's
                chat template is used.
        """
        processor_kwargs = processor_kwargs or {}

        if chat_template is None:
            if isinstance(self.chat_template, dict) and "default" in self.chat_template:
                chat_template = self.chat_template["default"]
            elif isinstance(self.chat_template, dict):
                raise ValueError(
                    'The processor has multiple chat templates but none of them are named "default". You need to specify'
                    " which one to use by passing the `chat_template` argument. Available templates are: "
                    f"{', '.join(self.chat_template.keys())}"
                )
            elif self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError(
                    "Cannot use apply_chat_template because this processor does not have a chat template."
                )
        else:
            if isinstance(self.chat_template, dict) and chat_template in self.chat_template:
                # It's the name of a template, not a full template string
                chat_template = self.chat_template[chat_template]
            else:
                # It's a template string, render it directly
                pass

        # Users might still be passing processing kwargs in `**kwargs` so we need to filter
        # out additional kwargs that the template expects via Jinja2 template introspection
        template_kwargs = _get_template_variables(chat_template)
        processor_kwargs_from_kwargs = {k: v for k, v in kwargs.items() if k not in template_kwargs}
        if processor_kwargs_from_kwargs:
            logger.warning(
                "Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`"
            )
            processor_kwargs = processor_kwargs_from_kwargs

        # Check if tokenizer is fast - use backend attribute if available, otherwise fall back to class name
        is_tokenizers_fast = False
        if hasattr(self, "tokenizer"):
            if hasattr(self.tokenizer, "backend"):
                is_tokenizers_fast = self.tokenizer.backend == "tokenizers"
            else:
                # Fallback to class name check
                is_tokenizers_fast = self.tokenizer.__class__.__name__.endswith("Fast")

        if continue_final_message:
            if add_generation_prompt:
                raise ValueError(
                    "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."
                )
            if return_assistant_tokens_mask:
                raise ValueError("continue_final_message is not compatible with return_assistant_tokens_mask.")

        if return_assistant_tokens_mask:
            if not is_tokenizers_fast:
                raise ValueError(
                    "`return_assistant_tokens_mask` is not possible with slow tokenizers. Make sure you have `tokenizers` installed. "
                    "If the error persists, open an issue to support a Fast tokenizer for your model."
                )
            else:
                processor_kwargs["return_offsets_mapping"] = (
                    True  # force offset mapping so we can infer token boundaries
                )
                processor_kwargs["return_text_replacement_offsets"] = True

        # Set the sampling rate to load the audio files if user hasn't already passed with `kwargs`.
        audio_kwargs_from_user = processor_kwargs.get("audio_kwargs", {})
        sampling_rate = kwargs.get(
            "sampling_rate", processor_kwargs.get("sampling_rate", audio_kwargs_from_user.get("sampling_rate"))
        )
        if sampling_rate is None:
            if hasattr(self._audio_processor, "sampling_rate"):
                sampling_rate = self._audio_processor.sampling_rate
            else:
                sampling_rate = 16_000

        load_audio_backend = kwargs.get(
            "load_audio_backend",
            processor_kwargs.get("load_audio_backend", audio_kwargs_from_user.get("load_audio_backend")),
        )
        if load_audio_backend is None:
            default_audio_kwargs = self.valid_processor_kwargs._defaults.get("audio_kwargs", {})
            load_audio_backend = default_audio_kwargs.get("load_audio_backend", "auto")

        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
        ):
            is_batched = True
            conversations = conversation
        else:
            is_batched = False
            conversations = [conversation]

        # Normalize OpenAI-style "image_url" content blocks to HuggingFace-style "image" blocks
        # OpenAI format: {"type": "image_url", "image_url": {"url": "..."}}
        # HuggingFace format: {"type": "image", "url": "..."}
        for conversation_idx, conversation in enumerate(conversations):  # noqa: PLR1704
            for message in conversation:
                if not isinstance(message.get("content"), list):
                    continue
                new_content = []
                for content in message["content"]:
                    if isinstance(content, dict) and content.get("type") == "image_url" and "image_url" in content:
                        image_url_info = content["image_url"]
                        url = image_url_info.get("url", "") if isinstance(image_url_info, dict) else image_url_info
                        new_content.append({"type": "image", "url": url})
                    else:
                        new_content.append(content)
                message["content"] = new_content

        if tokenize:
            batch_images, batch_videos = [], []
            batch_audios = []
            for conversation in conversations:
                images, videos = [], []
                for message in conversation:
                    content = message.get("content") or []
                    if isinstance(content, str):
                        continue
                    visuals = [
                        content_block for content_block in content if content_block["type"] in ["image", "video"]
                    ]
                    audio_fnames = [
                        content_block[key]
                        for content_block in content
                        for key in ["audio", "url", "path"]
                        if key in content_block and content_block["type"] == "audio"
                    ]
                    image_fnames = [
                        vision_info[key]
                        for vision_info in visuals
                        for key in ["image", "url", "path", "base64"]
                        if key in vision_info and vision_info["type"] == "image"
                    ]
                    images.extend(image_fnames)
                    video_fnames = [
                        vision_info[key]
                        for vision_info in visuals
                        for key in ["video", "url", "path"]
                        if key in vision_info and vision_info["type"] == "video"
                    ]
                    videos.extend(video_fnames)

                    # Audio models do not accept nested list of audios (yet!) so we construct a flat input audio list
                    if not load_audio_from_video:
                        for fname in audio_fnames:
                            batch_audios.append(
                                load_audio(fname, sampling_rate=sampling_rate, backend=load_audio_backend)
                            )
                    else:
                        for fname in video_fnames:
                            # This updates the template in-place and adds audio entry
                            # to ensure `audio` token is added by jinja
                            message["content"].append({"type": "audio"})
                            batch_audios.append(
                                load_audio(fname, sampling_rate=sampling_rate, backend=load_audio_backend)
                            )

                # Currently all processors can accept nested list of batches, but not flat list of visuals
                # So we'll make a batched list of images and let the processor handle it
                batch_images.append(images)
                batch_videos.append(videos)

        # `kwargs` overwrite special tokens if both are present
        template_kwargs = {**self.tokenizer.special_tokens_map, **kwargs}
        prompt, generation_indices = render_jinja_template(
            conversations=conversations,
            tools=tools,
            documents=documents,
            chat_template=chat_template,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
            continue_final_message=continue_final_message,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )

        if not is_batched:
            prompt = prompt[0]

        if tokenize:
            # Tokenizer's `apply_chat_template` never adds special tokens when tokenizing
            # But processor's `apply_chat_template` didn't have an option to tokenize, so users had to format the prompt
            # and pass it to the processor. Users thus never worried about special tokens relying on processor handling
            # everything internally. The below line is to keep BC for that and be able to work with model that have
            # special tokens in the template (consistent with tokenizers). We dont want to raise warning, it will flood command line
            # without actionable solution for users
            single_prompt = prompt[0] if is_batched else prompt
            if self.tokenizer.bos_token is not None and single_prompt.startswith(self.tokenizer.bos_token):
                processor_kwargs["add_special_tokens"] = False

            # Always sample frames by default unless explicitly set to `False` by users. If users do not pass `num_frames`/`fps`
            # sampling should not done for BC.
            if "do_sample_frames" not in processor_kwargs and (
                processor_kwargs.get("fps") is not None or processor_kwargs.get("num_frames") is not None
            ):
                processor_kwargs["do_sample_frames"] = True

            # Set only is user passes a non-None value. Otherwise wa want to use each processor's own defaults
            if return_tensors:
                processor_kwargs["return_tensors"] = return_tensors

            # Audio was loaded/resampled by us above, so let the audio processor know at which rate.
            # (we additionally preserve the location of the kwarg in the nested structure kwargs -> processor -> audio)
            if batch_audios:
                if "sampling_rate" in audio_kwargs_from_user:
                    processor_kwargs["audio_kwargs"] = {**audio_kwargs_from_user, "sampling_rate": sampling_rate}
                else:
                    processor_kwargs["sampling_rate"] = sampling_rate

            images_exist = any((im is not None) for im_list in batch_images for im in im_list)
            videos_exist = any((vid is not None) for vid_list in batch_videos for vid in vid_list)
            out = self(
                text=prompt,
                images=batch_images if images_exist else None,
                videos=batch_videos if videos_exist else None,
                audio=batch_audios or None,
                **processor_kwargs,
            )

            if return_dict:
                if return_assistant_tokens_mask:
                    assistant_masks = []
                    offset_mapping = out.pop("offset_mapping")
                    input_ids = out["input_ids"]
                    # We do some corrections here to ensure the assistant masks aren't
                    # misaligned when we expand up image tokens
                    replacement_offsets = out.pop("text_replacement_offsets", None)
                    if replacement_offsets is None or len(replacement_offsets) == 0:
                        replacement_offsets = [[]] * len(input_ids)
                    for i in range(len(input_ids)):
                        current_mask = [0] * len(input_ids[i])
                        placeholder_ends = [r["span"][1] for r in replacement_offsets[i]]
                        chars_gained = [0] + [r["new_span"][1] - r["span"][1] for r in replacement_offsets[i]]
                        for span in generation_indices[i]:
                            # Shift the span past any placeholders that were expanded before it
                            start_char, end_char = (
                                char + chars_gained[bisect.bisect_right(placeholder_ends, char)] for char in span
                            )
                            # Mask every token overlapping the span. Zero-width tokens (padding, added specials) never
                            # match, and a span truncated away simply matches nothing
                            for pos, (token_start, token_end) in enumerate(offset_mapping[i]):
                                if token_start < end_char and token_end > start_char:
                                    current_mask[pos] = 1
                        assistant_masks.append(current_mask)
                    out["assistant_masks"] = assistant_masks
                    out.convert_to_tensors(tensor_type=return_tensors)
                return out
            else:
                return out["input_ids"]
        return prompt

    def parse_response(
        self,
        response: "str | list[int] | list[str] | list[list[int]] | np.ndarray | torch.Tensor",
        schema: dict | None = None,
        *,
        prefix: "str | list[int] | list[str] | list[list[int]] | np.ndarray | torch.Tensor | None" = None,
        tools: list[dict | Callable] | None = None,
    ):
        """
        Converts an output string created by generating text from a model into a parsed message dictionary.
        This method is intended for use with chat models, and will read the tokenizer's `response_template`
        attribute to control parsing, unless a `schema` argument is passed directly.

        Accepts either a single sequence (returning a single message `dict`) or a batch (returning a `list` of
        message dicts, one per item).

        Args:
            response (`str`, token ids, 1D/2D tensor, or a list of these):
                The output generated by the model, either decoded text or token ids, as a single sequence or a
                batch.
            schema (`dict`, *optional*):
                A response template. If not provided, the tokenizer's `response_template` attribute is used.
            prefix (`str`, token ids, 1D/2D tensor, or a list of these):
                The prompt that came before generation. Many chat templates pre-write part of the message, so
                this is needed to parse correctly. For a batched `response`, pass either a single prefix
                (broadcast to every item) or one prefix per item. Only supported with new-style templates.
            tools (`list[Union[Dict, Callable]]`, *optional*):
                Tools available to the model, in the same format as `apply_chat_template` accepts.
                Tool-call arguments are cast using the calling tool's JSON schema.
        """
        if not hasattr(self, "tokenizer"):
            raise ValueError("Can't use parse_response on a processor class without a tokenizer!")
        return self.tokenizer.parse_response(response, schema, prefix=prefix, tools=tools)

    def post_process_multimodal_output(
        self, generated_outputs, skip_special_tokens=True, generation_mode=None, **kwargs
    ):
        """
        Post-process the output of a multimodal model to return the requested modality output.
        If the model cannot generated the requested modality, an error will be raised.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            generation_mode (`str`, *optional*):
                Generation mode indicated which modality to output and can be one of `["text", "image", "audio"]`.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text.
        """
        if generation_mode is not None and generation_mode != "text":
            raise ValueError(
                f"{self.__class__.__name__} got an unexpected generation_mode={generation_mode}. Supported options are only [`text`]"
            )
        return self.post_process_image_text_to_text(
            generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs
        )

    def post_process_image_text_to_text(self, generated_outputs, skip_special_tokens=True, **kwargs):
        """
        Post-process the output of a vlm to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `decode` method.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.decode(generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs)

    def _check_special_mm_tokens(self, text: list[str], text_inputs: "BatchFeature", modalities: list[str]):
        """
        Checks that number of special tokens in text and processed text is same. The count can be different
        if tokenized text was truncated, leading to issues in model code.
        """
        input_ids = text_inputs["input_ids"]
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        for modality in modalities:
            token_str = getattr(self, f"{modality}_token", None)
            token_id = getattr(self, f"{modality}_token_id", None)
            if token_str is not None and token_id is not None:
                ids_count = [list(ids).count(token_id) for ids in input_ids]
                text_count = [sample.count(token_str) for sample in text]

                if ids_count != text_count:
                    raise ValueError(
                        f"Mismatch in `{modality}` token count between text and `input_ids`. Got ids={ids_count} and text={text_count}. "
                        "Likely due to `truncation='max_length'`. Please disable truncation or increase `max_length`."
                    )
