# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Auto Model class."""

from collections import OrderedDict

from ...utils import logging
from .auto_factory import (
    _BaseAutoModelClass,
    _LazyAutoMapping,
    auto_class_update,
)
from .configuration_auto import CONFIG_MAPPING_NAMES

logger = logging.get_logger(__name__)

MODEL_MAPPING_NAMES = OrderedDict(
    [
        # Base model mapping
        ("qwen3", "Qwen3Model"),
        ("qwen3_asr", "Qwen3ASRModel"),
        ("qwen3_asr_encoder", "Qwen3ASREncoder"),
        ("wav2vec2", "Wav2Vec2Model"),
    ]
)

# Models that accept text and optionally multimodal data in inputs
# and can generate text and optionally multimodal data.
MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES = OrderedDict(
    [
        ("qwen3_asr", "Qwen3ASRForConditionalGeneration"),
    ]
)

MODEL_FOR_CTC_MAPPING_NAMES = OrderedDict(
    [
        # Model for Connectionist temporal classification (CTC) mapping
        ("wav2vec2", "Wav2Vec2ForCTC"),
    ]
)

MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)
MODEL_FOR_MULTIMODAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES)
MODEL_FOR_CTC_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CTC_MAPPING_NAMES)


class AutoModel(_BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING


AutoModel = auto_class_update(AutoModel)


class AutoModelForMultimodalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MULTIMODAL_LM_MAPPING


AutoModelForMultimodalLM = auto_class_update(AutoModelForMultimodalLM, head_doc="multimodal generation")


class AutoModelForCTC(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CTC_MAPPING


AutoModelForCTC = auto_class_update(AutoModelForCTC, head_doc="connectionist temporal classification")


__all__ = [
    "MODEL_FOR_CTC_MAPPING",
    "MODEL_FOR_MULTIMODAL_LM_MAPPING",
    "MODEL_MAPPING",
    "AutoModel",
    "AutoModelForCTC",
    "AutoModelForMultimodalLM",
]
