# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

from iantirta.models._lazy_import import _LazyModule

_import_structure = {
    "configuration_utils": [
        "BaseWatermarkingConfig",
        "CompileConfig",
        "ContinuousBatchingConfig",
        "GenerationConfig",
        "GenerationMode",
        "SynthIDTextWatermarkingConfig",
        "WatermarkingConfig",
    ],
    "candidate_generator": [
        "AssistedCandidateGenerator",
        "CandidateGenerator",
        "EarlyExitCandidateGenerator",
        "PromptLookupCandidateGenerator",
        "DFlashTokenCandidateGenerator",
    ],
    "logits_process": [
        "AlternatingCodebooksLogitsProcessor",
        "ClassifierFreeGuidanceLogitsProcessor",
        "EncoderNoRepeatNGramLogitsProcessor",
        "EncoderRepetitionPenaltyLogitsProcessor",
        "EpsilonLogitsWarper",
        "EtaLogitsWarper",
        "ExponentialDecayLengthPenalty",
        "ForcedBOSTokenLogitsProcessor",
        "ForcedEOSTokenLogitsProcessor",
        "InfNanRemoveLogitsProcessor",
        "LogitNormalization",
        "LogitsProcessor",
        "LogitsProcessorList",
        "MinLengthLogitsProcessor",
        "MinNewTokensLengthLogitsProcessor",
        "MinPLogitsWarper",
        "NoBadWordsLogitsProcessor",
        "NoRepeatNGramLogitsProcessor",
        "PrefixConstrainedLogitsProcessor",
        "RepetitionPenaltyLogitsProcessor",
        "SequenceBiasLogitsProcessor",
        "SuppressTokensLogitsProcessor",
        "SuppressTokensAtBeginLogitsProcessor",
        "SynthIDTextWatermarkLogitsProcessor",
        "TemperatureLogitsWarper",
        "TopHLogitsWarper",
        "TopKLogitsWarper",
        "TopPLogitsWarper",
        "TypicalLogitsWarper",
        "UnbatchedClassifierFreeGuidanceLogitsProcessor",
        "WhisperTimeStampLogitsProcessor",
        "WatermarkLogitsProcessor",
    ],
    "stopping_criteria": [
        "MaxLengthCriteria",
        "MaxTimeCriteria",
        "ConfidenceCriteria",
        "EosTokenCriteria",
        "StoppingCriteria",
        "StoppingCriteriaList",
        "validate_stopping_criteria",
        "StopStringCriteria",
    ],
    "continuous_batching": [
        "ContinuousBatchingManager",
        "ContinuousMixin",
        "FIFOScheduler",
        "PrefillFirstScheduler",
        "Scheduler",
    ],
    "utils": [
        "GenerationMixin",
        "GenerateBeamDecoderOnlyOutput",
        "GenerateBeamEncoderDecoderOutput",
        "GenerateDecoderOnlyOutput",
        "GenerateEncoderDecoderOutput",
    ],
    "watermarking": [
        "WatermarkDetector",
        "WatermarkDetectorOutput",
        "BayesianDetectorModel",
        "BayesianDetectorConfig",
        "SynthIDTextWatermarkDetector",
    ],
    "streamers": [
        "AsyncTextIteratorStreamer",
        "BaseStreamer",
        "TextIteratorStreamer",
        "TextStreamer",
        "TextDiffusionStreamer",
    ],
}

if TYPE_CHECKING:
    from .candidate_generator import (  # noqa: F401
        AssistedCandidateGenerator,
        CandidateGenerator,
        DFlashTokenCandidateGenerator,
        EarlyExitCandidateGenerator,
        PromptLookupCandidateGenerator,
    )
    from .configuration_utils import (  # noqa: F401
        BaseWatermarkingConfig,
        CompileConfig,
        ContinuousBatchingConfig,
        GenerationConfig,
        GenerationMode,
        SynthIDTextWatermarkingConfig,
        WatermarkingConfig,
    )
    from .continuous_batching import (  # noqa: F401
        ContinuousBatchingManager,
        ContinuousMixin,
        FIFOScheduler,
        PrefillFirstScheduler,
        Scheduler,
    )
    from .logits_process import (  # noqa: F401
        AlternatingCodebooksLogitsProcessor,
        ClassifierFreeGuidanceLogitsProcessor,
        EncoderNoRepeatNGramLogitsProcessor,
        EncoderRepetitionPenaltyLogitsProcessor,
        EpsilonLogitsWarper,
        EtaLogitsWarper,
        ExponentialDecayLengthPenalty,
        ForcedBOSTokenLogitsProcessor,
        ForcedEOSTokenLogitsProcessor,
        InfNanRemoveLogitsProcessor,
        LogitNormalization,
        LogitsProcessor,
        LogitsProcessorList,
        MinLengthLogitsProcessor,
        MinNewTokensLengthLogitsProcessor,
        MinPLogitsWarper,
        NoBadWordsLogitsProcessor,
        NoRepeatNGramLogitsProcessor,
        PrefixConstrainedLogitsProcessor,
        RepetitionPenaltyLogitsProcessor,
        SequenceBiasLogitsProcessor,
        SuppressTokensAtBeginLogitsProcessor,
        SuppressTokensLogitsProcessor,
        SynthIDTextWatermarkLogitsProcessor,
        TemperatureLogitsWarper,
        TopHLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
        TypicalLogitsWarper,
        UnbatchedClassifierFreeGuidanceLogitsProcessor,
        WatermarkLogitsProcessor,
        WhisperTimeStampLogitsProcessor,
    )
    from .stopping_criteria import (  # noqa: F401
        ConfidenceCriteria,
        EosTokenCriteria,
        MaxLengthCriteria,
        MaxTimeCriteria,
        StoppingCriteria,
        StoppingCriteriaList,
        StopStringCriteria,
        validate_stopping_criteria,
    )
    from .streamers import (  # noqa: F401
        AsyncTextIteratorStreamer,
        BaseStreamer,
        TextDiffusionStreamer,
        TextIteratorStreamer,
        TextStreamer,
    )
    from .utils import (  # noqa: F401
        GenerateBeamDecoderOnlyOutput,
        GenerateBeamEncoderDecoderOutput,
        GenerateDecoderOnlyOutput,
        GenerateEncoderDecoderOutput,
        GenerationMixin,
    )
    from .watermarking import (  # noqa: F401
        BayesianDetectorConfig,
        BayesianDetectorModel,
        SynthIDTextWatermarkDetector,
        WatermarkDetector,
        WatermarkDetectorOutput,
    )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
