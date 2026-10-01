# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of transformers, improved by iantirta.com
#
# Copyright 2022 The HuggingFace Inc. team.
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
"""Generation configuration class and utilities."""

import copy
import json
import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

from iantirta.models.exceptions import YetToImplement

from .. import __version__
from ..utils import (
    GENERATION_CONFIG_NAME,
    ExplicitEnum,
    PushToHubMixin,
    cached_file,
    logging,
    resolve_revision,
)

if TYPE_CHECKING:
    import torch

    from ..configuration_utils import PreTrainedConfig
    from ..modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)
METADATA_FIELDS = ("_from_model_config", "_commit_hash", "_original_object_hash", "transformers_version")
STATIC_CACHE_IMPLEMENTATIONS = ("static", "offloaded_static")
DYNAMIC_CACHE_IMPLEMENTATIONS = ("dynamic", "offloaded", "quantized")
# All the following are redundant and deprecated, but kept for BC
DEPRECATED_STATIC_CACHE_IMPLEMENTATIONS = (
    "sliding_window",
    "hybrid",
    "hybrid_chunked",
    "offloaded_hybrid",
    "offloaded_hybrid_chunked",
)
ALL_STATIC_CACHE_IMPLEMENTATIONS = STATIC_CACHE_IMPLEMENTATIONS + DEPRECATED_STATIC_CACHE_IMPLEMENTATIONS
ALL_CACHE_IMPLEMENTATIONS = ALL_STATIC_CACHE_IMPLEMENTATIONS + DYNAMIC_CACHE_IMPLEMENTATIONS


def _should_warn(outer_attr: str, inner_attr: str, user_set_attributes: set | None) -> bool:
    """Determine if we should raise a warning for the combination `outer_attr` and `inner_attr`, based on whether
    they were provided explicitly, i.e. if they were in `user_set_attributes`.
    For example, if `outer_attr="do_sample"`, the warnings should be suppressed for `inner_attr` flags (e.g. "top_p") that weren't
    explicitly set by the caller. When `do_sample=False` is explicitly required by the user, values such as `top_p` inherited
    from a model's `generation_config.json` are harmless when the user opts for greedy decoding.
    """
    outer_sample_set = user_set_attributes is not None and outer_attr in user_set_attributes
    inner_attr_set = user_set_attributes is not None and inner_attr in user_set_attributes
    # We should warn only if both are explicitly set, none are set, or only the inner_attr is set while outer_attr is not
    return (
        (outer_sample_set and inner_attr_set)
        or (not outer_sample_set and not inner_attr_set)
        or (inner_attr_set and not outer_sample_set)
    )


class GenerationMode(ExplicitEnum):
    """
    Possible generation modes, downstream of the [`~generation.GenerationMixin.generate`] method.
    """

    # Non-beam methods
    CONTRASTIVE_SEARCH = "contrastive_search"
    GREEDY_SEARCH = "greedy_search"
    SAMPLE = "sample"
    ASSISTED_GENERATION = "assisted_generation"
    DOLA_GENERATION = "dola_generation"
    # Beam methods
    BEAM_SEARCH = "beam_search"
    BEAM_SAMPLE = "beam_sample"
    CONSTRAINED_BEAM_SEARCH = "constrained_beam_search"
    GROUP_BEAM_SEARCH = "group_beam_search"


class GenerationConfig(PushToHubMixin):
    extra_output_flags = ("output_attentions", "output_hidden_states", "output_scores", "output_logits")

    # Tensor versions of token IDs, set by _prepare_special_tokens() at generation time
    _bos_token_tensor: "torch.Tensor | None"
    _eos_token_tensor: "torch.Tensor | None"
    _pad_token_tensor: "torch.Tensor | None"
    _decoder_start_token_tensor: "torch.Tensor | None"

    # Hash to detect whether the instance was modified after loading
    _original_object_hash: int | None

    # Set at runtime to correctly slice inputs in `_prefill` in case we restart from an existing non-empty Cache, and the mask would
    # otherwise be dropped due to containing only 1s. This allows to differentiate between restarting with full or sliced input_ids
    _mask_length: int | None

    def __init__(self, **kwargs):
        # Snapshot of the attributes the caller explicitly provided (before the `kwargs.pop(...)` calls below
        # consume them). Used by `validate()` to restrict "minor issue" warnings to flags actually set by the user,
        # as opposed to defaults inherited from a model's `generation_config.json`.
        user_set_attributes = set(kwargs.keys())

        # Parameters that control the length of the output
        self.max_length = kwargs.pop("max_length", None)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        self.min_length = kwargs.pop("min_length", None)
        self.min_new_tokens = kwargs.pop("min_new_tokens", None)
        self.early_stopping = kwargs.pop("early_stopping", None)
        self.max_time = kwargs.pop("max_time", None)
        self.stop_strings = kwargs.pop("stop_strings", None)

        # Parameters that control the generation strategy used
        self.do_sample = kwargs.pop("do_sample", None)
        self.num_beams = kwargs.pop("num_beams", None)
        self.use_mtp = kwargs.pop("use_mtp", None)

        # Parameters that control the cache
        self.use_cache = kwargs.pop("use_cache", None)
        self.cache_implementation = kwargs.pop("cache_implementation", None)
        self.cache_config = kwargs.pop("cache_config", None)
        self.max_cache_len = kwargs.pop("max_cache_len", None)

        # Parameters for manipulation of the model output logits
        self.temperature = kwargs.pop("temperature", None)
        self.top_k = kwargs.pop("top_k", None)
        self.top_p = kwargs.pop("top_p", None)
        self.min_p = kwargs.pop("min_p", None)
        self.top_h = kwargs.pop("top_h", None)
        self.typical_p = kwargs.pop("typical_p", None)
        self.epsilon_cutoff = kwargs.pop("epsilon_cutoff", None)
        self.eta_cutoff = kwargs.pop("eta_cutoff", None)
        self.repetition_penalty = kwargs.pop("repetition_penalty", None)
        self.encoder_repetition_penalty = kwargs.pop("encoder_repetition_penalty", None)
        self.length_penalty = kwargs.pop("length_penalty", None)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", None)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.renormalize_logits = kwargs.pop("renormalize_logits", None)
        self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
        self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
        self.remove_invalid_values = kwargs.pop("remove_invalid_values", None)
        self.exponential_decay_length_penalty = kwargs.pop("exponential_decay_length_penalty", None)
        self.suppress_tokens = kwargs.pop("suppress_tokens", None)
        self.begin_suppress_tokens = kwargs.pop("begin_suppress_tokens", None)
        self.sequence_bias = kwargs.pop("sequence_bias", None)
        self.token_healing = kwargs.pop("token_healing", None)
        self.guidance_scale = kwargs.pop("guidance_scale", None)

        self.watermarking_config = kwargs.pop("watermarking_config", None)
        if isinstance(self.watermarking_config, dict):
            raise YetToImplement("Watermark Config")

        # Parameters that define the output variables of `generate`
        self.num_return_sequences = kwargs.pop("num_return_sequences", None)
        self.output_attentions = kwargs.pop("output_attentions", None)
        self.output_hidden_states = kwargs.pop("output_hidden_states", None)
        self.output_scores = kwargs.pop("output_scores", None)
        self.output_logits = kwargs.pop("output_logits", None)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", None)

        # Special tokens that can be used at generation time
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Generation parameters exclusive to encoder-decoder models
        self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", None)
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # Assistant generation
        self.is_assistant = kwargs.pop("is_assistant", None)
        self.num_assistant_tokens = kwargs.pop("num_assistant_tokens", None)
        self.num_assistant_tokens_schedule = kwargs.pop("num_assistant_tokens_schedule", None)
        self.assistant_confidence_threshold = kwargs.pop("assistant_confidence_threshold", None)
        self.prompt_lookup_num_tokens = kwargs.pop("prompt_lookup_num_tokens", None)
        self.max_matching_ngram_size = kwargs.pop("max_matching_ngram_size", None)
        self.assistant_early_exit = kwargs.pop("assistant_early_exit", None)
        self.assistant_lookbehind = kwargs.pop("assistant_lookbehind", None)
        self.target_lookbehind = kwargs.pop("target_lookbehind", None)
        self.assistant_ensemble_weight = kwargs.pop("assistant_ensemble_weight", None)
        self.speculation_type = kwargs.pop("speculation_type", None)

        # Performance
        self.compile_config = kwargs.pop("compile_config", None)
        self.disable_compile = kwargs.pop("disable_compile", None)

        # Deprecated in 5.13
        self.continuous_batching_config = kwargs.pop("continuous_batching_config", None)
        if self.continuous_batching_config is not None:
            msg = (
                "Passing ContinuousBatchingConfig through GenerationConfig is deprecated and will be removed in v5.19. "
                "Please pass it separately using the continuous_batching_config kwarg."
            )
            warnings.warn(msg, FutureWarning, stacklevel=2)

        # Deprecated (moved to the Hub). TODO remove for v5
        self.low_memory = kwargs.pop("low_memory", None)
        self.penalty_alpha = kwargs.pop("penalty_alpha", None)
        self.dola_layers = kwargs.pop("dola_layers", None)
        self.diversity_penalty = kwargs.pop("diversity_penalty", None)
        self.num_beam_groups = kwargs.pop("num_beam_groups", None)
        self.constraints = kwargs.pop("constraints", None)
        self.force_words_ids = kwargs.pop("force_words_ids", None)

        self.prefill_chunk_size = kwargs.pop("prefill_chunk_size", None)

        # Common attributes
        # BC: generation configs saved by older versions may still carry `_commit_hash`, it is not used anymore.
        kwargs.pop("_commit_hash", None)
        self._from_model_config = kwargs.pop("_from_model_config", None)
        self.transformers_version = kwargs.pop("transformers_version", None)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing
            # a `GenerationConfig` from a model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise
        else:
            # Ensure backward compatibility for models that use `forced_bos_token_id` within their config
            if kwargs.get("force_bos_token_to_be_generated", False):
                self.forced_bos_token_id = self.bos_token_id
                logger.warning_once(
                    f"Please make sure the generation config includes `forced_bos_token_id={self.bos_token_id}`. "
                )

        # Validate the values of the attributes
        self.validate(user_set_attributes=user_set_attributes)

    def __hash__(self):
        return hash(self.to_json_string(ignore_metadata=True))

    def __eq__(self, other):
        if not isinstance(other, GenerationConfig):
            return False

        self_without_metadata = self.to_json_string(use_diff=False, ignore_metadata=True)
        other_without_metadata = other.to_json_string(use_diff=False, ignore_metadata=True)
        return self_without_metadata == other_without_metadata

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string(ignore_metadata=True)}"

    def get_generation_mode(self, assistant_model: Optional["PreTrainedModel"] = None) -> GenerationMode:
        # TODO joao: find out a way of not depending on external fields (e.g. `assistant_model`), then make this a
        # property and part of the `__repr__`
        if self.constraints is not None or self.force_words_ids is not None:
            generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
        elif self.num_beams is None or self.num_beams == 1:
            if self.do_sample is not True:
                if (
                    self.top_k is not None
                    and self.top_k > 1
                    and self.penalty_alpha is not None
                    and self.penalty_alpha > 0
                ):
                    generation_mode = GenerationMode.CONTRASTIVE_SEARCH
                else:
                    generation_mode = GenerationMode.GREEDY_SEARCH
            else:
                generation_mode = GenerationMode.SAMPLE
        else:
            if self.num_beam_groups is not None and self.num_beam_groups > 1:
                generation_mode = GenerationMode.GROUP_BEAM_SEARCH
            elif self.do_sample is True:
                generation_mode = GenerationMode.BEAM_SAMPLE
            else:
                generation_mode = GenerationMode.BEAM_SEARCH

        # Assisted generation may extend some generation modes
        if (
            assistant_model is not None
            or self.use_mtp
            or self.prompt_lookup_num_tokens is not None
            or self.assistant_early_exit is not None
        ):
            if generation_mode in ("greedy_search", "sample"):
                generation_mode = GenerationMode.ASSISTED_GENERATION
            else:
                logger.warning(
                    "You've set `assistant_model` or `use_mtp`, which triggers assisted generate. Currently, assisted generate "
                    "is only supported with Greedy Search and Sample. However, the base decoding mode (based on "
                    f"current flags) is {generation_mode} -- some of the set flags will be ignored."
                )

        # DoLa generation may extend some generation modes
        # TODO joao, manuel: remove this in v4.62.0
        if self.dola_layers is not None:
            if generation_mode in ("greedy_search", "sample"):
                generation_mode = GenerationMode.DOLA_GENERATION
            else:
                logger.warning(
                    "You've set `dola_layers`, which triggers DoLa generate. Currently, DoLa generate "
                    "is only supported with Greedy Search and Sample.  However, the base decoding mode (based on "
                    f"current flags) is {generation_mode} -- some of the set flags will be ignored."
                )
        return generation_mode

    @staticmethod
    def _get_default_generation_params() -> dict[str, Any]:
        """
        Defaults to be applied when unset by the model OR by the user, such that `model.generate()` works with minimal
        parameterization.

        Pretrained checkpoints should set these as appropriate in their `generation_config.json`, to establish
        a better default baseline. Be mindful that tests will often use these values.
        """
        return {
            "max_length": 20,
            "min_length": 0,
            "do_sample": False,
            "use_cache": True,
            "early_stopping": False,
            "num_beams": 1,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "typical_p": 1.0,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "encoder_no_repeat_ngram_size": 0,
            "bad_words_ids": None,
            "num_return_sequences": 1,
            "output_scores": False,
            "return_dict_in_generate": False,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "remove_invalid_values": False,
            "exponential_decay_length_penalty": None,
            "suppress_tokens": None,
            "begin_suppress_tokens": None,
            "epsilon_cutoff": 0.0,
            "eta_cutoff": 0.0,
            "encoder_repetition_penalty": 1.0,
            "num_assistant_tokens": 20,
            "num_assistant_tokens_schedule": "constant",
            "assistant_confidence_threshold": 0.4,
            "assistant_lookbehind": 10,
            "target_lookbehind": 10,
            # Deprecated arguments (moved to the Hub). TODO joao, manuel: remove in v4.62.0
            "num_beam_groups": 1,
            "diversity_penalty": 0.0,
        }

    def validate(self, strict=False, user_set_attributes: set[str] | None = None):
        """
        Validates the values of the attributes of the [`GenerationConfig`] instance. Raises exceptions in the presence
        of parameterization that can be detected as incorrect from the configuration instance alone.

        Note that some parameters not validated here are best validated at generate runtime, as they may depend on
        other inputs and/or the model, such as parameters related to the generation length.

        Args:
            strict (bool): If True, raise an exception for any issues found. If False, only log issues.
            user_set_attributes (set[str], *optional*): Names of attributes the caller explicitly provided. When
                supplied, "minor issue" warnings about conflicting flag combinations (e.g. sampling-only flags set
                while `do_sample=False`) only fire if the conflicting flag is in this set -- avoiding noisy warnings
                when the value was inherited from a model's default `generation_config.json`. When `None`, all set
                attributes are considered user-set (backward-compatible behavior for direct `validate()` calls).
        """
        minor_issues = {}  # format: {attribute_name: issue_description}

        # 1. Validation of individual attributes
        # 1.1. Decoding attributes
        if self.early_stopping not in {None, True, False, "never"}:
            raise ValueError(f"`early_stopping` must be a boolean or 'never', but is {self.early_stopping}.")
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError(f"`max_new_tokens` must be greater than 0, but is {self.max_new_tokens}.")
        if self.assistant_ensemble_weight is not None and not (0.0 < self.assistant_ensemble_weight < 1.0):
            raise ValueError(
                f"`assistant_ensemble_weight` must be in the open interval `(0.0, 1.0)`, "
                f"but is {self.assistant_ensemble_weight}. Use `None` for standard (lossless) speculative decoding."
            )
        if self.pad_token_id is not None and self.pad_token_id < 0:
            minor_issues["pad_token_id"] = (
                f"`pad_token_id` should be positive but got {self.pad_token_id}. This will cause errors when batch "
                "generating, if there is padding. Please set `pad_token_id` explicitly as "
                "`model.generation_config.pad_token_id=PAD_TOKEN_ID` to avoid errors in generation"
            )
        # 1.2. Cache attributes
        # "paged" re-routes to continuous batching and so it is a valid cache implementation. But we do not want to test
        # it with the `generate` as the other would be, so we we cannot add it to ALL_CACHE_IMPLEMENTATIONS
        valid_cache_implementations = ALL_CACHE_IMPLEMENTATIONS + ("paged",)
        if self.cache_implementation is not None and self.cache_implementation not in valid_cache_implementations:
            raise ValueError(
                f"Invalid `cache_implementation` ({self.cache_implementation}). Choose one of: "
                f"{valid_cache_implementations}"
            )
        if self.max_cache_len is not None and self.cache_implementation not in ALL_STATIC_CACHE_IMPLEMENTATIONS:
            logger.warning_once(
                f"`max_cache_len` is only used with static caches ({STATIC_CACHE_IMPLEMENTATIONS}); it will be "
                f"ignored with `cache_implementation={self.cache_implementation!r}`."
            )
        # 1.3. Performance attributes
        if self.compile_config is not None and not isinstance(self.compile_config, CompileConfig):
            raise ValueError(
                f"You provided `compile_config` as an instance of {type(self.compile_config)}, but it must be an "
                "instance of `CompileConfig`."
            )
        # 1.4. Watermarking attributes
        if self.watermarking_config is not None:
            self.watermarking_config.validate()

        # 2. Validation of attribute combinations
        # 2.1. detect sampling-only parameterization when not in sampling mode

        # Note that we check `is not True` in purpose. Boolean fields can also be `None` so we
        # have to be explicit. Value of `None` is same as having `False`, i.e. the default value

        if self.do_sample is not True:
            greedy_wrong_parameter_msg = (
                "`do_sample` is not set to `True`. However, `{flag_name}` is set to `{flag_value}` -- this flag is "
                "only used in sample-based generation modes. You should set `do_sample=True` or unset `{flag_name}`."
            )

            if (
                self.temperature is not None
                and self.temperature != 1.0
                and _should_warn("do_sample", "temperature", user_set_attributes)
            ):
                minor_issues["temperature"] = greedy_wrong_parameter_msg.format(
                    flag_name="temperature", flag_value=self.temperature
                )
            if (
                self.top_p is not None
                and self.top_p != 1.0
                and _should_warn("do_sample", "top_p", user_set_attributes)
            ):
                minor_issues["top_p"] = greedy_wrong_parameter_msg.format(flag_name="top_p", flag_value=self.top_p)
            if self.min_p is not None and _should_warn("do_sample", "min_p", user_set_attributes):
                minor_issues["min_p"] = greedy_wrong_parameter_msg.format(flag_name="min_p", flag_value=self.min_p)
            if self.top_h is not None and _should_warn("do_sample", "top_h", user_set_attributes):
                minor_issues["top_h"] = greedy_wrong_parameter_msg.format(flag_name="top_h", flag_value=self.top_h)
            if (
                self.typical_p is not None
                and self.typical_p != 1.0
                and _should_warn("do_sample", "typical_p", user_set_attributes)
            ):
                minor_issues["typical_p"] = greedy_wrong_parameter_msg.format(
                    flag_name="typical_p", flag_value=self.typical_p
                )
            if self.top_k is not None and self.top_k != 50 and _should_warn("do_sample", "top_k", user_set_attributes):
                minor_issues["top_k"] = greedy_wrong_parameter_msg.format(flag_name="top_k", flag_value=self.top_k)
            if (
                self.epsilon_cutoff is not None
                and self.epsilon_cutoff != 0.0
                and _should_warn("do_sample", "epsilon_cutoff", user_set_attributes)
            ):
                minor_issues["epsilon_cutoff"] = greedy_wrong_parameter_msg.format(
                    flag_name="epsilon_cutoff", flag_value=self.epsilon_cutoff
                )
            if (
                self.eta_cutoff is not None
                and self.eta_cutoff != 0.0
                and _should_warn("do_sample", "eta_cutoff", user_set_attributes)
            ):
                minor_issues["eta_cutoff"] = greedy_wrong_parameter_msg.format(
                    flag_name="eta_cutoff", flag_value=self.eta_cutoff
                )

        # 2.2. detect beam-only parameterization when not in beam mode. Same provenance filtering as above --
        # both `num_beams` and the beam-only flag must be user-set for the warning to fire.
        if self.num_beams is None or self.num_beams == 1:
            single_beam_wrong_parameter_msg = (
                "`num_beams` is set to {num_beams}. However, `{flag_name}` is set to `{flag_value}` -- this flag is "
                "only used in beam-based generation modes. You should set `num_beams>1` or unset `{flag_name}`."
            )

            if (
                self.early_stopping is not None
                and self.early_stopping is not False
                and _should_warn("num_beams", "early_stopping", user_set_attributes)
            ):
                minor_issues["early_stopping"] = single_beam_wrong_parameter_msg.format(
                    num_beams=self.num_beams, flag_name="early_stopping", flag_value=self.early_stopping
                )
            if (
                self.length_penalty is not None
                and self.length_penalty != 1.0
                and _should_warn("num_beams", "length_penalty", user_set_attributes)
            ):
                minor_issues["length_penalty"] = single_beam_wrong_parameter_msg.format(
                    num_beams=self.num_beams, flag_name="length_penalty", flag_value=self.length_penalty
                )

        # 2.4. check `num_return_sequences`
        if self.num_return_sequences is not None and self.num_return_sequences > 1:
            if self.num_beams is None or self.num_beams == 1:
                if not self.do_sample:
                    raise ValueError(
                        "Greedy methods (do_sample != True) without beam search do not support "
                        f"`num_return_sequences` different than 1 (got {self.num_return_sequences})."
                    )
            elif (
                self.num_beams is not None
                and self.num_return_sequences is not None
                and self.num_return_sequences > self.num_beams
            ):
                raise ValueError(
                    f"`num_return_sequences` ({self.num_return_sequences}) has to be smaller or equal to `num_beams` "
                    f"({self.num_beams})."
                )

        # 2.5. check cache-related arguments
        if self.use_cache is False:
            # In this case, all cache-related arguments should be unset. However, since `use_cache=False` is often used
            # passed to `generate` directly to hot-fix cache issues, let's raise a warning instead of an error
            # (otherwise a user might need to overwrite several parameters).
            no_cache_warning = (
                "You have not set `use_cache` to `True`, but {cache_arg} is set to {cache_arg_value}."
                "{cache_arg} will have no effect."
            )
            for arg_name in ("cache_implementation", "cache_config"):
                if getattr(self, arg_name) is not None:
                    minor_issues[arg_name] = no_cache_warning.format(
                        cache_arg=arg_name, cache_arg_value=getattr(self, arg_name)
                    )

        # 2.6. other incorrect combinations
        if self.return_dict_in_generate is not True:
            for extra_output_flag in self.extra_output_flags:
                if getattr(self, extra_output_flag) is True:
                    minor_issues[extra_output_flag] = (
                        f"`return_dict_in_generate` is NOT set to `True`, but `{extra_output_flag}` is. When "
                        f"`return_dict_in_generate` is not `True`, `{extra_output_flag}` is ignored."
                    )

        # 2.7. Forcing a token while suppressing it. If every forced (bos/eos) token is also suppressed, all logits
        # become `-inf` at the forcing step, yielding `nan` probabilities and a generation crash (see #24099).
        if self.suppress_tokens is not None:
            suppressed_tokens = set(self.suppress_tokens)
            for forced_attr in ("forced_bos_token_id", "forced_eos_token_id"):
                forced_tokens = getattr(self, forced_attr)
                if forced_tokens is None:
                    continue
                forced_tokens = {forced_tokens} if isinstance(forced_tokens, int) else set(forced_tokens)
                if forced_tokens and forced_tokens.issubset(suppressed_tokens):
                    raise ValueError(
                        f"Every token in `{forced_attr}` ({sorted(forced_tokens)}) is also in `suppress_tokens`. "
                        "Forcing a token while suppressing it sets all logits to `-inf` at the forcing step, which "
                        "produces `nan` probabilities and crashes generation. Remove the overlapping token(s) from "
                        f"either `{forced_attr}` or `suppress_tokens` (if you meant to prevent an early EOS token, use "
                        "`min_new_tokens` instead)."
                    )

        # 3. Check common issue: passing `generate` arguments inside the generation config
        generate_arguments = (
            "logits_processor",
            "stopping_criteria",
            "prefix_allowed_tokens_fn",
            "synced_gpus",
            "assistant_model",
            "streamer",
            "negative_prompt_ids",
            "negative_prompt_attention_mask",
        )
        for arg in generate_arguments:
            if hasattr(self, arg):
                raise ValueError(
                    f"Argument `{arg}` is not a valid argument of `GenerationConfig`. It should be passed to "
                    "`generate()` (or a pipeline) directly."
                )

        # Finally, handle caught minor issues. With default parameterization, we will throw a minimal warning.
        if len(minor_issues) > 0:
            # Full list of issues with potential fixes
            info_message = []
            for attribute_name, issue_description in minor_issues.items():
                info_message.append(f"- `{attribute_name}`: {issue_description}")
            info_message = "\n".join(info_message)
            info_message += (
                "\nIf you're using a pretrained model, note that some of these attributes may be set through the "
                "model's `generation_config.json` file."
            )

            if strict:
                raise ValueError("GenerationConfig is invalid: \n" + info_message)
            else:
                attributes_with_issues = list(minor_issues.keys())
                warning_message = (
                    f"The following generation flags are not valid and may be ignored: {attributes_with_issues}."
                )
                if logging.get_verbosity() >= logging.WARNING:
                    warning_message += " Set `TRANSFORMERS_VERBOSITY=info` for more details."
                logger.warning_once(warning_message)
                logger.info_once(info_message)

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        config_file_name: str | os.PathLike | None = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        raise YetToImplement("save pretained is not supported.")
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: str | os.PathLike,
        config_file_name: str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs,
    ) -> "GenerationConfig":
        r"""
        Instantiate a [`GenerationConfig`] from a generation configuration file.

        Args:
            pretrained_model_name (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a configuration file saved using the
                  [`~GenerationConfig.save_pretrained`] method, e.g., `./my_model_directory/`.
            config_file_name (`str` or `os.PathLike`, *optional*, defaults to `"generation_config.json"`):
                Name of the generation configuration JSON file to be loaded from `pretrained_model_name`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `hf auth login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>

            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.

                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            kwargs (`dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from this pretrained model.

        Examples:

        ```python
        >>> from iantirta.models.vendor.transformers import GenerationConfig

        >>> # Download configuration from iantirta.models.vendor.huggingface.co and cache.
        >>> generation_config = GenerationConfig.from_pretrained("openai-community/gpt2")

        >>> # E.g. config was saved using *save_pretrained('./test/saved_model/')*
        >>> generation_config.save_pretrained("./test/saved_model/")
        >>> generation_config = GenerationConfig.from_pretrained("./test/saved_model/")

        >>> # You can also specify configuration names to your generation configuration file
        >>> generation_config.save_pretrained("./test/saved_model/", config_file_name="my_configuration.json")
        >>> generation_config = GenerationConfig.from_pretrained("./test/saved_model/", "my_configuration.json")

        >>> # If you'd like to try a minor variation to an existing configuration, you can also pass generation
        >>> # arguments to `.from_pretrained()`. Be mindful that typos and unused arguments will be ignored
        >>> generation_config, unused_kwargs = GenerationConfig.from_pretrained(
        ...     "openai-community/gpt2", top_k=1, foo=False, do_sample=True, return_unused_kwargs=True
        ... )
        >>> generation_config.top_k
        1

        >>> unused_kwargs
        {'foo': False}
        ```"""
        config_file_name = config_file_name if config_file_name is not None else GENERATION_CONFIG_NAME

        proxies = kwargs.pop("proxies", None)
        subfolder = kwargs.pop("subfolder", "")
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        # Resolve the revision once, so that all the files of this load come from the same repository state.
        revision = resolve_revision(
            pretrained_model_name,
            revision,
            token=token,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )

        user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        config_path = os.path.join(pretrained_model_name, config_file_name)
        config_path = str(config_path)

        is_local = os.path.exists(config_path)
        if os.path.isfile(os.path.join(subfolder, config_path)):
            # Special case when config_path is a local file
            resolved_config_file = config_path
            is_local = True
        else:
            configuration_file = config_file_name
            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_config_file = cached_file(
                    pretrained_model_name,
                    configuration_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                )
            except OSError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:  # noqa: BLE001
                # For any other exception, we throw a generic error.
                raise OSError(
                    f"Can't load the configuration of '{pretrained_model_name}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"
                    f" name. Otherwise, make sure '{pretrained_model_name}' is the correct path to a directory"
                    f" containing a {configuration_file} file"
                )

        try:
            # Load config dict
            config_dict = cls._dict_from_json_file(resolved_config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise OSError(f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file.")

        if is_local:
            logger.info(f"loading configuration file {resolved_config_file}")
        else:
            logger.info(f"loading configuration file {configuration_file} from cache at {resolved_config_file}")

        if kwargs.get("_from_model_config", False):
            return cls.from_model_config(config_dict)
        elif kwargs.get("return_unused_kwargs") is True:
            config, unused_kwargs = cls.from_dict(config_dict, **kwargs)
            config._original_object_hash = hash(config)  # Hash to detect whether the instance was modified
            return config, unused_kwargs
        else:
            config = cls.from_dict(config_dict, **kwargs)
            config._original_object_hash = hash(config)  # Hash to detect whether the instance was modified
            return config

    @classmethod
    def _dict_from_json_file(cls, json_file: str | os.PathLike):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs) -> "GenerationConfig":
        """
        Instantiates a [`GenerationConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        # Those arguments may be passed along for our internal telemetry.
        # We remove them so they don't appear in `return_unused_kwargs`.
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)

        # The line below allows model-specific config to be loaded as well through kwargs, with safety checks.
        # See https://github.com/huggingface/transformers/pull/21269
        config = cls(**{**config_dict, **kwargs})
        unused_kwargs = config.update(**kwargs)

        logger.info(f"Generate config {config}")
        if return_unused_kwargs:
            return config, unused_kwargs
        else:
            return config

    def dict_dtype_to_str(self, d: dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("dtype") is not None and not isinstance(d["dtype"], str):
            d["dtype"] = str(d["dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self.dict_dtype_to_str(value)

    def to_diff_dict(self) -> dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = GenerationConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if key not in default_config_dict or key == "transformers_version" or value != default_config_dict[key]:
                serializable_config_dict[key] = value

        self.dict_dtype_to_str(serializable_config_dict)
        return serializable_config_dict

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)

        # Fields to ignore at serialization time
        if "_commit_hash" in output:
            del output["_commit_hash"]
        if "_original_object_hash" in output:
            del output["_original_object_hash"]
        if "_mask_length" in output:
            del output["_mask_length"]

        # Transformers version when serializing this file
        output["transformers_version"] = __version__

        self.dict_dtype_to_str(output)
        return output

    def to_json_string(
        self, use_diff: bool = True, ignore_metadata: bool = False, keys_to_pop: list[str] | None = None
    ) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `GenerationConfig()`
                is serialized to JSON string.
            ignore_metadata (`bool`, *optional*, defaults to `False`):
                Whether to ignore the metadata fields present in the instance
            keys_to_pop (`list[str]`, *optional*):
                Keys to pop from the config dictionary before serializing

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()

        if keys_to_pop is not None:
            for key in keys_to_pop:
                config_dict.pop(key, None)

        if ignore_metadata:
            for metadata_field in METADATA_FIELDS:
                config_dict.pop(metadata_field, None)

        def convert_keys_to_string(obj):
            if isinstance(obj, dict):
                return {str(key): convert_keys_to_string(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_keys_to_string(item) for item in obj]
            else:
                return obj

        def convert_dataclass_to_dict(obj):
            if isinstance(obj, dict):
                return {key: convert_dataclass_to_dict(value) for key, value in obj.items()}
            elif is_dataclass(obj):
                # Some of our dataclasses have a custom `to_dict()` method, and we prefer it
                if hasattr(obj, "to_dict"):
                    return obj.to_dict()
            else:
                return obj

        config_dict = convert_keys_to_string(config_dict)
        config_dict = convert_dataclass_to_dict(config_dict)

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(
        self, json_file_path: str | os.PathLike, use_diff: bool = True, keys_to_pop: list[str] | None = None
    ) -> None:
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `GenerationConfig()`
                is serialized to JSON file.
            keys_to_pop (`list[str]`, *optional*):
                Keys to pop from the config dictionary before serializing
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff, keys_to_pop=keys_to_pop))

    @classmethod
    def from_model_config(cls, model_config: Union["PreTrainedConfig", dict]) -> "GenerationConfig":
        """
        Instantiates a [`GenerationConfig`] from a [`PreTrainedConfig`]. This function is useful to convert legacy
        [`PreTrainedConfig`] objects, which may contain generation parameters, into a stand-alone [`GenerationConfig`].

        Args:
            model_config (`PreTrainedConfig | dict`):
                The model config that will be used to instantiate the generation config.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from those parameters.
        """
        config_dict = model_config.to_dict() if not isinstance(model_config, dict) else model_config
        config_dict.pop("_from_model_config", None)

        # Removes all `None` from the model config dict -- this lets the generation config defaults to take hold
        config_dict = {key: value for key, value in config_dict.items() if value is not None}
        generation_config = cls.from_dict(config_dict, return_unused_kwargs=False, _from_model_config=True)

        # Special case: some models have generation attributes set in the decoder. Use them if still unset in the
        # generation config (which in turn is defined from the outer attributes of model config).
        if isinstance(model_config, dict):
            decoder_possible_text_config_names = ("decoder", "generator", "text_config")
            for text_config_name in decoder_possible_text_config_names:
                if text_config := model_config.get(text_config_name):
                    model_config = text_config
                    break
        else:
            model_config = model_config.get_text_config(decoder=True)
            model_config = model_config.to_dict()

        default_generation_config = GenerationConfig()
        for attr in generation_config.to_dict():
            is_unset = getattr(generation_config, attr) == getattr(default_generation_config, attr)
            if attr in model_config and is_unset:
                setattr(generation_config, attr, model_config[attr])

        # If any `output_...` flag is set to `True`, we ensure `return_dict_in_generate` is set to `True`.
        if not generation_config.return_dict_in_generate and any(
            getattr(generation_config, extra_output_flag, False)
            for extra_output_flag in generation_config.extra_output_flags
        ):
            generation_config.return_dict_in_generate = True

        # Hash to detect whether the instance was modified
        generation_config._original_object_hash = hash(generation_config)
        return generation_config

    def update(self, defaults_only=False, allow_custom_entries=False, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
        returning all the unused kwargs.

        Args:
            defaults_only (`bool`, *optional*, defaults to `False`):
                Whether to update all keys in config with `kwargs` or only those that are set to `None` (i.e. default value).
            allow_custom_entries (`bool`, *optional*, defaults to `False`):
                Whether to allow updating custom entries into the config with `kwargs` if not present in the current config.
            kwargs (`dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if allow_custom_entries and not hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)
            elif hasattr(self, key):
                if not defaults_only or getattr(self, key) is None:
                    if key == "watermarking_config" and isinstance(value, dict):
                        raise YetToImplement("Watermark Config")
                    setattr(self, key, value)
                    to_remove.append(key)

        # Confirm that the updated instance is still valid. Only attributes *explicitly* updated in this call count
        # as user-set for warning purposes: defaults inherited from a model's config shouldn't emit warnings.
        self.validate(user_set_attributes=set(to_remove))

        # Remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs


@dataclass
class CompileConfig:
    """
    Class that holds arguments relative to `torch.compile` behavior, when using automatic compilation in `generate`.
    See [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html) for more details on the arguments.

    Args:
        fullgraph (`bool`, *optional*, defaults to `False`):
            If False (default), attempts to discover compilable regions that will be optimized. If True, then require
            that the entire function be capturable into a single graph. If this is not possible (that is, if there are
            graph breaks), then an error will be raised.
        dynamic (`bool` or `None`, *optional*):
            Whether to try to use dynamic shape graphs.
        backend (`str` or `Callable`, *optional*, defaults to `"inductor"`):
            Backend to be used.
        mode (`str`, *optional*, defaults to `"reduce-overhead"`):
            Controls balance between performance and overhead.
        options (`dict`, *optional*):
            A dictionary of options to pass to the backend.

    Examples:
    ```python
    >>> from iantirta.models.vendor.transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig

    >>> tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')
    >>> model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b').cuda()

    >>> # Automatic compile configuration, used with static cache
    >>> compile_config = CompileConfig(dynamic=True)

    >>> # Generation with static cache and compile config
    >>> input = tokenizer.encode("Hello there, how", return_tensors="pt").cuda()
    >>> output = model.generate(
    ...     input, do_sample=False, max_new_tokens=300, cache_implementation="static", compile_config=compile_config
    ... )
    >>> output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    ```
    """

    fullgraph: bool = False
    dynamic: bool | None = None
    backend: str | Callable = "inductor"
    mode: str = "reduce-overhead"
    options: dict | None = None
    # Used to flag our `generate` call to compile on e.g. CPU. Often not optimal, but useful for testing purposes.
    _compile_all_devices = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes this instance to a Python dictionary."""
        return copy.deepcopy({key: value for key, value in self.__dict__.items() if key != "_compile_all_devices"})


# TODO: add the @strict decorator to prevent attributes passed as args rather than kwargs
@dataclass
class ContinuousBatchingConfig:
    """
    Class that holds arguments relative to continuous batching, when using continuous batching through the
    `generate_batch` method or the `continuous_batching_context_manager` context manager.

    Args:
        page_size (`int`, *optional*, defaults to 256):
            The number of tokens stored for each layer inside a (full-attention) page. A block storing the cache of N
            layers has N pages (one per layer), each holding cache for `page_size` tokens for one layer. Default is 256.
        num_blocks (`int`, *optional*):
            Number of blocks in the KV cache. Auto-inferred from GPU memory when `None`.
        max_batch_tokens (`int`, *optional*):
            Maximum number of tokens in a batch. Auto-inferred from GPU memory when `None`.
        max_memory_percent (`float`, *optional*):
            Maximum percentage of free GPU memory (after the model is loaded) to use for the KV cache. When `None`,
            resolved at runtime to 0.9 if there is no logit processing and 0.8 if there is, to leave headroom for
            vocabulary-sized temporary tensors.
        max_requests_per_batch (`int`, *optional*):
            Maximum number of requests per batch. Auto-inferred from workload hints when `None`, with fallback of 1024.
        max_blocks_per_request (`int`, *optional*):
            Maximum blocks per request, used in the `flash_attn_with_kvcache` fast decode path to dimension
            the block table. Setting this to 0 disables the fast decode path. Default is None (auto-inferred).
        allow_block_sharing (`bool`, *optional*, defaults to `True`):
            Whether to allow block sharing for prefix caching. Block sharing can only be allowed, never forced,
            as some models do not support it. Disable if you have few short prompts but long generation lengths.
        use_async_batching (`bool`, *optional*):
            Whether to enable async double-buffering, which removes CPU overhead from the continuous batching
            loop at the cost of doubled VRAM usage. Auto-detected when `None`.
        use_cuda_graph (`bool` or `tuple[bool, bool]`, *optional*):
            Whether to enable CUDA graphs. This can be a tuple of booleans (one for the varlen path and one for the
            decode fast path), a boolean which will apply to both paths, or None (automatically inferred). After calling
            `decide_use_cuda_graphs`, the attribute will be a tuple of booleans. Default is None (automatically inferred).
        q_padding_interval_size (`int`, *optional*, defaults to 0):
            Query padding granularity in tokens for CUDA graphs. Uses a preset from `continuous_api.py` when
            set to 0.
        kv_padding_interval_size (`int`, *optional*, defaults to 0):
            KV padding granularity in tokens for CUDA graphs. Uses a preset from `continuous_api.py` when
            set to 0.
        varlen_compile_config (`CompileConfig`, *optional*):
            CompileConfig for varlen (prefill) path. Default is None (uses generation_config fallback)
            The varlen path handles batches with varying query and KV lengths, often benefiting from dynamic=True.
        decode_compile_config (`CompileConfig`, *optional*):
            CompileConfig for decode (fast) path. Default is None (uses generation_config fallback)
            The decode path handles batches has no dynamic KV length, so static shapes are a better fit.
        default_compile_level (`int`, *optional*, defaults to 0):
            If this is >0 and no compile config is provided for varlen or decode path, a default compile config will be
            provided. The level can go up to 3, and a higher level means more performance but longer warmup time.
        scheduler_type (`str`, *optional*, defaults to `"fifo"`):
            Scheduler type to use.
        safety_margin (`float`, *optional*):
            Safety margin used to limit the amount of offloading. Defaults to None (use class default).
        return_logprobs (`bool`, *optional*, defaults to `False`):
            Whether to return log probabilities along with the generated tokens.
        seed (`int | None`, *optional*):
            An optional seed for generation. If not specified, the internal seed will be set to a random value.
        cpu_offload_space (`float`, *optional*, defaults to 0.0):
            CPU swap space in GiB for KV cache offloading. A pre-allocated pinned CPU buffer of this size is
            created at initialization. When the GPU cache is full, evicted requests' KV caches are copied here
            instead of being discarded. 0 disables offloading (default).
        cpu_offload_space_safety_threshold (`float`, *optional*, defaults to 0.8):
            If `cpu_offload_space` exceeds this fraction of total system RAM, it is clamped to avoid host OOM.
            Set to 1.0 to disable the safety cap. Ignored when psutil is not available.
        max_queue_size (`int`, *optional*, defaults to 0):
            Maximum request queue size for serving. 0 means unlimited.
        per_request_processors (`bool`, *optional*, defaults to `False`):
            Enable per-request logits processor parameters. Default is False.
        drop_unsupported_processors (`bool`, *optional*, defaults to `True`):
            Remove unsupported logits processors instead of erroring. Default is True.
        disable_nccl_graph_mixing (`bool`, *optional*, defaults to `True`):
            Disable NCCL's safety net for parallel graph-captured comms. Never happens in CB and gives TP a perf boost.
        cpu_group_timeout (`float`, *optional*, defaults to 300.0):
            The time (in seconds) after which a CPU communication will timeout and the process will crash. Leave to None
            for no timeout. Default is 300 seconds.
        use_default_compile_configs (`bool | None`, *optional*):
            Deprecated in 5.11: please use default_compile_level instead.
        max_cached_graphs (`int`, *optional*):
            Deprecated in 5.13: maximum number of graph is no longer an issue.
        block_size (`int | None`, *optional*):
            Deprecated in 5.17: now page_size is used instead.
    """

    # The number of tokens stored inside a (full attention) page. A block storing the cache of N layers has N pages, one
    # per layer. Since different page types can hold different number of tokens, this is for a full attention page.
    # Default is 256. Must be at least 4 (for an efficient cache, it should be well above that)
    page_size: int = 256

    # Number of blocks the cache contains. Usually better to leave it as None and be auto inferred.
    num_blocks: int | None = None

    # The maximum number of tokens in a batch. Once the page size is set, this can be auto inferred using GPU size.
    max_batch_tokens: int | None = None

    # The max percentage of free GPU memory (after the model is loaded) to use for the KV cache. If None, auto resolved
    # to 0.9 (no logit processing) or 0.8 (logit processing) to leave headroom for temporary tensors.
    max_memory_percent: float | None = None

    # The maximum number of requests in a batch. Helps limiting the memory footprint of the logits, which scale with the
    # vocabulary size.
    max_requests_per_batch: int | None = None

    # This is only used in the flash_attn_with_kvcache fast decode path to dimension the block table. If it is set to 0,
    # the fast decode path will not be used. Auto-inferred from GPU memory when `None` (default).
    max_blocks_per_request: int | None = None

    # Block sharing can only be allowed, but never forced: some model just do not support it. If you only have a few
    # short prompts, but long generation lengths, you might want to disable block sharing.
    allow_block_sharing: bool = True

    # Enables asynchronous batching. This removes the CPU overhead from the continuous batching loop, at the cost of
    # doubling the VRAM usage. If None, will be automatically detected.
    use_async_batching: bool | None = None

    # Enables cuda graphs. This can be a tuple of booleans (one for the varlen path and one for the decode fast path), a
    # boolean which will apply to both paths, or None (automatically inferred). After calling `decide_use_cuda_graphs`,
    # the attribute will ALWAYS be a tuple of booleans.
    use_cuda_graph: bool | tuple[bool, bool] | None = None

    # If any of these parameters are set to a non-default, CUDA graphs will be used. Otherwise we automatically infer
    # if they should be turned on. Padding interval sizes are in tokens and further explained in the docstring at the
    # top of the continuous_batching/continuous_api.py file.
    q_padding_interval_size: int = 0
    kv_padding_interval_size: int = 0

    # Compile configs for the two execution paths. If None, uses the compile_config from generation_config as fallback.
    varlen_compile_config: CompileConfig | None = None
    decode_compile_config: CompileConfig | None = None
    # Compile level for the executions path, if no compile config is provided for the path. Default is 0 (no compile).
    # Level 1: `mode=default, dynamic=True`
    # Level 2: `mode=max-autotune-no-cudagraphs, dynamic=True`
    # Level 3: `mode=max-autotune-no-cudagraphs, dynamic=False`
    default_compile_level: int = 0

    # Scheduler type. FIFO by default. For all types available, checks SCHEDULER_MAPPING in scheduler.py
    scheduler_type: str = "fifo"
    # Safety margin: if the number of free blocks falls below (safety_margin * num_blocks), then new prefill requests
    # will not be scheduled to prioritize decoding active requests. Defaults to None (use class default).
    safety_margin: float | None = None

    # Whether to generate log probabilities, which is the log of the softmax of the processed logits. If True, the log
    # probabilities will be returned along with the generated tokens in the generation output.
    return_logprobs: bool = False

    # An optional seed for generation. If not specified, the internal seed will be set to a random value.
    seed: int | None = None

    # CPU swap space in GiB for KV cache offloading. When the GPU cache is full and a request must be evicted, its KV
    # cache is copied to this pre-allocated pinned CPU buffer instead of being discarded. Default to 0.0 GiB. You can
    # also set this to None to dimension the pool using only the safety threshold, but this will error out if psutil is
    # not available.
    # TODO: use async transfer and move this to a non-zero value
    cpu_offload_space: float | None = 0.0
    # Safety cap: if cpu_offload_space exceeds this fraction of total system RAM, it is clamped. Set to 0.0 to disable
    # offloading.
    cpu_offload_space_safety_threshold: float = 0.8

    # The parameters below are mostly useful in the context of serving
    max_queue_size: int = 0

    # Enables per-request logits processor parameters. When enabled, each request can specify its own values (e.g.,
    # temperature) via logits_processor_kwargs. When disabled, all requests use the default values.
    per_request_processors: bool = False
    # When True, processors explicitly marked as unsupported are removed with a warning. When False, all processors
    # are kept but warnings are logged for unsupported/unknown ones.
    drop_unsupported_processors: bool = True

    # Disable NCCL's safety net for parallel graph-captured communications. This means it is no longer safe to replay a
    # CUDA graph with NCCL communication at the same time as 1. another CUDA graph with captured comms 2. an eager comm.
    # This is turned on by default because the above never happens in CB and this gives a nice perf boost.
    disable_nccl_graph_mixing: bool = True

    # The time (in seconds) after which a CPU communication will timeout and the process will crash. Leave to None for
    # no timeout. Default is 300 seconds. This exists because dist has a gloo timeout of 30 minutes, which is way too
    # long for almost all use cases.
    cpu_group_timeout: float | None = 300.0

    # Deprecated arguments
    use_default_compile_configs: bool | None = None
    max_cached_graphs: int | None = None
    block_size: int | None = None

    def __post_init__(self):
        # Convert dicts to CompileConfig objects
        if isinstance(self.varlen_compile_config, dict):
            self.varlen_compile_config = CompileConfig(**self.varlen_compile_config)
        if isinstance(self.decode_compile_config, dict):
            self.decode_compile_config = CompileConfig(**self.decode_compile_config)

        # Only turn off graph mixing support if TP is on
        graph_mixing_supported = os.environ.get("NCCL_GRAPH_MIXING_SUPPORT", "1") == "1"
        distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        if self.disable_nccl_graph_mixing and graph_mixing_supported and distributed:
            logger.warning(
                "Setting NCCL_GRAPH_MIXING_SUPPORT = 0 because disable_nccl_graph_mixing is True and WORLD_SIZE > 1."
            )
            os.environ.setdefault("NCCL_GRAPH_MIXING_SUPPORT", "0")

        # Warn about deprecated arguments
        if self.use_default_compile_configs is not None:  # Deprecated in 5.11
            if self.use_default_compile_configs:
                level_msg = "setting default_compile_level to 3. Consider using a lower level for faster warmup time."
                self.default_compile_level = 3
            else:
                level_msg = "setting default_compile_level to 0."
                self.default_compile_level = 0
            logger.warning(
                "use_default_compile_configs is deprecated: please use default_compile_level instead. For backwards "
                f"compatibility, {level_msg}"
            )
        if self.max_cached_graphs is not None:  # Deprecated in 5.13
            logger.warning(
                "max_cached_graphs is deprecated: maximum number of graph is no longer an issue. Deprecated in 5.13."
            )
        if self.block_size is not None:  # Deprecated in 5.17
            logger.warning(
                "block_size is deprecated: please use page_size instead. For backwards compatibility, block_size will "
                "be used as the full attention page size."
            )
            self.page_size = self.block_size

    @property
    def cuda_graph_booleans(self) -> tuple[bool, bool]:
        """The cuda graph booleans for the varlen and decode paths."""
        if self.use_cuda_graph is None:
            return False, False
        if isinstance(self.use_cuda_graph, bool):
            return self.use_cuda_graph, self.use_cuda_graph
        return self.use_cuda_graph

    @property
    def fallback_max_blocks_per_request(self) -> int:
        """Fallback if no user-hint is given and decode path is available."""
        return 32
