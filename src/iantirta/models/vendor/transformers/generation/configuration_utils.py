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

from typing import Any


class GenerationConfig:
    # no-format
    """
    Class that holds a configuration for a generation task. A `generate` call supports the following generation methods
    for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

        - *greedy decoding* if `num_beams=1` and `do_sample=False`
        - *multinomial sampling* if `num_beams=1` and `do_sample=True`
        - *beam-search decoding* if `num_beams>1` and `do_sample=False`
        - *beam-search multinomial sampling* if `num_beams>1` and `do_sample=True`
        - *assisted decoding* if `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`

    To learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).

    <Tip>

    A large number of these flags control the logits or the stopping criteria of the generation. Make sure you check
    the [generate-related classes](https://huggingface.co/docs/transformers/internal/generation_utils) for a full
    description of the possible manipulations, as well as examples of their usage.

    </Tip>

    Note: the configuration fields that are still `None` will be overridden by `GenerationConfig._get_default_generation_params()`
    during the generation loop. If you want to use different values for these fields, make sure to explicitly set them in the
    generation config.

    Args:
        > Parameters that control the length of the output

        max_length (`int`, *optional*):
            `max_new_tokens` is recommended for controlling how many tokens the model generates.
            `max_length` remains for backward compatibility.

        max_new_tokens (`int`, *optional*):
            The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        min_length (`int`, *optional*):
            The minimum length of the sequence to be generated. Corresponds to the length of the input prompt +
            `min_new_tokens`. Its effect is overridden by `min_new_tokens`, if also set.
        min_new_tokens (`int`, *optional*):
            The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        early_stopping (`bool` or `str`, *optional*):
            Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
            `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
            beam search algorithm).
        max_time (`float`, *optional*):
            The maximum amount of time you allow the computation to run for in seconds. generation will still finish
            the current pass after allocated time has been passed.
        stop_strings (`str` or `list[str]`, *optional*):
            A string or a list of strings that should terminate generation if the model outputs them.

        > Parameters that control the generation strategy used

        do_sample (`bool`):
            Whether or not to use sampling ; use greedy decoding otherwise.
        num_beams (`int`, *optional*):
            Number of beams for beam search. 1 means no beam search.
        use_mtp: (`bool`):
            Whether or not to use Multi-Token Prediction (MTP) if the model supports it.

        > Parameters that control the cache

        use_cache (`bool`):
            Whether or not the model should use the past last key/values attentions (if applicable to the model) to
            speed up decoding.
        cache_implementation (`str`, *optional*):
            Name of the cache class that will be instantiated in `generate`, for faster decoding. Possible values are:

            - `"dynamic"`: [`DynamicCache`]
            - `"static"`: [`StaticCache`]
            - `"offloaded"`: [`DynamicCache(offloaded=True)`]
            - `"offloaded_static"`: [`StaticCache(offloaded=True)`]
            - `"quantized"`: [`QuantizedCache`]

            If none is specified, we will use the default cache for the model (which is often [`DynamicCache`]). See
            our [cache documentation](https://huggingface.co/docs/transformers/en/kv_cache) for further information.
        cache_config (`dict`, *optional*, default to `None`):
            Arguments used in the key-value cache class can be passed in `cache_config`.
        max_cache_len (`int`, *optional*):
            Only used with static caches (`cache_implementation` set to `"static"` or `"offloaded_static"`).
            Pre-sizes the cache to this length instead of the current call's `max_length`. Set it once to the
            largest call you expect so that repeated `generate()` calls with a longer prompt or a larger
            `max_new_tokens` (up to this ceiling) reuse the same cache instead of triggering a reallocation and a
            `torch.compile` recompilation.

        > Parameters for manipulation of the model output logits

        temperature (`float`, *optional*):
            The value used to module the next token probabilities. This value is set in a model's `generation_config.json` file. If it isn't set, the default value is 1.0
        top_k (`int`, *optional*):
            The number of highest probability vocabulary tokens to keep for top-k-filtering. This value is set in a model's `generation_config.json` file. If it isn't set, the default value is 50.
        top_p (`float`, *optional*):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
            `top_p` or higher are kept for generation. This value is set in a model's `generation_config.json` file. If it isn't set, the default value is 1.0
        min_p (`float`, *optional*):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between 0 and 1. Typical values are in the 0.01-0.2 range, comparably selective as setting `top_p` in
            the 0.99-0.8 range (use the opposite of normal `top_p` values).
        top_h (`float`, *optional*):
            Entropy budget scaling factor, which controls how much of the distribution’s entropy is preserved when sampling.
            Must be a value between 0 and 1. At each step, tokens are sorted by probability, and the smallest prefix of tokens
            is kept whose *renormalized* entropy is less than or equal to `top_h` times the entropy of the full distribution.
            Smaller values (e.g., 0.2–0.5) lead to more focused, deterministic outputs, while values closer to 1.0 allow more
            randomness and diversity. Typical values are in the 0.3–0.6 range.
        typical_p (`float`, *optional*):
            Local typicality measures how similar the conditional probability of predicting a target token next is to
            the expected conditional probability of predicting a random token next, given the partial text already
            generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that
            add up to `typical_p` or higher are kept for generation. See [this
            paper](https://huggingface.co/papers/2202.00666) for more details.
        epsilon_cutoff (`float`, *optional*):
            If set to float strictly between 0 and 1, only tokens with a conditional probability greater than
            `epsilon_cutoff` will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the
            size of the model. See [Truncation Sampling as Language Model
            Desmoothing](https://huggingface.co/papers/2210.15191) for more details.
        eta_cutoff (`float`, *optional*):
            Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly between
            0 and 1, a token is only considered if it is greater than either `eta_cutoff` or `sqrt(eta_cutoff) *
            exp(-entropy(softmax(next_token_logits)))`. The latter term is intuitively the expected next token
            probability, scaled by `sqrt(eta_cutoff)`. In the paper, suggested values range from 3e-4 to 2e-3,
            depending on the size of the model. See [Truncation Sampling as Language Model
            Desmoothing](https://huggingface.co/papers/2210.15191) for more details.
        repetition_penalty (`float`, *optional*):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://huggingface.co/papers/1909.05858) for more details.
        encoder_repetition_penalty (`float`, *optional*):
            The parameter for encoder_repetition_penalty. An exponential penalty on sequences that are not in the
            original input. 1.0 means no penalty.
        length_penalty (`float`, *optional*):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences.
        no_repeat_ngram_size (`int`, *optional*):
            If set to int > 0, all ngrams of that size can only occur once.
        bad_words_ids (`list[list[int]]`, *optional*):
            List of list of token ids that are not allowed to be generated. Check
            [`~generation.NoBadWordsLogitsProcessor`] for further documentation and examples.
        renormalize_logits (`bool`):
            Whether to renormalize the logits after applying all the logits processors (including the custom
            ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the score logits
            are normalized but some logit processors break the normalization.
        forced_bos_token_id (`int`, *optional*, defaults to `model.config.forced_bos_token_id`):
            The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful for
            multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be the target
            language token.
        forced_eos_token_id (`int` or list[int]`, *optional*, defaults to `model.config.forced_eos_token_id`):
            The id of the token to force as the last generated token when `max_length` is reached. Optionally, use a
            list to set multiple *end-of-sequence* tokens.
        remove_invalid_values (`bool`):
            Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to crash.
            Note that using `remove_invalid_values` can slow down generation.
        exponential_decay_length_penalty (`tuple(int, float)`, *optional*):
            This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been
            generated. The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where
            penalty starts and `decay_factor` represents the factor of exponential decay
        suppress_tokens (`list[int]`, *optional*):
            A list of tokens that will be suppressed at generation. The `SuppressTokens` logit processor will set their
            log probs to `-inf` so that they are not sampled.
        begin_suppress_tokens  (`list[int]`, *optional*):
            A list of tokens that will be suppressed at the beginning of the generation. The `SuppressBeginTokens` logit
            processor will set their log probs to `-inf` so that they are not sampled.
        sequence_bias (`dict[tuple[int], float]`, *optional*)):
            Dictionary that maps a sequence of tokens to its bias term. Positive biases increase the odds of the
            sequence being selected, while negative biases do the opposite. Check
            [`~generation.SequenceBiasLogitsProcessor`] for further documentation and examples.
        token_healing (`bool`):
            Heal tail tokens of prompts by replacing them with their appropriate extensions.
            This enhances the quality of completions for prompts affected by greedy tokenization bias.
        guidance_scale (`float`, *optional*):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality.
        watermarking_config (`BaseWatermarkingConfig` or `dict`, *optional*):
            Arguments used to watermark the model outputs by adding a small bias to randomly selected set of "green"
            tokens. See the docs of [`SynthIDTextWatermarkingConfig`] and [`WatermarkingConfig`] for more
            details. If passed as `Dict`, it will be converted to a `WatermarkingConfig` internally.

        > Parameters that define the output variables of generate

        num_return_sequences (`int`, *optional*):
            The number of independently computed returned sequences for each element in the batch.
        output_attentions (`bool`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more details.
        output_hidden_states (`bool`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more details.
        output_scores (`bool`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        output_logits (`bool`):
            Whether or not to return the unprocessed prediction logit scores. See `logits` under returned tensors for
            more details.
        return_dict_in_generate (`bool`):
            Whether or not to return a [`~utils.ModelOutput`], as opposed to returning exclusively the generated
            sequence. This flag must be set to `True` to return the generation cache (when `use_cache` is `True`)
            or optional outputs (see flags starting with `output_`)

        > Special tokens that can be used at generation time

        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        bos_token_id (`int`, *optional*):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`Union[int, list[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

        > Generation parameters exclusive to encoder-decoder models

        encoder_no_repeat_ngram_size (`int`, *optional*):
            If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
            `decoder_input_ids`.
        decoder_start_token_id (`int` or `list[int]`, *optional*):
            If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token or a list of length
            `batch_size`. Indicating a list enables different start ids for each element in the batch
            (e.g. multilingual models with different target languages in one batch)

        > Generation parameters exclusive to assistant generation
        is_assistant (`bool`):
            Whether the model is an assistant (draft) model.
        num_assistant_tokens (`int`, *optional*):
            Defines the number of _speculative tokens_ that shall be generated by the assistant model before being
            checked by the target model at each iteration. Higher values for `num_assistant_tokens` make the generation
            more _speculative_ : If the assistant model is performant larger speed-ups can be reached, if the assistant
            model requires lots of corrections, lower speed-ups are reached.
        num_assistant_tokens_schedule (`str`, *optional*):
            Defines the schedule at which max assistant tokens shall be changed during inference.
            - `"heuristic"`: When all speculative tokens are correct, increase `num_assistant_tokens` by 2 else
              reduce by 1. `num_assistant_tokens` value is persistent over multiple generation calls with the same assistant model.
            - `"heuristic_transient"`: Same as `"heuristic"` but `num_assistant_tokens` is reset to its initial value after each generation call.
            - `"constant"`: `num_assistant_tokens` stays unchanged during generation
        assistant_confidence_threshold (`float`, *optional*):
            The confidence threshold for the assistant model. If the assistant model's confidence in its prediction for the current token is lower
            than this threshold, the assistant model stops the current token generation iteration, even if the number of _speculative tokens_
            (defined by `num_assistant_tokens`) is not yet reached. The assistant's confidence threshold is adjusted throughout the speculative iterations to reduce the number of unnecessary draft and target forward passes, biased towards avoiding false negatives.
            `assistant_confidence_threshold` value is persistent over multiple generation calls with the same assistant model.
            It is an unsupervised version of the dynamic speculation lookahead
            from Dynamic Speculation Lookahead Accelerates Speculative Decoding of Large Language Models <https://huggingface.co/papers/2405.04304>.
        prompt_lookup_num_tokens (`int`, *optional*):
            The number of tokens to be output as candidate tokens.
        max_matching_ngram_size (`int`, *optional*):
            The maximum ngram size to be considered for matching in the prompt. Default to 2 if not provided.
        assistant_early_exit(`int`, *optional*):
            If set to a positive integer, early exit of the model will be used as an assistant. Can only be used with
            models that support early exit (i.e. models where logits from intermediate layers can be interpreted by the LM head).
        assistant_lookbehind(`int`, *optional*):
            If set to a positive integer, the re-encodeing process will additionally consider the last `assistant_lookbehind` assistant tokens
            to correctly align tokens. Can only be used with different tokenizers in speculative decoding.
            See this [blog](https://huggingface.co/blog/universal_assisted_generation) for more details.
        target_lookbehind(`int`, *optional*):
            If set to a positive integer, the re-encodeing process will additionally consider the last `target_lookbehind` target tokens
            to correctly align tokens. Can only be used with different tokenizers in speculative decoding.
            See this [blog](https://huggingface.co/blog/universal_assisted_generation) for more details.
        assistant_ensemble_weight (`float`, *optional*):
            Enables static ensemble verification in speculative decoding. If set to a value in `(0.0, 1.0)`,
            the verifier accepts tokens against the mixture `w * p_target + (1 - w) * q_draft` instead of
            `p_target`, trading a controlled distributional bias for a higher acceptance rate. Defaults
            to `None`, which keeps decoding lossless. Requires the assistant model to return logits, so it
            is not compatible with prompt lookup decoding.
        speculation_type (`str`, *optional*):
            The requested speculation type. Accepted values are [`dflash`].

        > Parameters related to performances and compilation

        compile_config (CompileConfig, *optional*):
            If using a compilable cache, this controls how `generate` will `compile` the forward pass for faster
            inference.
        disable_compile (`bool`):
            Whether to disable the automatic compilation of the forward pass. Automatic compilation happens when
            specific criteria are met, including using a compilable cache. Please open an issue if you find the
            need to use this flag.
    """

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

