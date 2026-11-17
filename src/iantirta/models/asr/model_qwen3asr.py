from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import logging
import torch

from iantirta.models import TextTiming, WordTiming

from .base_asr_model import BaseASRModelMixin

if TYPE_CHECKING:
    import numpy as np


__all__ = ["Qwen3ASRModel"]


T = TypeVar("T")


class Qwen3ASRModel(BaseASRModelMixin):
    """
    Qwen Wrapper for ASR Model.
    Mostly use for transcribing only and not timestamp
    alignment.
    
    as the backend for alignment of QwenASR3,
    uses a decode head where it uses LLM auto complete or guessing,
    for the timestampt, and not actual emissions of the audio.
    """
    _force_aligner = None

    def __init__(
        self,
        *args,
        max_inference_batch_size: int | None = 4,
        num_beams: int = 1,
        max_new_tokens: int = 512,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.max_inference_batch_size = max_inference_batch_size
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

        self.logger.info(
            "Running QwenASR with:\n"
            f"  max_inference_batch_size={self.max_inference_batch_size}\n"
            f"  num_beams={self.num_beams}\n"
            f"  max_new_tokens={self.max_new_tokens}\n"
        )

    @property
    def force_aligner(self):
        if self._force_aligner is None:
            from .model_ctc_wav2vec2 import Wav2Vec2CTCModel
            self._force_aligner = Wav2Vec2CTCModel(
                "facebook/mms-1b-all",
                device=self.device
            )
        return self._force_aligner

    def fix_word_repetation(
        self,
        text: str,
        threshold: int = 3,
        max_words: int = 10,
        *,
        logger: logging.Logger | None = None,
    ) -> str:
        """
        Detects and collapses consecutive repeated words or multi-word phrases
        while preserving legitimate non-consecutive word reuse.
        """
        words = text.split()
        n = len(words)
        if n < threshold:
            return text
        norm = [w.lower() for w in words]
        result = []
        i = 0
        collapse_events = 0

        while i < n:
            found = False
            for k in range(1, max_words + 1):
                if i + k * threshold > n:
                    break

                pattern = norm[i : i + k]
                if all(norm[i + rep * k : i + (rep + 1) * k] == pattern for rep in range(1, threshold)):
                    # Keep the first occurrence with original casing
                    kept_tokens = words[i : i + k]
                    result.extend(kept_tokens)

                    # Skip all full consecutive repetitions
                    occurrences = threshold
                    end_idx = i + threshold * k
                    while end_idx + k <= n and norm[end_idx : end_idx + k] == pattern:
                        end_idx += k
                        occurrences += 1

                    # Consume incomplete trailing cut-off at the very end of the text
                    # e.g. "... jangan takut jangan" -> stops cleanly at "jangan takut"
                    rem = n - end_idx
                    trailing_tokens = []
                    if 0 < rem < k and norm[end_idx:] == pattern[:rem]:
                        trailing_tokens = words[end_idx:]
                        end_idx = n

                    collapse_events += 1
                    phrase = " ".join(kept_tokens)
                    skipped_repeats = occurrences - 1
                    dropped_tokens = (end_idx - i) - k
                    (logger or self.logger).debug(
                        "Collapse #%d at word index %d..%d | phrase=%r (%d-gram) | "
                        "total_occurrences=%d (kept=1, skipped_repeats=%d, trailing_cutoff=%r, dropped_tokens=%d)",
                        collapse_events,
                        i,
                        end_idx - 1,
                        phrase,
                        k,
                        occurrences,
                        skipped_repeats,
                        " ".join(trailing_tokens) if trailing_tokens else None,
                        dropped_tokens,
                    )

                    i = end_idx
                    found = True
                    break

            if not found:
                result.append(words[i])
                i += 1

        (logger or self.logger).debug(
            "Repetition cleanup finished: %d collapse(s) | words %d -> %d",
            collapse_events, n, len(result),
        )
        return " ".join(result)

    # ------------------
    # Inference Method
    # ------------------

    def inputs(
        self,
        wavs: list[np.ndarray],
        languages: list[str | None],
        prompts: list[str | None],
    ):
        """
        Method that used to generate inputs for model generation.
        """
        return self.processor.apply_transcription_request(
            audio=wavs,
            language=languages,
            prompt=prompts,
            processor_kwargs={
                "padding": True,
                "sampling_rate": 16000,
                "return_tensors": "pt"
            }
        ).to(
            device=self.model.device,
            dtype=self.model.dtype
        )

    @torch.no_grad()
    def generate(self, inputs):
        """
        Method that used to output logits by ``model.generate`` func.
        """
        return self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            do_sample=False,
        )

    @torch.inference_mode()
    def _transcribe(
        self,
        wavs: list[np.ndarray],
        languages: list[str | None],
        references: list[str | None],
        *,
        return_timestamps: bool = True,
        logger: logging.Logger | None = None,
    ) -> list[TextTiming]:
        """
        Actually transcribe given audio
        """

        batch_size = self.max_inference_batch_size
        if batch_size is None or batch_size < 0:
            batch_size = len(wavs)

        outs: list[str] = []
        for i in range(0, len(wavs), batch_size):
            sub_wavs = wavs[i: i + batch_size]
            sub_languages = languages[i: i + batch_size]
            sub_references = references[i: i + batch_size]

            inputs = self.inputs(
                wavs=sub_wavs,
                languages=sub_languages,
                prompts=sub_references
            )
            outputs = self.generate(inputs)

            decoded = self.processor.decode(
                outputs[:, inputs["input_ids"].shape[1]:],
                return_format="transcription_only",         # ["raw", "parsed", "transcription_only"]
                skip_special_tokens=None,                   # True if `return_format` != `"raw"`
            )
            outs.extend(list(decoded))

            del inputs, outputs, decoded

        assert len(outs) == len(languages)

        results: list[TextTiming] = []
        for text, lang in zip(outs, languages):
            text = self.fix_word_repetation(text, logger=logger)
            words = []
            for word in text.split():
                words.append(WordTiming(
                    start=None,
                    end=None,
                    score=None,
                    word=word
                ))
            results.append(TextTiming(
                words=words, language=lang
            ))

        (logger or self.logger).debug(
            "Transcribe Result: %s",
            results
        )
        if return_timestamps:
            align_results = self._align(
                wavs=wavs,
                references=[res.clean for res in results],
                languages=languages
            )
            results = self.force_aligner.populate_timestamp(
                wavs=wavs,
                align_results=align_results,
                result=results
            )
        return results

    # Override
    def _align(
        self,
        wavs: list[np.ndarray] | None = None,
        languages: list[str] | None = None,
        references: list[str] | None = None,
    ) -> list[tuple[int, list]]:
        """
        Actually align given audio
        """
        return self.force_aligner._align(
            wavs=wavs,
            references=references,
            languages=languages
        )

    def detect_text_lang(self, ref: str) -> str:
        """
        Wrapper method for detect text languages,
        so that each model with its own unique lang, can
        override the result.
        """
        res = super().detect_text_lang(ref)
        if res in ("zh-cn", "zh-tw"):
            res = "zh"
        return res

    def load_model(self):
        super().load_model()

        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_name_or_path,
            device_map=self.device,
            dtype=self.dtype,
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            device_map=self.device,
        )
