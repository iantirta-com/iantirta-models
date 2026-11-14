from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypeVar

import torch

from iantirta.models import TextTiming

from .base_asr_model import BaseASRModelMixin

if TYPE_CHECKING:
    import numpy as np

__all__ = ["Wav2Vec2CTCModel"]

T = TypeVar("T")


class Wav2Vec2CTCModel(BaseASRModelMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger.info(
            "Running Wav2Vec2CTC with:\n"
        )

    def detect_text_lang(self, ref: str) -> str:
        """
        Wrapper method for detect text languages,
        so that each model with its own unique lang, can
        override the result.
        """
        from .languages import BASE_LANGS_2_MMS
        res = super().detect_text_lang(ref)
        res = BASE_LANGS_2_MMS.get(res, None)
        return res

    def _normalize_languages(
        self,
        languages: list[str] | None = None,
        *,
        logger: logging.Logger | None = None
    ) -> list[str] | None:
        if languages is None:
            return None

        from .languages import BASE_LANGS_2_MMS

        new_langs = []
        for lang in languages:
            new_lang = (
                BASE_LANGS_2_MMS.get(lang, None) or "eng"
                if len(lang.split('-')[0]) == 2
                else lang
            )
            if lang != "eng" and new_lang == "eng":
                (logger or self.logger).debug(
                    f"Failed to get lang of: ``{lang}``. "
                    "Requesting falling back to ``eng``."
                )
            new_langs.append(new_lang)

        assert len(new_langs) == len(languages)

        return new_langs

    # ------------------
    # Inference Method
    # ------------------

    def inputs(self, wavs: list[np.ndarray]):
        """
        Method that used to generate inputs for model generation.
        """
        return self.processor(
            audio=wavs,
            sampling_rate=self.samplerate,
            return_tensors="pt",
            padding=True
        ).to(
            device=self.model.device,
            dtype=self.model.dtype
        )

    @torch.no_grad()
    def generate(self, inputs):
        """
        Method that used to output logits by ``model.generate`` func.
        """
        return self.model(**inputs)

    @torch.inference_mode()
    def _transcribe(
        self,
        wavs: list[np.ndarray],
        languages: list[str | None],
        references: list[str | None],
        *,
        return_timestamps: bool = True,
    ) -> list[TextTiming]:
        """
        Actually transcribe given audio
        """
        raise NotImplementedError()

    def _align_emissions(
        self,
        emissions: torch.Tensor,
        references: list[str] | None = None,
        *,
        logger: logging.Logger | None = None
    ) -> list[tuple[int, list[T]]]:
        import torchaudio.functional as F

        assert emissions.size(0) == len(references), (
            "Emission and ref length missmatch, "
            f"{emissions.size(0)} == {len(references)}"
        )

        # Mms model dictionary token is in lower char
        norm_func = str.lower if "mms" in self.model.name_or_path else str.upper
        norm_refs = [
            norm_func(ref)
            for ref in references
        ]

        # `torchaudio.functional` alignment support batch == 1 only.
        assert len(norm_refs) == emissions.size(0) # Size 0 is [batch]

        results = []
        for i, norm_ref in enumerate(norm_refs):

            token_ids = self.processor.tokenizer(
                norm_ref.replace(" ", ""),
                return_tensors="pt",
                padding=True,
            )["input_ids"].to(
                dtype=torch.long,
                device=self.model.device
            )

            local_emissions = emissions[i].unsqueeze(0)

            (logger or self.logger).debug(
                f"[{i}] Aligning Emissions {local_emissions.shape}\n"
                f"Target: {self.processor.batch_decode(token_ids)}"
            )
    
            alignments, scores = F.forced_align(
                local_emissions,
                token_ids,
                blank=self.processor.tokenizer.pad_token_id
            )

            scores = scores.exp()

            token_spans: list[T] = F.merge_tokens(
                alignments[0],
                scores[0],
                blank=self.processor.tokenizer.pad_token_id
            )

            # Flatten
            words = norm_ref.split()
            (logger or self.logger).debug(f"{words}, {norm_ref}")
            n = [len(w) for w in words]
            assert sum(n) == len(token_spans), f"Length Missmatch: Token {len(token_spans)} | Words {sum(n)}"

            j = 0
            word_spans: list[T] = []
            for l in n:
                word_spans.append(token_spans[j:j+l])
                j+=l

            num_frames = local_emissions.size(1)
            results.append((num_frames, word_spans)) # word_spans == list[TokenSpan(start, end, token, score)]

        # Returning list[list[tuple[int, TokenSpan]]]
        # since we can't calculate its timestamp without enough data
        return results

    def _align(
        self,
        wavs: list[np.ndarray] | None = None,
        languages: list[str] | None = None,
        references: list[str] | None = None,
        *,
        emissions: torch.Tensor | list[torch.Tensor] | None = None,
        logger: logging.Logger | None = None
    ) -> list[tuple[int, list]]:
        """
        Actually align given audio
        """
        # If `audios` is pass group by lang
        # Else adapter already loaded
        if wavs is not None:
            languages = self._normalize_languages(languages, logger=logger)
            audio_groups: dict[str | None, list[tuple[int, np.ndarray]]] = self.group_by_language(
                wavs, languages
            )

            results = [None] * sum(len(items) for items in audio_groups.values())

            for lang, items in audio_groups.items():
                indices, sub_wavs, sub_references = [], [], []
                for i, wav in items:
                    indices.append(i)
                    sub_wavs.append(wav)
                    sub_references.append(references[i])

                self.processor.tokenizer.set_target_lang(lang)
                self.model.load_adapter(lang)

                inputs = self.inputs(sub_wavs)
                outputs = self.generate(inputs)

                logits = outputs.logits
                emissions = logits.log_softmax(dim=-1)

                (logger or self.logger).debug(
                    f"Alignment aligning, {lang} of audio length, "
                    f"{len(sub_references)}"
                )
                group_results = self._align_emissions(
                    emissions=emissions,
                    references=sub_references,
                    logger=logger,
                )

                for i, global_i in enumerate(indices):
                    results[global_i] = group_results[i]

            return results

        elif emissions is not None:
            return self._align_emissions(
                emissions=emissions,
                references=references,
                logger=logger,
            )

        raise ValueError("Either `audios` or `emissions` must be provided")

    # Override

    def load_model(self):
        super().load_model()

        from transformers import AutoModelForCTC, AutoProcessor

        self.model = AutoModelForCTC.from_pretrained(
            self.model_name_or_path,
            device_map=self.device,
            dtype=self.dtype,
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            device_map=self.device,
        )

    def populate_timestamp(
        self,
        wavs: list[np.ndarray],
        align_results: list[tuple[int, list[T]]],
        result: list[TextTiming]
    ) -> list[TextTiming]:
        """
        This method is used to populate alignment result
        into the transcribe result.
        """

        def make_score(spans) -> float:
            return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)

        for i, (num_frames, word_spans) in enumerate(align_results):
            # _ = len(audio_lists[i]) / num_frames / self.sr
            # MMS FA Uses 20ms, todo find a better way to get it. since the stride ratio is not correct yet
            parse_timestamp = lambda t: t * 0.02

            assert len(word_spans) == len(result[i].words), (
                f"WordSpan Length: {len(word_spans)}\n"
                f"Actual Words Length: {len(result[i].words)}\n"
                f"WordSpan: {[''.join(self.processor.decode(span.token)) for spans in word_spans for span in spans]}\n"
                f"Actual Words: {[w.word for w in result[i].words]}\n"
            )

            for spans, word in zip(word_spans, result[i].words):
                word.start=parse_timestamp(spans[0].start)
                word.end=parse_timestamp(spans[-1].end)
                word.score=make_score(spans)

        return result
