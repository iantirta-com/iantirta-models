from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

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
        num_beams: int = 4,
        max_new_tokens: int = 256,
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
