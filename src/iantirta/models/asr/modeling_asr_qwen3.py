from .modeling_asr import ASRModel
from iantirta.models import TextTiming, WordTiming
import numpy as np
import torch

class Qwen3ASR(ASRModel):
    """ Alibaba Qwen ASR Model Class. """
    force_align_model_id_or_path: str = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
    
    def _load_model(self, model_name_or_path, **kwargs) -> None:
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
        )

        merged_kwargs, kwargs = merge_kwargs(QwenASRKwargs, kwargs)
        self.config = QwenASRKwargs(merged_kwargs)
        self.max_batch_size = self.config["common_kwargs"].pop("max_inference_batch_size", 32)
        
        self.model = AutoModelForMultimodalLM.from_pretrained(model_name_or_path, **self.config["common_kwargs"]).to(env.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, **self.config["common_kwargs"])

        self.force_aligner = None # Instance later if needed.

        if kwargs:
            logger.warning(f"Unused kwargs in {type(self).__name__}: {kwargs}")

    def detect_language(self, audionp: AudioNumpy, *, seek: float, reference: str) -> str:
        lang = super().detect_language(audionp, seek=seek, reference=reference)
        return QWEN_LANGUAGES.get(lang, "English")
    
    def populate_timestamp(
        self,
        audio_lists: list[np.ndarray],
        align_results: list[tuple[int, list]],
        result: list[TextTiming]
    ) -> list[TextTiming]:
        return self.force_aligner.populate_timestamp(audio_lists, align_results, result)

    # ------------------
    # Inference Method
    # ------------------

    def inputs(
        self,
        wavs: list[np.ndarray],
        langs: list[str | None],
        prompts: list[str | None],
    ):
        return self.processor.apply_transcription_request(
            audio=wavs,
            language=langs,
            prompt=prompts,
            processor_kwargs=self.config["processor_kwargs"]
        ).to(device=self.model.device, dtype=self.model.dtype)

    @torch.no_grad()
    def _infer(self, inputs):
        return self.model.generate(
            **inputs,
            **self.config["generation_kwargs"]
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

        batch_size = self.max_batch_size
        if batch_size is None or batch_size < 0:
            batch_size = len(wavs)

        outs: list[str] = []
        for i in range(0, len(wavs), batch_size):
            sub_wavs = wavs[i: i + batch_size]
            sub_languages = languages[i: i + batch_size]
            sub_references = references[i: i + batch_size]
    
            inputs = self.inputs(
                sub_wavs,
                sub_languages,
                prompts=sub_references
            )

            outputs = self._infer(inputs)

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
                words.append(WordTiming(start=None, end=None, score=None, word=word))
            results.append(TextTiming(words=words, language=lang))

        if return_timestamps:
            align_results = self._align(
                wavs=wavs,
                transcripts=[res.text for res in results],
                languages=languages
            )
            results = self.force_aligner.populate_timestamp(wavs, align_results, results)

        return results

    def _align(
        self,
        wavs: list[np.ndarray] | None = None,
        transcripts: list[str] | None = None,
        languages: list[str] | None = None,
    ) -> list[tuple[int, list]]:

        with torch.inference_mode():
            if self.force_aligner is None:
                self.force_aligner = Wav2Vec2("facebook/mms-1b-all", **self.config["common_kwargs"])
            new_languages = [None] * len(languages)
            for i, lang in enumerate(languages):
                lang = REVERSE_QWEN_LANGUAGES.get(lang, "en")
                new_languages[i] = MMS_LANGS.get(lang) or "eng"
            out = self.force_aligner._align(
                audios, transcripts, new_languages
            )
        return out
