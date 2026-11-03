
import logging
from pathlib import Path

import numpy as np
import torch
from iantirta.audio import AudioSegment

__all__ = [
    "ASRModel"
]

class ASRResult:
    pass

class FAResult:
    pass


class ASRModel:
    """Wrapper for automatic speech recognition."""

    samplerate: int = 16000
    audio_channels: int = 1

    def __init__(
        self,
        model,
        processor,
        logger: logging.Logger | None = None
    ):
        self.model = model
        self.processor = processor

        self.logger = logger

    # ------------------
    # Helper method
    # ------------------

    def _prepare_asr_data(
        self,
        wav: np.ndarray,
        audio_segments: list[AudioSegment],
        reference: str,
        language: str | None,
    ):
        if not isinstance(audio_segments, list):
            raise TypeError('``audio_segments`` must be a list')

        references = reference
        if not isinstance(references, list):
            references = [references]
        references = references + [None] * (
            len(audio_segments) - len(references)
        )
    
        languages = language
        if not isinstance(languages, list):
            languages = [languages]
        languages = languages + [None] * (
            len(audio_segments) - len(languages)
        )

        wavs: list[np.ndarray] = []
        offsets: list[int] = []
        langs: list[str] = []
        detected_text_lang = detect_text_lang(reference)

        for aseg, lang, ref in zip(audio_segments, languages, references):
            wavs.append(
                slice_np(
                    wav,
                    aseg.start,
                    aseg.end,
                    self.sr
                )
            )
            offsets.append(aseg.start)
            langs.append(
                lang
                if lang not in (None, "auto")
                else detected_text_lang
            )

        return {
            'wavs': wavs,
            'offsets': offsets,
            'languages': langs,
            'references': references
        }

    # ------------------
    # Inference Method
    # ------------------

    @torch.inference_mode()
    def _transcribe(
        self,
        wavs: list[np.ndarray],
        languages: list[str | None],
        references: list[str | None],
    ):
        """
        Actually transcribe given wavs input
        """
        batch_size = self.max_batch_size
        if batch_size is None or batch_size < 0:
            batch_size = len(wavs)

        outs: list[str] = []
        for i in range(0, len(wavs), batch_size):
            sub_wavs = wavs[i: i + batch_size]
            sub_languages = languages[i: i + batch_size]
            sub_references = references[i: i + batch_size]

            inputs = self.processor.apply_transcription_request(
                audio=sub_wavs,
                language=sub_languages,
                prompt=sub_references,
                processor_kwargs=self.config["processor_kwargs"]
            ).to(device=self.model.device, dtype=self.model.dtype)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.config["generation_kwargs"]
                )

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

        # Force Alignment
        align_results = self._align(
            wavs=wavs,
            transcripts=[res.text for res in results],
            languages=languages
        )
        results = self.force_aligner.populate_timestamp(wavs, align_results, results)

        return results

    @torch.inference_mode()
    def _align(
        self,
        wavs: list[np.ndarray],
        languages: list[str | None],
        references: list[str | None],
    ):
        pass

    # ------------------
    # API method
    # ------------------

    def transcribe(
        self,
        audio_path: str | Path,
        audio_segments: list[AudioSegment],
        reference: str,
        language: str | None,
        *,
        logger: logging.Logger | None = None
    ) -> ASRResult:
        from iantirta.audio import load_audio

        wav: np.ndarray
        wav, sr = load_audio(
            audio_path,
            sampling_rate=self.samplerate,
            num_channels=self.audio_channels,
        )
        (logger or self.logger).debug(f'Successfully loading audio: sr={sr}, wav={wav.shape}, audio_path={audio_path}',)

        metadata = self._prepare_asr_data(
            wav=wav,
            audio_segments=audio_segments,
            reference=reference,
            language=language
        )
        (logger or self.logger).debug(f"Prepared ASR Metadata:\n{metadata}")
        offsets = metadata.pop('offsets')

        with torch.inference_mode():
            results = self._transcribe(**metadata)

        assert len(results) == len(offsets), f"produced asr result length missmatch, {len(results)} == len{offsets}"

        for asr_text, offset in zip(results, offsets):
            for word in asr_text.words:
                word.start = word.start + offset if word.start is not None else offset
                word.end = word.end + offset if word.end is not None else offset

        return ASRResult(texts=results)

    def align(
        self,
        audio_path: str | Path,
        audio_segments: list[AudioSegment],
        reference: str,
        language: str | None,
        *,
        logger: logging.Logger | None = None
    ):
        from iantirta.audio import load_audio
        wav: np.ndarray
        wav, sr = load_audio(
            audio_path,
            sampling_rate=self.samplerate,
            num_channels=self.audio_channels,
        )
        (logger or self.logger).debug(f'Successfully loading audio: sr={sr}, wav={wav.shape}, audio_path={audio_path}',)
        
        with torch.inference_mode():
            results = self._align(**metadata)

        assert len(results) == len(offsets), f"produced asr result length missmatch, {len(results)} == len{offsets}"

        for i, (offset, lang) in enumerate(zip(offsets, langs)):
            for word in hypothesis[i].words:
                word.start=word.start + offset
                word.end=word.end + offset
            if (ori_lang:=hypothesis[i].language) != lang:
                logger.warning(f"Alignment result Language is different: ori {ori_lang} != {lang}")

        return ASRResult(texts=hypothesis)
