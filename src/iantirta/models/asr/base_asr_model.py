from __future__ import annotations

import logging
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from iantirta.audio import AudioSegment, slice_np

from iantirta.models import BaseModelMixin, TextTiming

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "BaseASRModelMixin",
    "ASRResult",
]


_logger = logging.getLogger(__name__)


T = TypeVar("T")


def detect_text_language(text: str, *, logger: logging.Logger | None = None):
    try:
        import langdetect
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Cannot continue as ``langdetect`` is not installed.")

    text_langs = langdetect.detect_langs(text)
    lang = max(text_langs, key=lambda l: l.prob)
    (logger or _logger).debug(f"New End Detected Language: `{lang}`")

    return lang


def ensure_list(data):
    if not isinstance(data, list):
        return [data]
    return data


@dataclass(slots=True)
class ASRResult:
    """
    Class used to represent ASR Result from align or transcribe.
    """


class BaseASRModelMixin(BaseModelMixin):
    """
    Wrapper class mixin for ASR Based modeling
    """
    processor = None

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        import torch

        self.samplerate = 16000
        self.audio_channels = 1

        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name_or_path = model_name_or_path

        self.load_model()

    # ------------------
    # Helper Method
    # ------------------

    def load_audio(self, audio_path: str | Path) -> tuple[np.ndarray, int]:
        from iantirta.audio import load_audio

        return load_audio(
            audio_path,
            sampling_rate=self.samplerate,
            num_channels=self.audio_channels,
        )

    def group_by_language(
        self,
        wavs: list[np.ndarray],
        languages: list[str | None]
    ) -> dict[str | None, list[tuple[int, np.ndarray]]]:
        """
        Group audio wavs, into each of languages, None included.
        """
        groups = defaultdict(list)
        assert len(wavs) == len(languages), f"Audios and languages list length missmatch {len(wavs)} == {len(languages)}"

        for i, (audio, lang) in enumerate(zip(wavs, languages)):
            groups[lang].append((i, audio))

        return groups

    def prepare_infer_data(
        self,
        wav: np.ndarray,
        audio_segments: AudioSegment | list[AudioSegment] | None = None,
        references: str | list[str] | None = None,
        languages: str | list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Method used to prepare inference. 
        """

        if not audio_segments:
            duration = len(wav) / self.samplerate
            audio_segments = [AudioSegment(start=0.0, end=duration)]

        reference = (
            references
            if isinstance(references, str)
            else references[0] if isinstance(references, list)
            and len(references) == 1
            else None
        )

        global_detected_lang = None
        if reference is not None:
            global_detected_lang = detect_text_language(reference)
        references = ensure_list(references)

        offsets, langs, wavs = [], [], []

        for aseg, lang, ref in zip(audio_segments, languages, references):
            wavs.append(slice_np(wav, aseg.start, aseg.end, self.samplerate))
            offsets.append(aseg.start)
            langs.append(
                lang
                if lang not in (None, "auto")
                else global_detected_lang
                if global_detected_lang
                else detect_text_language(ref)
                if ref is not None
                else None
            )

            return {
                "wavs": wavs,
                "offsets": offsets,
                "languages": langs,
                "references": references,
            }

    # ------------------
    # Inference Method
    # ------------------

    @abstractmethod
    def inputs(self, wavs: list[np.ndarray]):
        """
        Method that used to generate inputs for model generation.
        """
        raise NotImplementedError()

    @abstractmethod
    def generate(self, inputs):
        """
        Method that used to output logits by ``model.generate`` func.
        """
        raise NotImplementedError()

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

    def _align(
        self,
        wavs: list[np.ndarray] | None = None,
        languages: list[str] | None = None,
        references: list[str] | None = None,
    ) -> list[tuple[int, list]]:
        """
        Actually align given audio
        """
        raise NotImplementedError()

    # ------------------
    # API method
    # ------------------

    def transcribe(
        self,
        audio_path: str | Path,
        audio_segments: AudioSegment | list[AudioSegment] | None = None,
        references: str | list[str] | None = None,
        languages: str | list[str] | None = None,
        *,
        return_timestamps: bool = True,
        logger: logging.Logger | None = None
    ) -> ASRResult:
        """
        Main method for transcribe
        """

        wav: np.ndarray
        wav, sr = self.load_audio(audio_path=audio_path)
        (logger or self.logger).debug(f'Successfully loading audio: sr={sr}, wav={wav.shape}, audio_path={audio_path}',)

        infer_data = self.prepare_infer_data(
            wav=wav,
            audio_segments=audio_segments,
            references=references,
            languages=languages,
        )
        offsets = infer_data.pop("offsets")

        results = self._transcribe(
            **infer_data,
            return_timestamps=return_timestamps,
        )

        assert len(results) == len(offsets), f"produced asr result length missmatch, {len(results)} == len{offsets}"

        for asr_text, offset in zip(results, offsets):
            for word in asr_text.words:
                word.start = word.start + offset if word.start is not None else offset
                word.end = word.end + offset if word.end is not None else offset

        return ASRResult(texts=results)

    def align(
        self,
        *,
        audio_path: str | Path,
        audio_segments: AudioSegment | list[AudioSegment],
        hypothesis: list[TextTiming],
        languages: str | list[str] | None = None,
        logger: logging.Logger | None = None
    ) -> ASRResult:
        """
        Main method for force alignment
        """

        wav: np.ndarray
        wav, sr = self.load_audio(audio_path=audio_path)
        (logger or self.logger).debug(f'Successfully loading audio: sr={sr}, wav={wav.shape}, audio_path={audio_path}',)

        infer_data = self.prepare_infer_data(
            wav=wav,
            audio_segments=audio_segments,
            references=[text.clean for text in hypothesis],
            languages=languages,
        )
        offsets = infer_data.pop("offsets")

        results = self._align(
            **infer_data,
        )

        assert len(results) == len(offsets), f"produced asr result length missmatch, {len(results)} == len{offsets}"

        hypothesis = self.populate_timestamp(
            wavs=infer_data['wavs'],
            align_results=results,
            result=hypothesis
        )

        for i, (offset, lang) in enumerate(zip(
            offsets, infer_data['languages']
        )):
            for word in hypothesis[i].words:
                word.start=word.start + offset
                word.end=word.end + offset

            if (ori_lang:=hypothesis[i].language) != lang:
                (logger or self.logger).warning(f"Alignment result Language is different: ori {ori_lang} != {lang}")

        return ASRResult(texts=results)

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
        raise NotImplementedError()
