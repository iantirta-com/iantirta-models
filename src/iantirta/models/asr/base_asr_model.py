from __future__ import annotations

import logging
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, TypeVar

from iantirta.audio import AudioSegment, slice_audio

from iantirta.models import BaseModelMixin, TextTiming

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "ASRResult",
    "BaseASRModelMixin",
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

    return lang.lang


def ensure_list(data):
    if not isinstance(data, list):
        return [data]
    return data


RE_CJK = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+')


def sec2ass(s: float) -> str:
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f'{h:0>1.0f}:{m:0>2.0f}:{s:0>2.2f}'


@dataclass(slots=True)
class ASRResult:
    """
    Class used to represent ASR Result from align or transcribe.
    """
    texts: list[TextTiming]

    def to_line_idx(self, reference: str):
        # import iantirta.text as text_utils
        # reference = normalizekaldi(reference)
        lines = [
            line.strip()
            for line in reference.split("\n")
            if line.strip() and
            not line.startswith('[')
        ]
        clusters, i = [], 0
        n = [
            len(line.split())
            for line in lines
        ]
        words = [
            w
            for text in self.texts
            for w in text.words
        ]
        assert len(words) == sum(n)
        for l in n:
            clusters.append(words[i: i + l])
            i += l
        assert len(clusters) == len(lines)
        new_group_texts = [
            TextTiming(words=cluster)
            for cluster in clusters
        ]
        self.texts = new_group_texts
        return self

    def populate_ass(self):
        n_segments = len(self.texts)

        # Text Gap
        gaps = []
        for i in range(n_segments - 1):
            next_start = float(round(self.texts[i+1].start, 2))
            current_end = float(round(self.texts[i].end, 2))
            gap = round(next_start - current_end, 2)
            gaps.append(gap)
        assert len(gaps) == n_segments - 1

        # Populating Gaps
        sliced_gaps = []
        for gap in gaps:
            start_gap = round(gap * 0.3, 2)
            end_gap = round(gap - start_gap, 2)
            sliced_gaps.append((start_gap, end_gap))
        
        for i, text in enumerate(self.texts):
            if i == n_segments - 1:
                pad_end = 1.0 # just add 1.0s
            else:
                pad_end = min(sliced_gaps[i][0], 1.0)
            padded_end = round(text.end + pad_end, 2)
            if i == 0:
                pad_start = 0.8
            else:
                pad_start = min(sliced_gaps[i-1][1], 0.8)
            padded_start = max(0.0, round(text.start - pad_start, 2))

            # Effect
            fade_in_ms = max(0, min(300, int((text.start - padded_start) * 1000))) # 300ms max to show text before highlighting
            fade_out_ms = max(0, min(300, int((padded_end - text.end) * 1000))) # 300ms max to hide text before next word

            # TODO: Find better way
            is_cjk = bool(RE_CJK.search(text.text))
            style = "CJK_Duet" if is_cjk else "Lat_Duet"
            
            # Karaoke Content
            k_tokens = []
            total_word_dur = 0
            for j, word in enumerate(text.words):
                dur_cs = max(0.0, round(word.duration * 100))
                tokens = f"{{\\kf{dur_cs}}}{word.word.strip()} "
                if j == 0:
                    prev_gap = word.start - padded_start
                else:
                    prev_gap = word.start - text.words[j-1].end

                dur_start_cs = max(0.0, round(prev_gap * 100))
                tokens = f"{{\\kf{dur_start_cs}}}" + tokens
                total_dur = dur_cs + dur_start_cs

                if j == len(text.words) - 1:
                    next_gap = padded_end - word.end
                    dur_end_cs = max(0.0, round(next_gap * 100))
                    tokens = tokens + f"{{\\kf{dur_end_cs}}}"
                    total_dur += dur_end_cs
                
                k_tokens.append(tokens)
                total_word_dur += total_dur
            assert total_word_dur == round((padded_end - padded_start) * 100)

            karaoke_content = "".join(k_tokens)
            text.ass_event = (
                f"Dialogue: 0,{sec2ass(padded_start)},{sec2ass(padded_end)},"
                f"{style},,0,0,0,,"
                f"{{\\fad({fade_in_ms},{fade_out_ms})}}"
                f"{{\\an2}}{karaoke_content}"
            )

        return self

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

    def detect_text_lang(self, ref: str) -> str:
        """
        Wrapper method for detect text languages,
        so that each model with its own unique lang, can
        override the result.
        """
        return detect_text_language(ref)

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
            global_detected_lang = self.detect_text_lang(reference)
        references = ensure_list(references)

        languages = ensure_list(languages)

        languages = languages + [None] * (len(audio_segments) - len(languages))
        assert len(languages) == len(audio_segments), (
            "Given languages length missmatch "
            f"{len(languages)} == {len(audio_segments)}"
        )
        references = references + [None] * (len(audio_segments) - len(references))
        assert len(references) == len(audio_segments), (
            "Given references length missmatch "
            f"{len(references)} == {len(audio_segments)}"
        )

        offsets, langs, wavs = [], [], []

        for aseg, lang, ref in zip(audio_segments, languages, references):
            wavs.append(slice_audio(wav, aseg.start, aseg.end, self.samplerate))
            offsets.append(aseg.start)
            langs.append(
                lang
                if lang not in (None, "auto")
                else global_detected_lang
                if global_detected_lang
                else self.detect_text_lang(ref)
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

        wav = wav.squeeze()
        (logger or self.logger).debug(f"Squeezing audio: wav={wav.shape}")

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

        wav = wav.squeeze()
        (logger or self.logger).debug(f"Squeezing audio: wav={wav.shape}")

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
