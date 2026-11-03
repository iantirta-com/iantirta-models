
from abc import abstractmethod
from collections import defaultdict
import logging
from pathlib import Path

from iantirta.audio import AudioSegment, slice_np
import numpy as np
import torch

from iantirta.models import TextTiming, WordTiming


def ensure_list(data):
    if not isinstance(data, list):
        return [data]
    return data


class ASRResult:
    pass

class FAResult:
    pass

class ASRModel:
    """ Base HF ASR Model Class. """
    model = None
    processor = None
    
    def __init__(
        self, 
        model_name_or_path: str,
        *,
        logger: logging.Logger | None = None,
        **kwargs
    ) -> None:
        self.samplerate = 16000
        self.audio_channels = 1

        self.lid_model = None
        self._load_model(model_name_or_path, **kwargs)

        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def _load_model(self, model_name_or_path: str, **kwargs) -> None:
        raise NotImplementedError()

    def detect_language(self, audionp: AudioNumpy, *, seek: float, reference: str) -> str:
        if not self.lid_model:
            from kplus.pipelines.asr import BaseASR
            self.lid_model = BaseASR.from_model(whisper="large-v3", extra_models=[])
        if reference:
            langdetect = env.langdetect
            audio_langs: list[tuple[str, float]] = self.lid_model.detect_language(audionp, seek=seek, return_probs=True)
            text_langs = langdetect.detect_langs(reference)
            audio_langs_map = {lang: prob for lang, prob in audio_langs}
            for lang in text_langs:
                audio_langs_map[lang.lang] = audio_langs_map.get(lang.lang, 0.0) + lang.prob
            lang = max(audio_langs_map, key=audio_langs_map.get)
            logger.debug(f"New End Detected Language: `{lang}`")
            return lang
        else:
            return self.lid_model.detect_language(audionp, seek=seek)

    def populate_timestamp(
        self,
        audio_lists: list[np.ndarray],
        align_results: list[tuple[int, list]],
        result: list[TextTiming]
    ) -> list[TextTiming]:
        raise NotImplementedError()

    # ------------------
    # Helper Method
    # ------------------

    def group_by_language(
        self,
        wavs: list[np.ndarray],
        languages: list[str | None]
    ) -> dict[str | None, list[tuple[int, np.ndarray]]]:
        """
        Method to group wavs into each language
        """
        groups = defaultdict(list)
        assert len(wavs) == len(languages), (
            f"Audios and languages list length missmatch {len(wavs)} == {len(languages)}"
        )

        for i, (audio, lang) in enumerate(zip(wavs, languages)):
            groups[lang].append((i, audio))

        return groups

    def prepare_inference_data(
        self,
        wav: np.ndarray,
        audio_segments: AudioSegment | list[AudioSegment] | None = None,
        languages: str | list[str] | None = None,
        references: str | list[str] | None = None,
    ) -> tuple[list[np.ndarray], list, list, list]:
        """ Return type:: """

        if not audio_segments:
            duration = len(wav) / self.samplerate
            audio_segments = [AudioSegment(start=0.0, end=duration)]

        audio_segments = ensure_list(audio_segments)
        languages = ensure_list(languages)

        reference = (
            references
            if isinstance(references, str)
            else references[0] if isinstance(references, list)
            and len(references) == 1
            else None
        )
        global_detected_lang = None
        if reference is not None:
            global_detected_lang = detect_text_languages(reference)
        references = ensure_list(references)

        offsets, langs, wavs = [], [], []
        
        languages = languages + [None] * (len(audio_segments) - len(languages))
        assert len(languages) == len(audio_segments), f"Given languages length missmatch {len(languages)} == {len(audio_segments)}"

        references = references + [None] * (len(audio_segments) - len(references))
        assert len(references) == len(audio_segments), f"Given references length missmatch {len(references)} == {len(audio_segments)}"

        for aseg, lang, ref in zip(audio_segments, languages, references):
            wavs.append(slice_np(wav, aseg.start, aseg.end, self.samplerate))
            offsets.append(aseg.start)
            langs.append(
                lang
                if lang not in (None, "auto")
                else global_detected_lang if global_detected_lang
                else detect_text_language(ref) if ref is not None
                else None
            )

        return wavs, offsets, langs, references

    # ------------------
    # Inference Method
    # ------------------

    @abstractmethod
    def inputs(self, wavs: list[np.ndarray]):
        raise NotImplementedError()

    @abstractmethod
    def _infer(self, inputs):
        raise NotImplementedError()

    def _transcribe(self,
        wavs: list[np.ndarray],
        languages: list[str | None],
        references: list[str | None],
        *,
        return_timestamps: bool = True,
    ) -> list[TextTiming]:
        raise NotImplementedError()

    def _align(self,
        wavs: list[np.ndarray] | None = None,
        transcripts: list[str] | None = None,
        languages: list[str] | None = None,
    ) -> list[tuple[int, list]]:
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
        from iantirta.audio import load_audio

        wav: np.ndarray
        wav, sr = load_audio(
            audio_path,
            sampling_rate=self.samplerate,
            num_channels=self.audio_channels,
        )
        (logger or self.logger).debug(f'Successfully loading audio: sr={sr}, wav={wav.shape}, audio_path={audio_path}',)

        wavs, offsets, langs, references = self.prepare_inference_data(
            wav=wav,
            audio_segments=audio_segments,
            languages=languages,
            references=references,
        )

        with torch.inference_mode():
            results = self._transcribe(
                wavs=wavs,
                languages=langs,
                references=references,
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
        audio_path: str | Path,
        audio_segments: AudioSegment | list[AudioSegment],
        hypothesis: list[TextTiming],
        languages: str | list[str] | None = None,
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

        wavs, offsets, langs, texts = self.prepare_inference_data(
            wav,
            audio_segments,
            languages,
            references=[text.clean for text in hypothesis]
        )

        with torch.inference_mode():
            results = self._align(
                wavs=wavs,
                transcripts=texts,
                languages=langs,
            )

        assert len(results) == len(offsets), f"produced asr result length missmatch, {len(results)} == len{offsets}"

        hypothesis = self.populate_timestamp(wavs, results, hypothesis)

        for i, (offset, lang) in enumerate(zip(offsets, langs)):
            for word in hypothesis[i].words:
                word.start=word.start + offset
                word.end=word.end + offset

            if (ori_lang:=hypothesis[i].language) != lang:
                logger.warning(f"Alignment result Language is different: ori {ori_lang} != {lang}")

        return ASRResult(texts=hypothesis)
