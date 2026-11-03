
import numpy as np


__all__ = [
    "ASRModel"
]


@dataclass
class ASRResult:
    texts: list[TextTiming]


class ASRModel:
    """Wrapper for automatic speech recognition."""

    samplerate: int = 16000
    audio_channels: int = 1

    def __init__(
        self,
        logger: logging.Logger | None = None
    ):
        self.logger = logger

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
    # API method
    # ------------------

    def transcribe(
        self,
        audio_path: str | Path,
        audio_segments: list[AudioSegment],
        reference: str,
        language: str | None,
    ) -> ASRResult:
        from iantirta.audio import load_audio

        wav: np.ndarray
        wav, sr = load_audio(
            audio_path,
            sampling_rate=self.samplerate,
            num_channels=self.audio_channels,
        )

        metadata = self._prepare_asr_data(
            wav=wav,
            audio_segments=audio_segments,
            reference=reference,
            language=language
        )
        offsets = metadata.pop('offsets')

        with torch.inference_mode():
            results = self._transcribe(**metadata)

        return ASRResult(texts=results)