import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
import logging

if TYPE_CHECKING:
    import torch

    from ..models.demucs import BagOfModels


__all__ = [
    "AudioSeparation",
    "SeparationData"
]

@dataclass
class SeparationData:
    """"""
    vocal: "torch.Tensor"
    instrument: "torch.Tensor"
    samplerate: int


class AudioSeparation:
    """ Wrapper for audio separation, currently support demucs."""
    def __init__(
        self,
        model: "BagOfModels",
        device: str | None = None,
        shifts: int = 1,
        overlap: float = 0.25,
        split: bool = True,
        segment: int | None = None,
        jobs: int = 0,
        progress: bool = False,
        callback: Callable[[dict], None] | None = None,
        callback_arg: dict | None = None,
        logger: logging.Logger | None = None,
    ):
        self._model = model
        self._audio_channels = self._model.audio_channels
        self._samplerate = self._model.samplerate

        self._shifts = shifts
        self._overlap = overlap
        self._split = split
        self._segment = segment
        self._jobs = jobs
        self._progress = progress
        self._callback = callback
        self._callback_arg = callback_arg

        if not device:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        self.logger = logger
        if not logger:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
        
    def separate_tensor(
        self,
        wav: "torch.Tensor",
        sample_rate: int,
    ):
        if sample_rate != self._samplerate:
            pass

        from ..models.demucs.apply import _replace_dict, apply_model

        ref = wav.mean(0)
        mean = ref.mean()
        std = ref.std() + 1e-8

        out = apply_model(
            self._model,
            ((wav - mean) / std)[None],
            segment=self._segment,
            shifts=self._shifts,
            split=self._split,
            overlap=self._overlap,
            device=self._device,
            num_workers=self._jobs,
            callback=self._callback,
            callback_arg=_replace_dict(
                self._callback_arg, ("audio_length", wav.shape[1])
            ),
            progress=self._progress,
        )

        out = out * std + mean
        return (wav, dict(zip(self._model.sources, out[0])))

    def separate(
        self,
        audio_path: str | os.PathLike,
        mix_method: Literal["subtract", "add"] = "subtract"
    ):
        if mix_method not in ("subtract", "add",):
            raise ValueError(f"Invalid mix method: {mix_method}")
        
        import torch
        from iantirta.audio import load_audio

        wav: np.ndarray
        wav, sr = load_audio(
            audio_path,
            sampling_rate=self._samplerate,
            num_channels=self._audio_channels,
        )
        self.logger.debug(f'Successfully loading audio: sr={sr}, wav={wav.shape}',)
        wav: torch.Tensor = torch.from_numpy(wav)
        self.logger.debug(f'Successfully convert: wav={wav.shape}',)

        origin, res = self.separate_tensor(wav, sr)

        vocal = res.pop("vocals")

        if mix_method == "add":
            instrument = torch.zeros_like(next(iter(res.values())))
            for i in res.values():
                instrument += i

        elif mix_method == "subtract":
            instrument = origin - vocal

        return SeparationData(
            vocal=vocal,
            instrument=instrument,
            samplerate=sr
        )

if __name__ == "__main__":
    from pathlib import Path
    from timeit import timeit

    import librosa
    import numpy as np
    import torchaudio
    from iantirta.audio import load_audio
    from rich import inspect as i

    audio_path = Path("~/.cache/iantirta/downloaded_video/Bernadya - Kita Buat Menyenangkan (Official Lyric Video).wav").expanduser().resolve()
    assert audio_path.is_file(), str(audio_path)

    def audio_1():
        wav, sr = torchaudio.load(str(audio_path))
        return wav.numpy().astype(np.float32)

    def audio_2():
        wav, sr = load_audio(audio_path)
        return wav

    def audio_3():
        wav, sr = librosa.load(str(audio_path), sr=None, mono=None)
        return wav

    wav1 = audio_1()
    wav2 = audio_2()
    wav3 = audio_3()
    assert wav1.shape == wav2.shape == wav3.shape, f"Shapes differ: {wav1.shape}, {wav2.shape}, {wav3.shape}"

    # Use allclose to account for minor floating-point differences between decoders
    np.testing.assert_allclose(wav1, wav2, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(wav1, wav3, rtol=1e-3, atol=1e-4)
    print("✅ All audio outputs match successfully!")

    iterations = 200
    t3 = timeit(audio_3, number=iterations)
    t1 = timeit(audio_1, number=iterations)
    t2 = timeit(audio_2, number=iterations)

    print("torchaudio   ", t1)
    print("New          ", t2)
    print("Librosa      ", t3)

    # i(waveform)
    # i(src_sampling_rate)