import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.demucs import BagOfModels


__all__ = [
    "AudioSeparation"
]

class AudioSeparation:
    """ Wrapper for audio separation, currently support demucs."""
    def __init__(self, model: "BagOfModels"):
        self.model = model

    def separate(self, audio_path: str | os.PathLike):
        pass


if __name__ == "__main__":
    from pathlib import Path
    from timeit import timeit

    import librosa
    import numpy as np
    import torchaudio
    from rich import inspect as i

    from iantirta.models.audio_separation.audio_utils.load_audio import load_audio

    audio_path = Path(r"d:\IANTIRTACOM\dataset\mahalini.wav")
    assert audio_path.is_file()

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