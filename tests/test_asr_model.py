
import pytest

from iantirta.models import BaseASRModelMixin


class TestASRModel:
    @pytest.fixture(autouse=True)
    def setup_class_model(self):
        self.model = BaseASRModelMixin("unit_test")

        assert self.model.samplerate == 16000
        assert self.model.audio_channels == 1

    @pytest.fixture(autouse=True)
    def setup_class_vars(self, separated_path, setup_class_model):
        wav, sr = self.model.load_audio(audio_path=separated_path)
        assert self.model.samplerate == sr

        *shape, C, T = wav.shape
        assert C == self.model.audio_channels

        self.wav = wav

    def test_prepare_infer_data(self, audio_segments, lyrics):
        infer_data = self.model.prepare_infer_data(
            wav=self.wav,
            audio_segments=audio_segments,
            references=lyrics,
            languages=None,
        )
        field_names = [
            "wavs", "offsets",
            "languages", "references",
        ]
        for name in field_names:
            assert name in infer_data
            outvar = infer_data[name]
            assert isinstance(outvar, list)

            assert len(outvar) == len(audio_segments)
        
        print(infer_data)
        offsets = infer_data.pop("offsets")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        __file__
    ]))
