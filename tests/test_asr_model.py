
import pytest
import numpy as np

from unittest.mock import patch

from iantirta.models import BaseASRModelMixin, Qwen3ASRModel, Wav2Vec2CTCModel
from iantirta.models.asr.base_asr_model import detect_text_language


class TestASRModel:
    @pytest.fixture(autouse=True)
    def setup_class_model(self):
        self.model = BaseASRModelMixin("unit_test")

        assert self.model.samplerate == 16000
        assert self.model.audio_channels == 1

    @pytest.fixture
    def use_model_qwen(self):
        with patch.object(Qwen3ASRModel, "load_model"):
            self.model = Qwen3ASRModel(
                "unit_test_qwen"
            )
            yield

    @pytest.fixture
    def use_model_mms(self):
        with patch.object(Wav2Vec2CTCModel, "load_model"):
            self.model = Wav2Vec2CTCModel(
                "unit_test_mms"
            )
            yield

    @pytest.fixture(autouse=True)
    def setup_class_vars(self, separated_path, setup_class_model):
        wav, sr = self.model.load_audio(audio_path=separated_path)
        assert self.model.samplerate == sr

        *shape, C, T = wav.shape
        assert C == self.model.audio_channels

        self.wav = wav.squeeze()

    @pytest.mark.parametrize(
        "lyric_fname",
        ["indo_lyric", "mandarin_lyric"]
    )
    def test_prepare_infer_data(self, request, audio_segments, lyric_fname):
        """
        Test againts infer data.
        Using different lyrics works here.
        because this is only preparing data and lyric
        here still doesnt known its correlation with audio segment.
        """
        lyric = request.getfixturevalue(lyric_fname)

        infer_data = self.model.prepare_infer_data(
            wav=self.wav,
            audio_segments=audio_segments,
            references=lyric,
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

        assert len(offsets) == len(audio_segments)
        assert len(infer_data["wavs"]) == len(audio_segments)

    @pytest.mark.parametrize(
        "lyric_fname",
        ["indo_lyric", "mandarin_lyric"]
    )
    def test_detect_language(self, request, lyric_fname):
        lyric = request.getfixturevalue(lyric_fname)
        lang = detect_text_language(lyric)
        assert isinstance(lang, str)

    @pytest.mark.parametrize(
        "lyric_fname",
        ["indo_lyric", "mandarin_lyric"]
    )
    def test_qwen_detect_language_iso636(self, use_model_qwen, lyric_fname, request):
        lyric = request.getfixturevalue(lyric_fname)
        lang = self.model.detect_text_lang(lyric)

        assert isinstance(lang, str)
        assert len(lang) == 2  # 2 letter
        print(lang)

    @pytest.mark.parametrize(
        "lyric_fname",
        ["indo_lyric", "mandarin_lyric"]
    )
    def test_mms_detect_language_iso(self, use_model_mms, lyric_fname, request):
        lyric = request.getfixturevalue(lyric_fname)
        lang = self.model.detect_text_lang(lyric)

        assert isinstance(lang, str)
        assert len(lang) == 3  # 3 letter
        print(lang)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        __file__
    ]))
