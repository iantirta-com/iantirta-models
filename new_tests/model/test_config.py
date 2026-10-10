import pytest

from iantirta.models import AutoConfig


@pytest.mark.manual
@pytest.mark.parametrize(
    ("repo", "class_name"),
    [
        ("facebook/mms-1b-all", "Wav2Vec2Config"),
        ("Qwen/Qwen3-ASR-1.7B-hf", "Qwen3ASRConfig"),
        ("Qwen/Qwen3-ForcedAligner-0.6B-hf", "Qwen3ASRConfig"),
    ]
)
def test_auto_config(repo, class_name):
    config = AutoConfig.from_pretrained(repo)

    assert config is not None
    assert config.__class__.__name__ == class_name
    assert config.__module__.startswith("iantirta.models.")
    assert "vendor" not in config.__module__
