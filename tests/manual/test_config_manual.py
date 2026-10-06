
import pytest

from iantirta.models.core.auto import AutoConfig

pytestmark = pytest.mark.manual


def test_auto_config_mapping():
    from iantirta.models.core.auto import CONFIG_MAPPING
    print(CONFIG_MAPPING._modules)
    print(CONFIG_MAPPING["wav2vec2"])


@pytest.mark.parametrize(
    ("repo", "class_name"),
    [
        ("facebook/mms-1b-all", "Wav2Vec2Config"),
        ("Qwen/Qwen3-ASR-1.7B-hf", "Qwen3ASRConfig"),
        ("Qwen/Qwen3-ForcedAligner-0.6B-hf", "Qwen3ASRConfig"),
    ]
)
def test_config_manual(repo, class_name):
    config = AutoConfig.from_pretrained(repo)

    assert config is not None
    assert config.__class__.__name__ == class_name
    assert config.__module__.startswith("iantirta.models.")
    assert "vendor" not in config.__module__

    if hasattr(config, "sub_configs"):
        for subconfig in config.sub_configs:
            sub_config = getattr(config, subconfig)
            assert sub_config.__module__.startswith("iantirta.models.")
            assert "vendor" not in sub_config.__module__
