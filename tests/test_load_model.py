
import pytest

from iantirta.models.load_model import load_model


@pytest.mark.manual
def test_load_model_qwen():
    load_model(
        'Qwen/Qwen3-ASR-1.7B-hf',
        'asr',
        model_type_options={
            'max_inference_batch_size': 1,
            'num_beams': 4,
        }
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        __file__
    ]))
