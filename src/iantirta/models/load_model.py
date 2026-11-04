
from pathlib import Path
from typing import Literal

_all__ = [
    "load_model"
]


def load_model(
    model_name_or_path: str | Path,
    model_type: Literal["separation", "asr", 'ctc_align'],
    *,
    model_type_options: dict | None = None,
    **kwargs
):
    match model_type:
        case "separation":
            from .audio_separation import AudioSeparationModel
            wrapper = AudioSeparationModel(model_name_or_path, **model_type_options)
        case "asr":
            from .asr import Qwen3ASRModel
            wrapper = Qwen3ASRModel(model_name_or_path, **model_type_options)
        case 'ctc_align':
            from .asr import Wav2Vec2CTCModel
            wrapper = Wav2Vec2CTCModel(model_name_or_path, **model_type_options)
        case _:
            raise ValueError("Not Implemented")

    return wrapper


if __name__ == "__main__":
    model = load_model(
        "adefossez/Demucs-mdx_extra_q",
        "separation"
    )
