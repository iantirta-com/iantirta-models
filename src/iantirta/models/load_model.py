
from pathlib import Path
from typing import Literal

class AutoModel:

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        **kwargs
    ):
        if "adefossez" in model_name_or_path or "demucs" in model_name_or_path:
            from .models.demucs import DemucsBagOfModel
            
            return DemucsBagOfModel.from_pretrained(model_name_or_path, **kwargs)
        else:
            import transformers
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
            architectures = getattr(config, "architectures", None)

            if not architectures:
                raise NotImplementedError("empty arch not supported.")

            if len(architectures) > 1:
                raise NotImplementedError("Multiple Architectures Not implemented")

            model_class = getattr(transformers, architectures[0], None)

            if not model_class:
                raise NotImplementedError("Empty model class is not supported")

            return model_class.from_pretrained(model_name_or_path, config=config, **kwargs)

def load_model(
    model_name_or_path: str | Path,
    model_type: Literal["separation", "asr"],
    **kwargs
):
    model = AutoModel.from_pretrained(model_name_or_path, **kwargs)

    match model_type:
        case "separation":
            from .audio_separation import AudioSeparation
            wrapper = AudioSeparation(model)
        case "asr":
            from .asr import ASRModel
            wrapper = ASRModel(model)
        case _:
            raise ValueError("Not Implemented")

    return wrapper


if __name__ == "__main__":
    model = load_model(
        "adefossez/Demucs-mdx_extra_q",
        "separation"
    )