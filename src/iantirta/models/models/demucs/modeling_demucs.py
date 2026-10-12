import importlib
import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

from .apply import BagOfModels, Model
from .states import load_model

__all__ = [
    "DemucsPreTrainedModel"
]


class DemucsPreTrainedModel:
    
    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | os.PathLike | None,
        **kwargs,
    ):
        import yaml
        from transformers.utils.hub import cached_file, cached_files

        _, name = repo_id.rsplit("/", 1)
        if "-" in name and name.startswith("Demucs-"):
            name = name[len("Demucs-"):]
        
        with open(cached_file(repo_id, f"{name}.yaml", **kwargs)) as file:
            bag = yaml.safe_load(file)

        models_path = cached_files(
            repo_id, 
            [f"{sig}.safetensors" for sig in bag['models']]
        )
        models = [cls.load_safetensors_model(model_path) for model_path in models_path]
    
        return BagOfModels(models, bag.get('weights'), bag.get('segment'))

    @staticmethod
    def load_safetensors_model(path: str | Path) -> Model:
        """Load a single model from a safetensors file, with the model class and init
        arguments stored as json in the safetensors metadata."""
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as file:
            metadata = file.metadata()
            tensors = {key: file.get_tensor(key) for key in file.keys()}
        state: Any
        if 'structure' in metadata:
            state = _unflatten_state(tensors, json.loads(metadata['structure']))
        else:
            state = tensors
        
        module, name = metadata['klass'].rsplit(".", 1)
        if module.startswith("demucs."):
            module = f"iantirta.models.models.demucs.{module[len('demucs.'):]}"
        
        klass = getattr(importlib.import_module(module), name)
        args = _decode_json(json.loads(metadata['args']))
        kwargs = _decode_json(json.loads(metadata['kwargs']))
        return load_model({'klass': klass, 'args': args, 'kwargs': kwargs, 'state': state})


def _decode_json(value):
    """Decode the json encoding of the model init arguments, in particular
    fractions (e.g. the `segment` param of HTDemucs)."""
    if isinstance(value, dict):
        if value.get("_type") == "fraction":
            return Fraction(value["numerator"], value["denominator"])
        return {key: _decode_json(item) for key, item in value.items()}
    elif isinstance(value, list):
        return [_decode_json(item) for item in value]
    return value


def _unflatten_state(tensors: dict[str, Any], structure):
    """Rebuild a nested model state (e.g. diffq packed states) from the flat
    safetensors tensor dict and the json structure stored in its metadata.
    Inverse of `_flatten_state` in `tools/export_hf.py`."""
    def unflatten(node):
        if isinstance(node, dict):
            if "_tensor" in node:
                return tensors[node["_tensor"]]
            elif "_dict" in node:
                return {key: unflatten(item) for key, item in node["_dict"]}
            elif "_list" in node:
                return [unflatten(item) for item in node["_list"]]
            elif "_tuple" in node:
                return tuple(unflatten(item) for item in node["_tuple"])
            elif "_class" in node:
                module, name = node["_class"].rsplit(".", 1)
                return getattr(importlib.import_module(module), name)
            else:
                raise ValueError(f"Invalid structure node {node}.")
        return node
    return unflatten(structure)


if __name__ == "__main__":
    from rich import inspect as i
    name = "mdx_extra_q"
    repo_id = f"adefossez/Demucs-{name}"
    
    m = DemucsPreTrainedModel.from_pretrained(repo_id)
    i(m)