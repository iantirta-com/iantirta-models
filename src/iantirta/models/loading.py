from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch


SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"

WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"


def load_state_dict(
    files: Iterable[str | Path],
) -> dict[str, torch.Tensor]:
    files = [Path(file) for file in files]

    if not files:
        raise ValueError("No checkpoint files were provided.")

    suffixes = {file.suffix for file in files}

    if suffixes == {".safetensors"}:
        return _load_safetensors(files)

    if suffixes == {".bin"}:
        return _load_pytorch(files)

    raise ValueError(
        "Unsupported or mixed checkpoint formats: "
        f"{sorted(suffixes)}"
    )


def _load_safetensors(
    files: list[Path],
) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    state_dict: dict[str, torch.Tensor] = {}

    for file in files:
        state_dict.update(load_file(file, device="cpu"))

    return state_dict


def _load_pytorch(
    files: list[Path],
) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}

    for file in files:
        loaded = torch.load(
            file,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(loaded, dict):
            raise ValueError(
                f"Expected state dict in {file}, "
                f"got {type(loaded).__name__}"
            )

        state_dict.update(loaded)

    return state_dict
