# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of demucs, improved by iantirta.com
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import yaml

from iantirta.models.core.config import PreTrainedConfig
from iantirta.models.vendor.huggingface_hub.dataclasses import strict


@strict
class DemucsConfig(PreTrainedConfig):
    segment: float | int | None = None
    weights: list[list[float]] | None = None

    def __post_init__(self, **kwargs):
        self._bag_models = kwargs.pop("models", None)
        super().__post_init__(**kwargs)

    @property
    def bag_models(self) -> list[str] | None:
        return getattr(self, "_bag_models", None)

    @bag_models.setter
    def bag_models(self, value):
        self._bag_models = value

    @classmethod
    def _dict_from_json_file(cls, json_file: str | os.PathLike):
        if json_file.endswith(".json"):
            return super()._dict_from_json_file(json_file)
        else:
            try:
                with open(json_file) as file:
                    return yaml.safe_load(file)
            except:  # noqa: TRY203
                raise
