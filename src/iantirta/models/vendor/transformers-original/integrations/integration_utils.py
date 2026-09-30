# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Integrations with other Python libraries.
"""

import copy
import functools
import importlib.metadata
import importlib.util
import json
import numbers
import os
import re
import secrets
import shutil
import sys
import tempfile
import warnings
from dataclasses import fields
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import packaging.version


if os.getenv("WANDB_MODE") == "offline":
    print("[INFO] Running in WANDB offline mode")

from .. import PreTrainedModel, TrainingArguments
from .. import __version__ as version
from ..utils import (
    PushToHubMixin,
    flatten_dict,
    is_datasets_available,
    is_pandas_available,
    is_torch_available,
    logging,
)


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

# comet_ml requires to be imported before any ML frameworks
_MIN_COMET_VERSION = "3.43.2"
try:
    _comet_version = importlib.metadata.version("comet_ml")
    _is_comet_installed = True

    _is_comet_recent_enough = packaging.version.parse(_comet_version) >= packaging.version.parse(_MIN_COMET_VERSION)

    # Check if the Comet API Key is set
    import comet_ml

    if comet_ml.config.get_config("comet.api_key") is not None:
        _is_comet_configured = True
    else:
        _is_comet_configured = False
except (importlib.metadata.PackageNotFoundError, ImportError, ValueError, TypeError, AttributeError, KeyError):
    _comet_version = None
    _is_comet_installed = False
    _is_comet_recent_enough = False
    _is_comet_configured = False

_has_neptune = (
    importlib.util.find_spec("neptune") is not None or importlib.util.find_spec("neptune-client") is not None
)
if TYPE_CHECKING and _has_neptune:
    try:
        _neptune_version = importlib.metadata.version("neptune")
        logger.info(f"Neptune version {_neptune_version} available.")
    except importlib.metadata.PackageNotFoundError:
        try:
            _neptune_version = importlib.metadata.version("neptune-client")
            logger.info(f"Neptune-client version {_neptune_version} available.")
        except importlib.metadata.PackageNotFoundError:
            _has_neptune = False


# Integration functions:
def is_wandb_available():
    if importlib.util.find_spec("wandb") is not None:
        import wandb

        # wandb might still be detected by find_spec after an uninstall (leftover files or metadata), but not actually
        # import correctly. To confirm it's fully installed and usable, we check for a key attribute like "run".
        return hasattr(wandb, "run")
    else:
        return False


def is_trackio_available():
    return importlib.util.find_spec("trackio") is not None


def is_clearml_available():
    return importlib.util.find_spec("clearml") is not None


def is_comet_available():
    if _is_comet_installed is False:
        return False

    if _is_comet_recent_enough is False:
        logger.warning(
            "comet_ml version %s is installed, but version %s or higher is required. "
            "Please update comet_ml to the latest version to enable Comet logging with pip install 'comet-ml>=%s'.",
            _comet_version,
            _MIN_COMET_VERSION,
            _MIN_COMET_VERSION,
        )
        return False

    if _is_comet_configured is False:
        logger.warning(
            "comet_ml is installed but the Comet API Key is not configured. "
            "Please set the `COMET_API_KEY` environment variable to enable Comet logging. "
            "Check out the documentation for other ways of configuring it: "
            "https://www.comet.com/docs/v2/guides/experiment-management/configure-sdk/#set-the-api-key"
        )
        return False

    return True


def is_tensorboard_available():
    return importlib.util.find_spec("tensorboard") is not None or importlib.util.find_spec("tensorboardX") is not None


def is_optuna_available():
    return importlib.util.find_spec("optuna") is not None


def is_ray_available():
    return importlib.util.find_spec("ray") is not None


def is_ray_tune_available():
    if not is_ray_available():
        return False
    return importlib.util.find_spec("ray.tune") is not None


def is_azureml_available():
    if importlib.util.find_spec("azureml") is None:
        return False
    if importlib.util.find_spec("azureml.core") is None:
        return False
    return importlib.util.find_spec("azureml.core.run") is not None


def is_mlflow_available():
    if os.getenv("DISABLE_MLFLOW_INTEGRATION", "FALSE").upper() == "TRUE":
        return False
    return importlib.util.find_spec("mlflow") is not None


def is_dagshub_available():
    return None not in [importlib.util.find_spec("dagshub"), importlib.util.find_spec("mlflow")]


def is_neptune_available():
    return _has_neptune


def is_codecarbon_available():
    return importlib.util.find_spec("codecarbon") is not None


def is_flytekit_available():
    return importlib.util.find_spec("flytekit") is not None


def is_flyte_deck_standard_available():
    if not is_flytekit_available():
        return False
    return importlib.util.find_spec("flytekitplugins.deck") is not None


def is_dvclive_available():
    return importlib.util.find_spec("dvclive") is not None


def is_swanlab_available():
    return importlib.util.find_spec("swanlab") is not None


def is_kubeflow_available():
    if os.getenv("DISABLE_KUBEFLOW_INTEGRATION", "FALSE").upper() == "TRUE":
        return False
    return os.getenv("KUBEFLOW_TRAINER_SERVER_URL") is not None




def get_available_reporting_integrations():
    integrations = []
    if is_azureml_available() and not is_mlflow_available():
        integrations.append("azure_ml")
    if is_comet_available():
        integrations.append("comet_ml")
    if is_dagshub_available():
        integrations.append("dagshub")
    if is_dvclive_available():
        integrations.append("dvclive")
    if is_mlflow_available():
        integrations.append("mlflow")
    if is_neptune_available():
        integrations.append("neptune")
    if is_tensorboard_available():
        integrations.append("tensorboard")
    if is_wandb_available():
        integrations.append("wandb")
    if is_codecarbon_available():
        integrations.append("codecarbon")
    if is_clearml_available():
        integrations.append("clearml")
    if is_swanlab_available():
        integrations.append("swanlab")
    if is_trackio_available():
        integrations.append("trackio")
    if is_kubeflow_available():
        integrations.append("kubeflow")
    return integrations




