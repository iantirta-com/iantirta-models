# iantirta-models
Fast Reusable Modeling ML utilities

[![PyPI](https://img.shields.io/pypi/v/iantirta-models)](https://pypi.org/project/iantirta-models/)
[![Python](https://img.shields.io/pypi/pyversions/iantirta-models)](https://pypi.org/project/iantirta-models/)
[![Tests](https://github.com/iantirta-com/iantirta-models/actions/workflows/test.yml/badge.svg)](https://github.com/iantirta-com/iantirta-models/actions/workflows/test.yml)
[![License](https://img.shields.io/github/license/iantirta-com/iantirta-models)](https://github.com/iantirta-com/iantirta-models/blob/main/LICENSE)

[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://iantirta-com.github.io/iantirta-models/)
[![GitHub](https://img.shields.io/badge/GitHub-iantirta--com-181717?logo=github)](https://github.com/iantirta-com/iantirta-models)

Model utilities and model implementations for Python.

`iantirta-models` is the model layer of the Iantirta.com Python ecosystem. It provides common infrastructure for loading, configuring, and working with machine-learning models without requiring users to install a large framework stack for every model.

The package is designed to work with models distributed through repositories such as Hugging Face while keeping model loading and runtime dependencies under the control of Iantirta.

## Features

- Model and configuration abstractions
- Local and remote model loading
- PyTorch checkpoint loading
- Safetensors support
- Model caching
- Hugging Face model repository integration
- Vendored model infrastructure
- Audio and speech model implementations
- Automatic speech recognition model support
- Source separation model support

## Installation

Install the base package:

```bash
pip install iantirta-models
```

For models that require PyTorch:

```bash
pip install "iantirta-models[asr]"
```

or:

```bash
pip install "iantirta-models[separation]"
```

Install all runtime model dependencies:

```bash
pip install "iantirta-models[all]"
```

For development:

```bash
pip install "iantirta-models[dev]"
```

## Usage
The package provides common model and
configuration interfaces:

```py
from iantirta.models import Model, ModelConfig
```

Models can be configured and loaded independently
from the application layer.

For example,an application such as iantirta-audio
can use iantirta-models for model loading and
inference while keeping audio processing and
transcription workflows in the audio package.

## Model Repositories
Models may be distributed through external model repositories such as Hugging Face.
The repository is treated as a source of model files and metadata. Runtime model loading is handled by iantirta-models.
Model files are cached locally to avoid downloading the same files repeatedly.

## Supported Models

The project is currently in alpha development.

Current and planned model implementations include:
- Demucs
- Wav2Vec2
- Qwen3-ASR

The available models and APIs may change between alpha releases.

## License
Copyright © Iantirta.com.
This project is licensed under the Apache License 2.0.
See LICENSE for the full license text.

Some model implementations and vendored components may originate from third-party projects and retain their respective copyright notices and licenses.
