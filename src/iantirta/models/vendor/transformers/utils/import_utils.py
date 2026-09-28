# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of transformers, improved by iantirta.com
#
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
from __future__ import annotations

import functools
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import logging
import operator
import os
import re
import shutil
import subprocess
import sys
import warnings
from collections import OrderedDict
from collections.abc import Callable
from enum import Enum
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any

import packaging.version
from packaging import version

logger = logging.getLogger(__name__)


PACKAGE_DISTRIBUTION_MAPPING = importlib.metadata.packages_distributions()


def _candidate_distribution_names(pkg_name: str) -> list[str]:
    """Distribution names to try for the import name `pkg_name`, most likely first.

    The distribution name may differ from the import name (`PIL` is imported, but `pillow` is distributed), and
    `packages_distributions()` maps one to the other -- but only for wheels shipping a `top_level.txt` on
    Python < 3.12, which `torch` >= 2.14 does not. So keep the import name itself as a candidate.
    """
    # Per PEP 503, underscores and hyphens are equivalent in package names.
    normalized_pkg_name = pkg_name.replace("_", "-")
    distributions = PACKAGE_DISTRIBUTION_MAPPING.get(pkg_name, [])
    candidates = [
        *(name for name in (normalized_pkg_name, pkg_name) if name in distributions),
        *distributions,
        normalized_pkg_name,
        pkg_name,
    ]
    return list(dict.fromkeys(candidates))  # de-duplicate, keeping first-seen order


def _is_package_available(pkg_name: str, return_version: bool = False) -> tuple[bool, str]:
    """Check if `pkg_name` exist, and optionally try to get its version"""
    spec = importlib.util.find_spec(pkg_name)
    package_exists = spec is not None
    package_version = "N/A"
    if package_exists and return_version:
        for distribution_name in _candidate_distribution_names(pkg_name):
            try:
                package_version = importlib.metadata.version(distribution_name)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        else:
            # No metadata under any candidate name (editable install without a `dist-info`, for example).
            # Last resort: importing defeats the lazy imports these checks guard, costing every
            # `import transformers` the package's whole import tree.
            package = importlib.import_module(pkg_name)
            package_version = getattr(package, "__version__", "N/A")
            # No version + no __file__ means a namespace package (PEP 420) shadowing on sys.path, not a real install.
            if package_version == "N/A" and getattr(package, "__file__", None) is None:
                package_exists = False
        logger.debug(f"Detected {pkg_name} version: {package_version}")

    if return_version:
        return package_exists, package_version
    else:
        return package_exists, None


KERNELS_MIN_VERSION = "0.17.0"
KERNELS_MAX_VERSION = "0.18.0"


def _make_compile_constant(fn):
    """Mark `fn`'s result as a trace-time constant, so `torch.compile` inlines it instead of tracing it.

    This is `torch._dynamo.assume_constant_result`, spelled without importing torch: this module is what
    decides whether torch is installed, so it must never import it (and doing so would pull torch into
    `import transformers`, which is deliberately torch-free).

    Apply it *under* `@lru_cache`, not above: dynamo steps past the cache wrapper and only reads the
    marker on the function it actually traces.

    Only for helpers whose answer is fixed for the lifetime of the process — an install probe, a hardware
    capability, an environment variable. Never for a runtime query such as `is_cuda_stream_capturing`,
    where inlining a value that legitimately changes would silently bake a transient into the graph.
    """
    fn._dynamo_marked_constant = True
    return fn


@lru_cache
@_make_compile_constant
def is_kernels_available(
    MIN_VERSION: str = KERNELS_MIN_VERSION,
    MAX_VERSION: str = KERNELS_MAX_VERSION
) -> bool:
    is_available, kernels_version = (
        _is_package_available("kernels", return_version=True)
    )
    viable_version = False
    if kernels_version != "N/A":
        viable_version = version.parse(kernels_version) >= version.parse(MIN_VERSION) and version.parse(
            kernels_version
        ) < version.parse(MAX_VERSION)
    return is_available and viable_version


@lru_cache
@_make_compile_constant
def is_torch_available() -> bool:
    try:
        is_available, torch_version = _is_package_available("torch", return_version=True)
        parsed_version = version.parse(torch_version)
        if is_available and parsed_version < version.parse("2.5.0"):
            logger.warning_once(f"Disabling PyTorch because PyTorch >= 2.5 is required but found {torch_version}")
        return is_available and version.parse(torch_version) >= version.parse("2.5.0")
    except packaging.version.InvalidVersion:
        return False


def is_torchdynamo_compiling() -> bool:
    # Importing torch._dynamo causes issues with PyTorch profiler (https://github.com/pytorch/pytorch/issues/130622)
    # hence rather relying on `torch.compiler.is_compiling()` when possible (torch>=2.3)
    try:
        import torch

        return torch.compiler.is_compiling()
    except Exception:  # noqa: BLE001
        return False


#region Backend

# docstyle-ignore
PYTORCH_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your environment. Check out the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""


BACKENDS_MAPPING = OrderedDict(
    ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
)


def requires_backends(obj, backends):
    """
    Method that automatically raises in case the specified backends are not available. It is often used during class
    initialization to ensure the required dependencies are installed:

    ```py
    requires_backends(self, ["torch"])
    ```

    The backends should be defined in the `BACKEND_MAPPING` defined in `transformers.utils.import_utils`.

    Args:
        obj: object to be checked
        backends: list or tuple of backends to check.
    """
    if not isinstance(backends, list | tuple):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__

    failed = []
    for backend in backends:
        if isinstance(backend, Backend):
            available, msg = backend.is_satisfied, backend.error_message
        else:
            available, msg = BACKENDS_MAPPING[backend]

        if not available():
            failed.append(msg.format(name))

    if failed:
        raise ImportError("".join(failed))


class VersionComparison(Enum):
    EQUAL = operator.eq
    NOT_EQUAL = operator.ne
    GREATER_THAN = operator.gt
    LESS_THAN = operator.lt
    GREATER_THAN_OR_EQUAL = operator.ge
    LESS_THAN_OR_EQUAL = operator.le

    @staticmethod
    def from_string(version_string: str) -> VersionComparison:
        string_to_operator = {
            "=": VersionComparison.EQUAL,
            "==": VersionComparison.EQUAL,
            "!=": VersionComparison.NOT_EQUAL,
            ">": VersionComparison.GREATER_THAN,
            "<": VersionComparison.LESS_THAN,
            ">=": VersionComparison.GREATER_THAN_OR_EQUAL,
            "<=": VersionComparison.LESS_THAN_OR_EQUAL,
        }

        return string_to_operator[version_string]


@lru_cache
def split_package_version(package_version_str) -> tuple[str, str, str]:
    pattern = r"([a-zA-Z0-9_-]+)([!<>=~]+)([0-9.]+)"
    match = re.match(pattern, package_version_str)
    if match:
        return (match.group(1), match.group(2), match.group(3))
    else:
        raise ValueError(f"Invalid package version string: {package_version_str}")


class Backend:
    def __init__(self, backend_requirement: str):
        self.package_name, self.version_comparison, self.version = split_package_version(backend_requirement)

        if self.package_name not in BACKENDS_MAPPING:
            raise ValueError(
                f"Backends should be defined in the BACKENDS_MAPPING. Offending backend: {self.package_name}"
            )

    def get_installed_version(self) -> str:
        """Return the currently installed version of the backend"""
        is_available, current_version = _is_package_available(self.package_name, return_version=True)
        if not is_available:
            raise RuntimeError(f"Backend {self.package_name} is not available.")
        return current_version

    def is_satisfied(self) -> bool:
        return VersionComparison.from_string(self.version_comparison).value(
            version.parse(self.get_installed_version()), version.parse(self.version)
        )

    def __repr__(self) -> str:
        return f'Backend("{self.package_name}", {VersionComparison[self.version_comparison]}, "{self.version}")'

    @property
    def error_message(self):
        return (
            f"{{0}} requires the {self.package_name} library version {self.version_comparison}{self.version}. That"
            f" library was not found with this version in your environment."
        )


def requires(*, backends=()):
    """
    This decorator enables two things:
    - Attaching a `__backends` tuple to an object to see what are the necessary backends for it
      to execute correctly without instantiating it
    - The '@requires' string is used to dynamically import objects
    """

    if not isinstance(backends, (tuple, list)):
        raise TypeError("Backends should be a tuple or list.")
    backends = tuple(backends)

    applied_backends = []
    for backend in backends:
        if backend in BACKENDS_MAPPING:
            applied_backends.append(backend)
        else:
            if any(key in backend for key in ["=", "<", ">"]):
                applied_backends.append(Backend(backend))
            else:
                raise ValueError(f"Backend should be defined in the BACKENDS_MAPPING. Offending backend: {backend}")

    def inner_fn(fun):
        if isinstance(fun, type):
            # For classes, just attach the metadata — don't wrap, as that would
            # turn the class into a plain function and break isinstance checks.
            fun.__backends = applied_backends
            return fun

        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            requires_backends(fun, applied_backends)
            return fun(*args, **kwargs)

        wrapper.__backends = applied_backends  # type: ignore [unresolved-attribute]
        return wrapper

    return inner_fn

#endregion
