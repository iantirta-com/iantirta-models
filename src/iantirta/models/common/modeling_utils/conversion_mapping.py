# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of transformers, improved by iantirta.com
#
# Copyright (C) 2025 the HuggingFace Inc. team. All rights reserved.
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

from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING

from .loading_core import (
    WeightConverter,
    WeightRenaming,
    WeightTransform,
)

if TYPE_CHECKING:
    from ...core.model.pretrained import PreTrainedModel
    from .quantization.base import HfQuantizer


_MODEL_TO_CONVERSION_PATTERN = {}


def _build_checkpoint_conversion_mapping():
    mapping = {
        "legacy": [
            WeightRenaming(
                source_patterns="LayerNorm.gamma",
                target_patterns="LayerNorm.weight",
            ),
            WeightRenaming(
                source_patterns="LayerNorm.beta",
                target_patterns="LayerNorm.bias",
            ),
        ],
    }
    mapping["legacy"] += [
            WeightRenaming(
            source_patterns=".weight_g$",
            target_patterns=".parametrizations.weight.original0",
        ),
        WeightRenaming(
            source_patterns=".weight_v$",
            target_patterns=".parametrizations.weight.original1",
        ),
    ]
    
    for model_type, base_pattern in _MODEL_TO_CONVERSION_PATTERN.items():
        if model_type in mapping:
            continue
        mapping[model_type] = mapping[base_pattern].copy()

    return mapping


_checkpoint_conversion_mapping_cache = None


def get_checkpoint_conversion_mapping(model_type):
    global _checkpoint_conversion_mapping_cache
    if _checkpoint_conversion_mapping_cache is None:
        _checkpoint_conversion_mapping_cache = _build_checkpoint_conversion_mapping()
    return deepcopy(_checkpoint_conversion_mapping_cache.get(model_type))


USER_REGISTERED_MAPPINGS = set()


def register_checkpoint_conversion_mapping(
    model_type_or_class_name: str,
    mapping: list[WeightConverter | WeightRenaming],
    overwrite: bool = False,
) -> None:
    """
    Register a conversion mapping for a model type string or a class name.

    Class names take priority over `model_type` strings during lookup (see
    `extract_weight_conversions_for_model`), making it possible to define
    task-head-specific or class-specific conversions that differ from the shared
    `model_type` baseline.
    """
    global _checkpoint_conversion_mapping_cache
    if _checkpoint_conversion_mapping_cache is None:
        _checkpoint_conversion_mapping_cache = _build_checkpoint_conversion_mapping()
    if model_type_or_class_name in _checkpoint_conversion_mapping_cache and not overwrite:
        raise ValueError(
            f"Conversion mapping for '{model_type_or_class_name}' already exists. Pass overwrite=True to replace it."
        )
    _checkpoint_conversion_mapping_cache[model_type_or_class_name] = mapping
    # Keep track of what was added manually by the user
    USER_REGISTERED_MAPPINGS.add(model_type_or_class_name)


def get_model_conversion_mapping(
    model: PreTrainedModel,
    key_mapping: dict[str, str] | None = None,
    hf_quantizer: HfQuantizer | None = None,
    add_legacy: bool = True,
) -> list[WeightTransform]:
    """
    Collect the ordered list of weight transforms for `model` (used during
    loading and, when reversed, during saving).

    Each `PreTrainedModel` sub-module is looked up by class name then
    `model_type`.  Root transforms are applied globally; sub-module transforms
    have their `scope_prefix` set so they only match keys under that prefix.  After any
    sub-module is processed, both its class name and `model_type` are marked
    seen to prevent `XForY` / `XModel` pairs from applying the same mapping
    twice via different lookup paths.
    """
    from ...core.model.pretrained import PreTrainedModel

    # note: this function is used in PEFT, so changing the API requires coordination
    weight_conversions = []

    # Load models with explicit, user-provided key mapping
    if key_mapping is not None:
        weight_conversions = [WeightRenaming(source_patterns=k, target_patterns=v) for k, v in key_mapping.items()]

    # Maps each identifier (class name or model_type) to the module paths that have
    # already claimed it.  A later module is skipped only when one of those paths is
    # an ancestor of the current module path — siblings are never ancestors of each
    # other, so two sibling sub-models with the same model_type both get their own
    # scoped transforms.  A child is an ancestor of everything nested under it, which
    # prevents a parent's transforms from being duplicated with a scoped copy for the child.
    seen_identifiers: defaultdict[str, list[str]] = defaultdict(list)

    for module_name, submodule in model.named_modules():
        # Skip if it's not a submodel
        if not isinstance(submodule, PreTrainedModel):
            continue

        class_name = type(submodule).__name__
        model_type = submodule.config.model_type

        # Skip it if it's custom code and it was NOT registered by the user directly: it may have the same `model_type`/ClassName
        # as a native model inside the library, but it uses custom modeling so it should not share the conversions
        if (
            submodule.is_custom_code()
            and class_name not in USER_REGISTERED_MAPPINGS
            and model_type not in USER_REGISTERED_MAPPINGS
        ):
            continue

        # Skip if an ancestor already claimed this class (its unscoped transforms already cover this subtree).
        if any(seen == "" or module_name.startswith(seen + ".") for seen in seen_identifiers[class_name]):
            continue

        # Class name takes priority — a class-specific mapping bypasses the model_type
        # deduplication check (e.g. LlavaModel nested inside LlavaForConditionalGeneration
        # must still get its own scoped mapping even after "llava" is marked seen).
        conversions = get_checkpoint_conversion_mapping(class_name)
        found_via_class = conversions is not None

        if not found_via_class:
            # Same ancestor check as above, but via model_type for modules without a class-specific mapping.
            if model_type and any(
                seen == "" or module_name.startswith(seen + ".") for seen in seen_identifiers[model_type]
            ):
                continue
            if model_type is not None:
                conversions = get_checkpoint_conversion_mapping(model_type)

        if conversions is None:
            continue

        is_root_model = module_name == ""
        if not is_root_model:
            # Scope each transform so it only matches keys under this sub-module's prefix - but we still allow to
            # arbitrary add/remove base_model_prefix to load ForXXX model from BaseModel and the opposite
            # Note that we need 2 removeprefix calls here, as only one level of nesting would not have the ending dot to module_name
            scope_prefix = module_name.removeprefix(model.base_model_prefix)
            scope_prefix = scope_prefix.removeprefix(".")
            for transform in conversions:
                transform.scope_prefix = scope_prefix
                transform.base_model_prefix = model.base_model_prefix
        weight_conversions.extend(conversions)

        seen_identifiers[class_name].append(module_name)
        # Only record model_type when the hit was via model_type. When the hit was via
        # class name, other sub-modules sharing the same model_type but without a
        # class-specific mapping (e.g. DetrModel under DetrForSegmentation) must still
        # be reachable so their base transforms are picked up and scoped.
        if not found_via_class and model_type:
            seen_identifiers[model_type].append(module_name)

    if add_legacy:
        weight_conversions.extend(get_checkpoint_conversion_mapping("legacy"))

    # Let the quantizer rewrite / augment the conversion pipeline. This is where the
    # FP8 dequantizer (when `dequantize=True`) prepends a `Fp8Dequantize` op to
    # every existing converter so that per-block scales are applied *before* any
    # expert-merge / concat ops flatten the per-expert structure away.
    if hf_quantizer is not None:
        weight_conversions = hf_quantizer.update_weight_conversions(weight_conversions)

    return weight_conversions
