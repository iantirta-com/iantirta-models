from __future__ import annotations

from pathlib import Path
import os
import torch
import sys
from torch import nn
from contextlib import contextmanager
import copy
from collections import defaultdict
from contextlib import AbstractContextManager, ExitStack
import logging
from dataclasses import dataclass, field
from concurrent.futures import Future, ThreadPoolExecutor
import re
from tqdm import tqdm
import shutil
from collections import OrderedDict, defaultdict

from .config import ModelConfig
from .files import ensure_file
from .loading import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    load_state_dict,
)

logger = logging.getLogger(__name__)

# For I/O bound operations (i.e. here reading files), it is better to have fewer threads, e.g. 4 is a good default.
# Having too many is actually harming performances quite a lot, i.e. using 16 can sometimes lead to taking TWICE
# as much time to load the same model
GLOBAL_WORKERS = min(4, os.cpu_count() or 4)



class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `fastcore` library.
    """

    def __init__(
        self,
        context_managers: list[AbstractContextManager]
    ):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)


@contextmanager
def local_torch_dtype(
    dtype: torch.dtype,
    model_class_name: str | None = None
):
    """
    Locally change the torch default dtype to `dtype`, and restore the old one upon exiting the context.
    If `model_class_name` is provided, it's used to provide a more helpful error message if `dtype` is not valid.
    """
    # Just a more helping error before we set `torch.set_default_dtype` later on which would crash in this case
    if not dtype.is_floating_point:
        if model_class_name is not None:
            error_message = (
                f"{model_class_name} cannot be instantiated under `dtype={dtype}` as it's not a floating-point dtype"
            )
        else:
            error_message = f"Cannot set `{dtype}` as torch's default as it's not a floating-point dtype"
        raise ValueError(error_message)

    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(original_dtype)


def _get_dtype(
    dtype: str | torch.dtype | dict | None,
    files: list[Path] | None,
    config: ModelConfig,
    weights_only: bool,
) -> tuple[ModelConfig, torch.dtype]:
    """Find the correct `dtype` to use based on provided arguments. Also update the `config` based on the
    inferred dtype. We do the following:
    1. If dtype is "auto", we try to read the config, else auto-detect dtype from the loaded state_dict, by checking
    its first weights entry that is of a floating type - we assume all floating dtype weights are of the same dtype
    2. Else, use the dtype provided as a dict or str
    """
    if dtype is not None:
        if isinstance(dtype, str):
            if dtype == "auto":
                if hasattr(config, "dtype") and config.dtype is not None:
                    dtype = config.dtype
                    logger.info(f"Will use dtype={dtype} as defined in model's config object")
                else:
                    if files is not None and str(files[0]).endswith(".gguf"):
                        dtype = None
                    else:
                        raise NotImplementedError
                        state_dict = load_state_dict(
                            files[0],
                            map_location="meta",
                            weights_only=weights_only
                        )
                        dtype = get_state_dict_dtype(state_dict)
                    if dtype is not None:
                        logger.info(
                            f"Since the `dtype` attribute can't be found in model's config object, "
                            f"will use dtype={dtype} as derived from model's weights"
                        )
            elif hasattr(torch, dtype):
                dtype = getattr(torch, dtype)
            else:
                raise ValueError(
                    "`dtype` provided as a `str` can only be `'auto'`, or a string representation of a valid `torch.dtype`"
                )

            # cast it to a proper `torch.dtype` object
            dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        elif not isinstance(dtype, (dict, torch.dtype)):
            raise ValueError(
                f"`dtype` can be one of: `torch.dtype`, `'auto'`, a string of a valid `torch.dtype` or a `dict` with valid `dtype` "
                f"for each sub-config in composite configs, but received {dtype}"
            )
    else:
        # set torch.get_default_dtype() (usually fp32) as the default dtype if `None` is provided
        dtype = torch.get_default_dtype()

    # Get the main dtype
    if isinstance(dtype, dict):
        main_dtype = dtype.get("", torch.get_default_dtype())
        main_dtype = getattr(torch, main_dtype) if isinstance(main_dtype, str) else main_dtype

        logger.warning_once(
            "Using different dtypes per module is deprecated and will be removed in future versions "
            "Setting different dtypes per backbone model might cause device errors downstream, therefore "
            f"setting the dtype={main_dtype} for all modules."
        )

    else:
        main_dtype = dtype

    # Set it on the config and subconfigs
    config.dtype = main_dtype
    for sub_config_key in config.sub_configs:
        raise NotImplementedError
        if (sub_config := getattr(config, sub_config_key)) is not None:
            sub_config.dtype = main_dtype

    return config, main_dtype


def process_target_pattern(pattern: str) -> tuple[str, str | None]:
    """
    Process a target pattern for reverse mapping (when targets become sources).

    This handles several edge cases in checkpoint conversion mappings:
    - Removes `^` prefix and `$` suffix (start/end of string anchors)
    - Removes negative lookahead/lookbehind assertions
    - Detects capturing groups and replaces them with `\\1` backreference

    Args:
        pattern: The target pattern to process for reverse mapping.

    Returns:
        A tuple of (processed_pattern, captured_group) where captured_group is
        the original capturing group found (e.g., "(encoder|decoder)") or None.
    """
    # Some mapping contains `^` to notify start of string when matching -> remove it during reverse mapping
    pattern = pattern.removeprefix("^")
    # Some mapping contains `$` to notify end of string when matching -> remove it during reverse mapping
    pattern = pattern.removesuffix("$")
    # Remove negative lookahead/behind if any. This is ugly but needed for reverse mapping of
    # Qwen2.5, Sam3, Ernie4.5 VL MoE! It needs to be non greedy in case there are several
    pattern = re.sub(r"\(\?.+?\)?\)", "", pattern)
    # Remove the backslash for literal dots
    pattern = pattern.replace(r"\.", ".")
    # Allow capturing groups in patterns, i.e. to add/remove a prefix to all keys (e.g. timm_wrapper, sam3)
    capturing_group_match = re.search(r"\(.+?\)", pattern)
    captured_group = None
    if capturing_group_match:
        captured_group = capturing_group_match.group(0)
        pattern = pattern.replace(captured_group, r"\1", 1)
    return pattern, captured_group


def process_source_pattern(source_pattern: str, target_pattern: str) -> str:
    """
    Process a source pattern for reverse mapping (when sources become targets).
    This is useful because usually if the original source (so now the target in reverse mode) had a `^` or `$`
    to restrict to start/end of string, we should do the same in reverse mode. This is why this method is conditioned
    on the target pattern, we want to do it only for pairs (source, target) when the original source (so the current target
    in reverse mode) had it.
    """
    if target_pattern.startswith("^"):
        source_pattern = f"^{source_pattern}" if not source_pattern.startswith("^") else source_pattern
    if target_pattern.endswith("$"):
        source_pattern = f"{source_pattern}$" if not source_pattern.endswith("$") else source_pattern

    return source_pattern


class WeightTransform:
    # Restrict the attributes that can be attached
    __slots__ = (
        "source_patterns",
        "target_patterns",
        "compiled_sources",
        
        "_original_source_patterns",
        "_original_target_patterns",
        "_was_used",
        "scope_prefix",
        "base_model_prefix",
    )
    
    def __init__(
        self,
        source_patterns: str | list[str],
        target_patterns: str | list[str]
    ):
        self.source_patterns: list[str] = source_patterns
        self.target_patterns: list[str] = target_patterns
        # Those are needed to be able to reverse correctly the transform, as the patterns may be processed
        self._original_source_patterns = self.source_patterns.copy()
        self._original_target_patterns = self.target_patterns.copy()

        # Flag to notice if the Transform was used
        self._was_used = False
        
        # Optional scope_prefix/base_model_prefix. When used, the transform will only match and apply to keys containing
        # either `base_model_prefix.scope_prefix.` or `scope_prefix.` prefixes
        self.scope_prefix: str | None = None
        self.base_model_prefix: str | None = None

        # We need to process a few exceptions here when instantiating the reverse mapping (i.e. the targets become
        # sources, and sources become targets). The issues lie in the sources usually, so here we need to check the
        # targets for the reversed mapping

        # Process target_patterns: detect capturing groups and replace with \1
        # Store the original capturing group patterns for reverse mapping
        target_capturing_groups: list[str] = []
        for i, pattern in enumerate(self.target_patterns):
            self.target_patterns[i], captured_group = process_target_pattern(pattern)
            if captured_group is not None:
                target_capturing_groups.append(captured_group)

        # Validate that we only have one unique capturing group pattern across all targets
        # This ensures deterministic reverse mapping when sources have \1 backreferences
        unique_capturing_groups = set(target_capturing_groups)
        if len(unique_capturing_groups) > 1:
            raise ValueError(
                f"Multiple different capturing groups found in target_patterns: {unique_capturing_groups}. "
                f"All target patterns must use the same capturing group pattern."
            )
        unique_capturing_group = unique_capturing_groups.pop() if unique_capturing_groups else None

        # We also need to check capturing groups in the sources during reverse mapping (e.g. timm_wrapper, sam3)
        for i, pattern in enumerate(self.source_patterns):
            # Replace capturing groups
            if r"\1" in pattern:
                if unique_capturing_group is None:
                    raise ValueError(
                        f"Source pattern '{pattern}' contains \\1 backreference, but no capturing groups "
                        f"found in target_patterns."
                    )
                # Use the unique capturing group from target_patterns for all sources
                pattern = pattern.replace(r"\1", unique_capturing_group, 1)
            # Potentially process a bit more for consistency - only if they are consistent pairs, i.e. the length is the same
            if len(self.source_patterns) == len(self.target_patterns):
                pattern = process_source_pattern(pattern, self._original_target_patterns[i])
            self.source_patterns[i] = pattern

        # Construct the regex we will use to rename keys from the sources to the targets
        branches = []
        for i, source_pattern in enumerate(self.source_patterns):
            group_name = f"g{i}"
            pattern = source_pattern.replace("*.", r".*\.")
            branches.append(f"(?P<{group_name}>{pattern})")
        self.compiled_sources = re.compile("|".join(branches))


    def __setattr__(self, name, value):
        if name in ("source_patterns", "target_patterns"):
            # We do not allow to re-set the patterns, as they are linked between each other and changing one
            # without the other can mess-up with the capturing groups/compiled sources
            if hasattr(self, name):
                raise ValueError(f"Cannot assign to field {name}, you should create a new instance")
            # Switch str to list
            elif isinstance(value, str):
                value = [value]
        object.__setattr__(self, name, value)

    def _scoped_match(self, source_key: str) -> tuple[str | None, str, re.Match[str]] | None:
        """
        Strip `scope_prefix` (if any) from `source_key`, then match `compiled_sources` against the
        remaining suffix.

        Returns `(prefix_dot, key_to_match, match_object)` on match, else `None`. `prefix_dot` is
        the prefix consumed from `source_key`: either `f"{scope_prefix}."` or that same string with
        one `base_model_prefix` level stripped or prepended when the former didn't match.
        `None` when `scope_prefix` is unset.
        """
        key_to_match = source_key
        prefix = None
        if self.scope_prefix is not None:
            scope_prefix = f"{self.scope_prefix}." if self.scope_prefix != "" else ""
            base_model_prefix = f"{self.base_model_prefix}." if self.base_model_prefix != "" else ""
            # First, try to match the longest sequence, i.e. base_model_prefix + scope_prefix
            if source_key.startswith(base_model_prefix + scope_prefix):
                prefix = base_model_prefix + scope_prefix
            # Then, try to strip the base_model_prefix, in case we load a ForXXX model from BaseModel weights
            elif source_key.startswith(scope_prefix):
                prefix = scope_prefix
            # In this case, no match is ever possible
            else:
                return None
            key_to_match = source_key.removeprefix(prefix)

        match_object = self.compiled_sources.search(key_to_match)
        if match_object is None:
            return None
        return (prefix, key_to_match, match_object)


    def rename_source_key(self, source_key: str) -> tuple[str, str | None]:
        """
        Return a tuple (renamed_key, source_pattern_producing_the_match).
        Try renaming `source_key` according to the source and target patterns of the current WeightTransform.
        In case of a one-to-many transform, i.e. we have several target patterns, the matching source pattern
        will be replaced by the first of all the target patterns (they are then correctly expanded in the Operations).
        """
        matched = self._scoped_match(source_key)
        if matched is None:
            return source_key, None

        prefix_dot, key_to_match, match_object = matched

        # We have a match, so the Transform was used
        self._was_used = True

        # Find the source that produced the match (it's the first group that matched, as the search stops after first branch match)
        matching_group_name = next(name for name, val in match_object.groupdict().items() if val is not None)
        source_pattern_that_matched = self.source_patterns[int(matching_group_name[1:])]
        # If we matched, we always replace with the first target pattern, in case we have several (one to many transform)
        replacement = self.target_patterns[0]
        # Allow capturing groups in patterns, i.e. to add a prefix to all keys (e.g. timm_wrapper, sam3)
        if r"\1" in replacement:
            # The index of the internal group we need to replace is the index of the matched named group as it comes
            # inside that matched named group
            replaced_group_idx = self.compiled_sources.groupindex[matching_group_name] + 1
            replacement = replacement.replace(r"\1", match_object.group(replaced_group_idx))
        renamed_key = key_to_match.replace(match_object.group(0), replacement, 1)
        if prefix_dot is not None:
            renamed_key = prefix_dot + renamed_key
        return renamed_key, source_pattern_that_matched

    def was_used(self) -> bool:
        """
        Return whether the current Transform matched any weights during loading/saving. This is needed as some
        weight renaming transforms are not bijective, i.e. if we drop/add full parts of a name with PrefixChange, we
        lose some information that we cannot get back if we don't know if the Transform was used before already (say we
        have a prefix to drop, we need to know whether the checkpoints we loaded before contained the said prefix or not
        before adding it back, or not, during saving).
        """
        return self._was_used

class WeightRenaming(WeightTransform):
    # Special case of WeightTransform that only renames keys without any conversion.

    # Needs to be empty, otherwise the class will not be slotted
    __slots__ = ()


class WeightConverter(WeightTransform):
    __slots__ = ("operations", "force_cpu")

    def __init__(
        self,
        source_patterns: str | list[str],
        target_patterns: str | list[str],
        operations: list[ConversionOps],
        force_cpu: bool = False,
    ):
        super().__init__(source_patterns, target_patterns)
        

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

    return mapping
    
_checkpoint_conversion_mapping_cache = None

def get_checkpoint_conversion_mapping(model_type):
    global _checkpoint_conversion_mapping_cache
    if _checkpoint_conversion_mapping_cache is None:
        _checkpoint_conversion_mapping_cache = _build_checkpoint_conversion_mapping()
    return copy.deepcopy(_checkpoint_conversion_mapping_cache.get(model_type))

    
def get_model_conversion_mapping(
    model: Model,
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

    # note: this function is used in PEFT, so changing the API requires coordination
    weight_conversions = []

    # Maps each identifier (class name or model_type) to the module paths that have
    # already claimed it.  A later module is skipped only when one of those paths is
    # an ancestor of the current module path — siblings are never ancestors of each
    # other, so two sibling sub-models with the same model_type both get their own
    # scoped transforms.  A child is an ancestor of everything nested under it, which
    # prevents a parent's transforms from being duplicated with a scoped copy for the child.
    seen_identifiers: defaultdict[str, list[str]] = defaultdict(list)

    for module_name, submodule in model.named_modules():
        # Skip if it's not a submodel
        if not isinstance(submodule, Model):
            continue

        class_name = type(submodule).__name__
        model_type = submodule.config.model_type

        # Skip it if it's custom code and it was NOT registered by the user directly: it may have the same `model_type`/ClassName
        # as a native model inside the library, but it uses custom modeling so it should not share the conversions
        # if (
        #     submodule.is_custom_code()
        #     and class_name not in USER_REGISTERED_MAPPINGS
        #     and model_type not in USER_REGISTERED_MAPPINGS
        # ):
        #     continue

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
            raise NotImplementedError(f"{module_name} - {submodule}\n : {conversions}")
            # Scope each transform so it only matches keys under this sub-module's prefix - but we still allow to
            # arbitrary add/remove base_model_prefix to load ForXXX model from BaseModel and the opposite
            # Note that we need 2 removeprefix calls here, as only one level of nesting would not have the ending dot to module_name
            scope_prefix = module_name.removeprefix(model.base_model_prefix)
            scope_prefix = scope_prefix.removeprefix(".")
            for transform in conversions:
                transform.scope_prefix = scope_prefix
                transform.base_model_prefix = model.base_model_prefix
        raise NotImplementedError(f"{module_name} - {submodule}\n : {conversions}")
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

    return weight_conversions


def get_torch_context_manager_or_global_device():
    """
    Test if a device context manager is currently in use, or if it is not the case, check if the default device
    is not "cpu". This is used to infer the correct device to load the model on, in case `device_map` is not provided.
    """
    device_in_context = torch.tensor([]).device
    default_device = torch.get_default_device()
    # This case means no context manager was used -> we still check if the default that was potentially set is not cpu
    if device_in_context == default_device:
        if default_device != torch.device("cpu"):
            return default_device
        return None
    return device_in_context


def check_and_set_device_map(
    device_map: "torch.device | int | str | dict | None"
) -> dict | str | None:
    # Potentially detect context manager or global device, and use it (only if no device_map was provided)
    if device_map is None:
        device_in_context = get_torch_context_manager_or_global_device()
        if device_in_context == torch.device("meta"):
            raise RuntimeError(
                "You are using `from_pretrained` with a meta device context manager or `torch.set_default_device('meta')`.\n"
                "This is an anti-pattern as `from_pretrained` wants to load existing weights.\nIf you want to initialize an "
                "empty model on the meta device, use the context manager or global device with `from_config`, or `ModelClass(config)`"
            )
        device_map = device_in_context

    # change device_map into a map if we passed an int, a str or a torch.device
    if isinstance(device_map, torch.device):
        device_map = {"": device_map}
    elif (
        isinstance(device_map, str) and
        device_map not in [
            "auto", "balanced",
            "balanced_low_0", "sequential"
        ]
    ):
        try:
            if device_map == "cuda":
                # setting to the local rank
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                device_map = f"cuda:{local_rank}"
            device_map = {"": torch.device(device_map)}
        except RuntimeError:
            raise ValueError(
                "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
                f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
            )
    elif isinstance(device_map, int):
        if device_map < 0:
            raise ValueError(
                "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
            )
        else:
            device_map = {"": device_map}

    if device_map is not None:
        if not is_accelerate_available():
            raise ValueError(
                "Using a `device_map`, `tp_plan`, `torch.device` context manager or setting `torch.set_default_device(device)` "
                "requires `accelerate`. You can install it with `pip install accelerate`"
            )
    return device_map


PALETTE = {
    "reset": "[0m",
    "red": "[31m",
    "yellow": "[33m",
    "orange": "[38;5;208m",
    "purple": "[35m",
    "green": "[32m",
    "bold": "[1m",
    "italic": "[3m",
    "dim": "[2m",
}

_DIGIT_RX = re.compile(r"(?<=\.)(\d+)(?=\.|$)")  # numbers between dots or at the end


def _pattern_of(key: str) -> str:
    """Replace every dot-delimited integer with '*' to get the structure."""
    return _DIGIT_RX.sub("*", key)


def _fmt_indices(values: list[int], cutoff=10) -> str:
    """Format a list of ints as single number, {a, ..., b}, or first...last."""
    if len(values) == 1:
        return str(values[0])
    values = sorted(values)
    if len(values) > cutoff:
        return f"{values[0]}...{values[-1]}"
    return ", ".join(map(str, values))


def update_key_name(mapping: dict[str, Any]) -> dict[str, Any]:
    """
    Merge keys like 'layers.0.x', 'layers.1.x' into 'layers.{0, 1}.x'
    BUT only merge together keys that have the exact same value.
    Returns a new dict {merged_key: value}.
    """
    # (pattern, value) -> list[set[int]] (per-star index values)
    not_mapping = False
    if not isinstance(mapping, dict):
        mapping = {k: k for k in mapping}
        not_mapping = True

    bucket: dict[str, list[set[int] | Any]] = defaultdict(list)
    for key, val in mapping.items():
        digs = _DIGIT_RX.findall(key)
        patt = _pattern_of(key)
        for i, d in enumerate(digs):
            if len(bucket[patt]) <= i:
                bucket[patt].append(set())
            bucket[patt][i].add(int(d))
        bucket[patt].append(val)

    out_items = {}
    for patt, values in bucket.items():
        sets, val = values[:-1], values[-1]
        parts = patt.split("*")  # stars are between parts
        final = parts[0]
        for i in range(1, len(parts)):
            if i - 1 < len(sets) and sets[i - 1]:
                insert = _fmt_indices(sorted(sets[i - 1]))
                if len(sets[i - 1]) > 1:
                    final += "{" + insert + "}"
                else:
                    final += insert
            else:
                final += "*"
            final += parts[i]

        out_items[final] = val
    out = OrderedDict(out_items)
    if not_mapping:
        return out.keys()
    return out

_ansi_re = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ansi_re.sub("", str(s))


def _pad(text, width):
    t = str(text)
    pad = max(0, width - len(_strip_ansi(t)))
    return t + " " * pad


def _make_table(rows, headers):
    # compute display widths while ignoring ANSI codes
    cols = list(zip(*([headers] + rows))) if rows else [headers]
    widths = [max(len(_strip_ansi(x)) for x in col) for col in cols]
    header_line = " | ".join(_pad(h, w) for h, w in zip(headers, widths))
    sep_line = "-+-".join("-" * w for w in widths)
    body = [" | ".join(_pad(c, w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join([header_line, sep_line] + body)


def _style(s, color):
    """Return color/style-formatted input `s` if `sys.stdout` is interactive, e.g. connected to a terminal."""
    if sys.stdout is not None and sys.stdout.isatty():
        return f"{PALETTE[color]}{s}{PALETTE['reset']}"
    else:
        return s


def _get_terminal_width(default=80):
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return default

def log_state_dict_report(
    model,
    pretrained_model_name_or_path: str,
    ignore_mismatched_sizes: bool,
    loading_info: LoadStateDictInfo,
    logger: logging.Logger | None = None,
):
    """
    Log a readable report about state_dict loading issues.

    This version is terminal-size aware: for very small terminals it falls back to a compact
    Key | Status view so output doesn't wrap badly.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Re-raise errors early if needed
    if loading_info.error_msgs:
        error_msg = "\n\t".join(loading_info.error_msgs)
        if "size mismatch" in error_msg:
            error_msg += (
                "\n\tYou may consider adding `ignore_mismatched_sizes=True` to `from_pretrained(...)` if appropriate."
            )
        raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

    # Pipeline-parallel details require walking the model state dict, so only collect them at info verbosity.
    report_model = model if logger.isEnabledFor(logging.INFO) else None
    report = loading_info.create_loading_report(report_model)
    if report is None:
        return

    prelude = f"{PALETTE['bold']}{model.__class__.__name__} LOAD REPORT{PALETTE['reset']} from: {pretrained_model_name_or_path}\n"

    # Log the report as warning
    logger.warning(prelude + report)

    # Re-raise in those case, after the report
    if loading_info.conversion_errors:
        raise RuntimeError(
            "We encountered some issues during automatic conversion of the weights. For details look at the `CONVERSION` entries of "
            "the above report!"
        )
    if not ignore_mismatched_sizes and loading_info.mismatched_keys:
        raise RuntimeError(
            "You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!"
        )


@dataclass(frozen=True)
class LoadStateDictConfig:
    """
    Config for loading weights. This allows bundling arguments that are just
    passed around.
    """

    pretrained_model_name_or_path: str | None = None
    
    use_safetensors: bool | None = None
    ignore_mismatched_sizes: bool = False
    sharded_metadata: dict | None = None
    device_map: dict | None = None
    disk_offload_folder: str | None = None
    offload_buffers: bool = False
    dtype: torch.dtype | None = None
    dtype_plan: dict = field(default_factory=dict)

    device_mesh: "DeviceMeshLike | None" = None
    weights_only: bool = True
    weight_mapping: list[WeightConverter | WeightRenaming] | None = None
    disable_mmap: bool | None = None

@dataclass
class LoadStateDictInfo:
    """
    Mutable container for state-dict loading results and diagnostics. Each entry in this structure is mutable,
    and will usually be mutated in-place during the loading pipeline.

    Attributes:
        missing_keys (`set[str]`):
            Keys that are missing from the loaded checkpoints but expected in the model's architecture.
        unexpected_keys (`set[str]`):
            Keys that are found in the checkpoints, but not expected in the model's architecture.
        skipped_pp_keys (`set[str]`):
            Checkpoint keys owned by another pipeline-parallel rank and intentionally not loaded here.
        mismatched_keys (`set[tuple[str, tuple[int], tuple[int]]]`):
            Keys that are found in the checkpoints and are expected in the model's architecture, but with a different shape.
        error_msgs ( `list[str]`):
            Some potential error messages.
        conversion_errors (`dict[str, str]`):
            Errors happening during the on-the-fly weight conversion process.
    """

    missing_keys: set[str]
    unexpected_keys: set[str]
    mismatched_keys: set[tuple[str, tuple[int], tuple[int]]]
    error_msgs: list[str]
    conversion_errors: dict[str, str]
    skipped_pp_keys: set[str]

    def missing_and_mismatched(self):
        """Return all effective missing keys, including `missing` and `mismatched` keys."""
        return self.missing_keys | {k[0] for k in self.mismatched_keys}


    def create_loading_report(self, model=None) -> str | None:
        """Generate the minimal table of a loading report."""
        term_w = _get_terminal_width()

        rows = []
        tips = "\n\nNotes:"

        if self.unexpected_keys:
            tips += f"\n- {_style('UNEXPECTED:', 'orange')}\t" + _style(
                "can be ignored when loading from different task/architecture; not ok if you expect identical arch.",
                "italic",
            )
            for k in update_key_name(self.unexpected_keys):
                status = _style("UNEXPECTED", "orange")
                rows.append([k, status, "", ""])

        if self.missing_keys:
            tips += f"\n- {_style('MISSING:', 'red')}\t" + _style(
                "those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.",
                "italic",
            )
            for k in update_key_name(self.missing_keys):
                status = _style("MISSING", "red")
                rows.append([k, status, ""])

        if self.mismatched_keys:
            tips += f"\n- {_style('MISMATCH:', 'yellow')}\t" + _style(
                "ckpt weights were loaded, but they did not match the original empty weight shapes.", "italic"
            )
            iterator = {a: (b, c) for a, b, c in self.mismatched_keys}
            for key, (shape_ckpt, shape_model) in update_key_name(iterator).items():
                status = _style("MISMATCH", "yellow")
                data = [
                    key,
                    status,
                    f"Reinit due to size mismatch - ckpt: {str(shape_ckpt)} vs model:{str(shape_model)}",
                ]
                rows.append(data)

        if self.conversion_errors:
            tips += f"\n- {_style('CONVERSION:', 'purple')}\t" + _style(
                "originate from the conversion scheme", "italic"
            )
            for k, v in update_key_name(self.conversion_errors).items():
                status = _style("CONVERSION", "purple")
                _details = f"\n\n{v}\n\n"
                rows.append([k, status, _details])

        stage = None
        if model is not None:
            stage = getattr(model, "_pp_stage", None)
        if stage is not None:
            owned, skipped = _pp_report_key_owners(stage, model, self.skipped_pp_keys)
            for status, color, note, key_owners in (
                ("OWNED", "green", "checkpoint weights loaded on this pipeline stage.", owned),
                ("SKIPPED", "dim", "checkpoint weights owned by another pipeline stage.", skipped),
            ):
                if key_owners:
                    tips += f"\n- {_style(f'{status}:', color)}\t{note}"
                    for key, owner in update_key_name(key_owners).items():
                        rows.append([key, _style(status, color), f"PP rank {owner}"])

        # If nothing is wrong, return None
        if len(rows) == 0:
            return None

        headers = ["Key", "Status"]
        if term_w > 200:
            headers += ["Details"]
        else:
            headers += ["", ""]
        table = _make_table(rows, headers=headers)
        report = table + tips

        return report


def dot_natural_key(s: str):
    """
    Sort key for state-dict names: split on `"."` and sort digits numerically and strings alphabetically. It emits a
    tuple at each point to sort ints first and strings second to avoid int-string comparison failures.
    """
    parts = []
    for part in s.split("."):
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            # This will remove all trailing digit characters, as `rstrip` actually considers it as a set of chars
            text_part = part.rstrip("0123456789")
            trailing_digits = part[len(text_part) :]
            # Sort numeric suffixes numerically, so `shard_2` precedes `shard_11` for example
            if trailing_digits != "":
                parts.append((1, text_part, int(trailing_digits)))
            else:
                parts.append((1, text_part))
    return parts


def rename_source_key(
    source_key: str,
    weight_renamings: list[WeightRenaming],
    weight_converters: list[WeightConverter],
    base_model_prefix: str | None = None,
    meta_state_dict: dict | None = None,
    reverse: bool = False,
) -> tuple[str, str | None]:
    """
    Rename a checkpoint key by first applying all `WeightRenaming`s, then at most one `WeightConverter`.
    This means that the `WeightConverter` must specify the final name for the key, but some `WeightRenaming`s can act on other
    parts of that key beforehand. If `reverse` is True, i.e. when reverting all the `WeightTransform`s, the opposite is performed
    to be coherent and correctly respect the `scope_prefix` of all `WeightTransform`s: first try to match 1 `WeightConverter`, and only
    then try to apply all `WeightRenaming`s. Indeed, in reverse mode, all reverse transforms should be applied in the opposite order to
    be consistent.
    Note that we proceed in this way because there is no need for a Converter-then-Rename order because Converters act only on specific
    leaf patterns, so no subsequent Renamings should ever target their output.

    Args:
        source_key (`str`):
            The original checkpoint key to rename.
        weight_renamings (`list[WeightRenaming]`):
            Applied in order; every matching renaming fires (they may chain).
        weight_converters (`list[WeightConverter]`):
            Applied after all renamings; at most one may match. Subsequent converters are skipped.
        base_model_prefix (`str`, *optional*):
            Base-model prefix to add or strip when both `base_model_prefix` and `meta_state_dict` are given.
        meta_state_dict (`dict`, *optional*):
            Meta state dict used to decide whether `base_model_prefix` should be added or stripped.
        reverse (`bool`, *optional*):
            This specifies if we are reverting all the `WeightTransform`s (saving back original format).

    Returns:
        `tuple[str, str | None]`: The renamed key and the matched converter's source pattern
        (or `None` if no converter matched).
    """
    renamed_key = source_key
    converter_source_pattern = None
    # 1. If `reverse` is False: apply all renamings in turns (if multiple match, it's the responsibility of the mappings to make sure
    # they are coherent).
    # Else, first apply the `WeightConverter` if any
    first_iterable = weight_renamings if not reverse else weight_converters
    for transform in first_iterable:
        renamed_key, source_pattern = transform.rename_source_key(renamed_key)
        # Only break after match is we are using the `WeightConverter`s here, i.e. `reverse` is True
        if reverse and source_pattern is not None:
            converter_source_pattern = source_pattern
            break

    # 2. If `reverse` is False: apply renaming through weight conversions on the key if we have any WeightConverter (here we stop after
    # the first match, as we assume only 1 converter can match any source key).
    # Else, apply all the `WeightRenaming`s
    second_iterable = weight_converters if not reverse else weight_renamings
    for transform in second_iterable:
        renamed_key, source_pattern = transform.rename_source_key(renamed_key)
        # Only break after match is we are using the `WeightConverter`s here, i.e. `reverse` is False
        if not reverse and source_pattern is not None:
            converter_source_pattern = source_pattern
            break

    # 3. check if we need to add or remove base_model_prefix if necessary (only during loading, not saving)
    if base_model_prefix is not None and meta_state_dict is not None:
        if (
            renamed_key.startswith(base_model_prefix)
            and meta_state_dict.get(re.sub(f"^{base_model_prefix}.", "", renamed_key, count=1)) is not None
        ):
            renamed_key = re.sub(f"^{base_model_prefix}.", "", renamed_key, count=1)
        elif meta_state_dict.get(f"{base_model_prefix}.{renamed_key}") is not None:
            renamed_key = f"{base_model_prefix}.{renamed_key}"

    return renamed_key, converter_source_pattern


def _add_unmatched_checkpoint_key(
    key: str,
    model: Model,
    loading_info: LoadStateDictInfo,
) -> None:
    """Classify a checkpoint key that does not match the current model state dict.

    During pipeline-parallel loading, each stage receives keys for the entire model even though its local state dict
    contains only that stage's parameters. A key owned by another stage is therefore recorded as intentionally skipped
    instead of unexpected. Without the key, it would be considered unexpected.
    """
    stage = getattr(model, "_pp_stage", None)
    if stage is None:
        loading_info.unexpected_keys.add(key)
        return

    base_model = getattr(model, model.base_model_prefix)
    owner_rank = stage.find_rank_for_key(key, len(base_model.layers), model.base_model_prefix)
    owned_by_another_stage = owner_rank is not None and owner_rank != stage.pp_rank

    if owned_by_another_stage:
        loading_info.skipped_pp_keys.add(key)
    else:
        loading_info.unexpected_keys.add(key)


def convert_and_load_state_dict_in_model(
    model: PreTrainedModel,
    state_dict: dict[str, Any],
    load_config: LoadStateDictConfig,
    disk_offload_index: dict | None = None,
):
    base_model_prefix = model.base_model_prefix
    device_map = load_config.device_map or {"": "cpu"}
    dtype = load_config.dtype
    disk_offload_folder = load_config.disk_offload_folder
    offload_buffers = load_config.offload_buffers
    dtype_plan = load_config.dtype_plan or {}
    weight_mapping = load_config.weight_mapping or []
    meta_model_state_dict = model.state_dict()
    model_buffers = {k for k, _ in model.named_buffers()}

    # We start from all missing keys, and we will remove/add them from the proper containers as loading advances
    loading_info = LoadStateDictInfo(
        missing_keys=set(meta_model_state_dict.keys()),
        unexpected_keys=set(),
        mismatched_keys=set(),
        conversion_errors={},
        error_msgs=[],
        skipped_pp_keys=set(),
    )

    # We use threading by default, if not explicitly deactivated via env variable. If we have to offload,
    # we cannot use it either to control the memory as we are under memory constraints, so we need to be sequential.
    # When doing on-the-fly quantization, we also use sync loading to avoid worker threads loading full-precision
    # tensors to GPU faster than the main thread can quantize them, which would cause a large memory spike.
    if (
        #is_env_variable_true("HF_DEACTIVATE_ASYNC_LOAD") or
        "disk" in device_map.values()
    ):
        thread_pool = None
    else:
        thread_pool = ThreadPoolExecutor(max_workers=GLOBAL_WORKERS)

    renamings = [
        entry
        for entry in weight_mapping
        if isinstance(entry, WeightRenaming)
    ]
    converters = [
        entry
        for entry in weight_mapping
        if isinstance(entry, WeightConverter)
    ]
    param_name_to_load: dict[str, WeightRenaming | WeightConverter] = {}

    if dtype_plan != {}:
        dtype_policy_alt, dtype_policy_by_group_name, _ = build_glob_alternation(list(dtype_plan.keys()))

    pattern_to_converter = {
        k: converter
        for converter in converters
        for k in converter.source_patterns
    }
    print(f"Pattern to convert - {pattern_to_converter}")

    state_dict = sorted(
        state_dict.items(),
        key=lambda kv: dot_natural_key(kv[0])
    )
    for original_key, tensor in state_dict:
        # 1. Rename the key according to all renaming and weight conversion patterns.
        renamed_key, source_pattern = rename_source_key(
            original_key, renamings, converters, base_model_prefix, meta_model_state_dict
        )
        print("="*50)
        print(">>", renamed_key, original_key, source_pattern)
        print("="*50)
        if renamed_key not in meta_model_state_dict and original_key in meta_model_state_dict:
            # Key should probably not have been renamed but we might need the `prefix` to be added.
            renamed_key, source_pattern = rename_source_key(
                original_key, [], [], base_model_prefix=base_model_prefix, meta_state_dict=meta_model_state_dict
            )

        # 2. finally, collect the tensor into the proper converter
        if renamed_key in meta_model_state_dict:
            empty_param = meta_model_state_dict.get(renamed_key)
            # If we enter here, we have a WeightConverter operation to perform
            if source_pattern is not None:
                new_converter = deepcopy(pattern_to_converter[source_pattern])
                # each target key gets its own converter instance
                mapping = param_name_to_load.setdefault(renamed_key, new_converter)
            # Otherwise, only potential renaming
            else:
                mapping = param_name_to_load.setdefault(renamed_key, WeightRenaming(original_key, renamed_key))
                source_pattern = original_key

            # 3. Handle dtype casting

            _dtype = dtype
            if dtype_plan != {} and dtype_policy_alt.search(renamed_key):
                matched_dtype_pattern = dtype_policy_alt.search(renamed_key)
                if matched_dtype_pattern is not None:
                    _dtype = dtype_plan[dtype_policy_by_group_name[matched_dtype_pattern.lastgroup]]
            elif empty_param is not None and empty_param.dtype != _dtype:
                _dtype = empty_param.dtype  # usually correct when initializing

            # Per-expert sharding (EP) needs `tensor_idx` = the expert index so the
            # distributed op selects whole experts. The signal is a `MergeModulelist`
            # in the chain; it isn't always `operations[0]` (e.g. an FP8 quantizer
            # prepends a scale-decode op), so scan the whole chain rather than just the head.
            tensor_idx = (
                len(mapping.collected_tensors.get(source_pattern, []))
                if isinstance(mapping, WeightConverter)
                and any(isinstance(op, MergeModulelist) for op in mapping.operations)
                else None
            )

            # 4. Handle DTensor sharding or device_map placement
            param_device = get_device(device_map, renamed_key, valid_torch_device=True)
            sharding_op = None
            if is_dtensor(empty_param):
                sharding_op = DtensorShardOperation(empty_param)

            # Some parameters are so large (qwen4_exp ple_embedding is about ~95 GiB) that we cannot afford to perform the Operations
            # directly on the device, as it will completely blow up the memory during the ops memory spike. So defer to "cpu", then
            # accelerate will take care of putting back on correct device after loading
            # Note that we only do it with `device_map` but not with `tp_plan`, as tp will perform local sharding before, so memory
            # spike during conversion ops should be fine
            if sharding_op is None and isinstance(mapping, WeightConverter) and mapping.force_cpu:
                param_device = "cpu"

            future_or_tensor = spawn_materialize(
                thread_pool,
                tensor,
                param_device,
                _dtype,
                sharding_op=sharding_op,
                tensor_idx=tensor_idx,
            )

            mapping.add_tensor(renamed_key, original_key, source_pattern, future_or_tensor)
        elif source_pattern is not None:  # add all target keys as unexpected
            mapping = pattern_to_converter[source_pattern]
            for k in mapping.target_patterns:
                _add_unmatched_checkpoint_key(
                    renamed_key.replace(mapping.target_patterns[0], k),
                    model,
                    loading_info,
                )
        else:
            _add_unmatched_checkpoint_key(renamed_key, model, loading_info)
    try:
        for first_param_name, mapping in tqdm(param_name_to_load.items(), desc="Loading weights"):
            try:
                realized_value = mapping.convert(
                    first_param_name,
                    model=model,
                    config=model.config,
                    loading_info=loading_info,
                )
                for target_name, param in realized_value.items():
                    param = param[0] if isinstance(param, list) else param
                    param_device = get_device(device_map, target_name)
                    # Exception for params that are so huge that they need conversions on cpu no matter what - we put them back on
                    # the correct device now (if the device_map managed to make them fit on any device, as usually they will stay on
                    # cpu and this is a no-op)
                    if getattr(mapping, "force_cpu", False) and param_device != "disk":
                        param = param.to(param_device)
                    # Offloading support
                    if param_device == "disk" and (target_name not in model_buffers or offload_buffers):
                        disk_offload_index = offload_and_maybe_resave_param(
                            target_name, param, loading_info, disk_offload_folder, disk_offload_index, mapping
                        )
                    else:
                        set_param_for_module(
                            model,
                            target_name,
                            param,
                            loading_info,
                            hf_quantizer,
                        )

                # Cleanup all the tensors that were gathered before next iteration
                del realized_value

            except SkipParameters:
                continue

    # Close the pool, independently of whether the code was interrupted or finished successfully
    finally:
        if thread_pool is not None:
            # `cancel_futures=True` in case the program was interrupted, to avoid wasting time on exit
            thread_pool.shutdown(wait=False, cancel_futures=True)

    # Keep the current weight conversion mapping for later saving (in case it was coming directly from the user), but
    # only if it was used, i.e. it matched any weight from the checkpoints
    model_specific_conversions = [conversion for conversion in weight_mapping if conversion.was_used()]
    model._weight_conversions = model_specific_conversions

    return loading_info, disk_offload_index


class Model(nn.Module):
    
    # General model properties
    config_class = ModelConfig
    base_model_prefix: str = ""

    # Specific dtype upcasting
    # `_keep_in_fp32_modules` will upcast to fp32 only if the requested dtype is fp16
    # `_keep_in_fp32_modules_strict` will upcast to fp32 independently if the requested dtype is fp16 or bf16
    _keep_in_fp32_modules: set[str] | list[str] | None = None
    _keep_in_fp32_modules_strict: set[str] | list[str] | None = None


    def __init__(self, config: ModelConfig):
        super().__init__()

        if not isinstance(config, self.config_class):
            raise TypeError(
                f"config must be an instance of "
                f"{self.config_class.__name__}"
            )

        self.config = config

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        It is also used to obtain all correct static properties (parallelism plans, tied_weights_keys, _keep_in_fp32_modules, etc)
        correctly in the case of composite models (that is, the top level model should know about those properties from its children).
        """
        # Attach the different parallel plans and tied weight keys to the top-most model, so that everything is
        # easily available.
        self.init_parallel_plans()
        # Current submodel should register its tied weights
        self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=False)
        # Current submodel should register its `_keep_in_fp32_modules`
        self._keep_in_fp32_modules = set(self._keep_in_fp32_modules or [])
        self._keep_in_fp32_modules_strict = set(self._keep_in_fp32_modules_strict or [])
        # Current submodel must register its `_no_split_modules`/`_skip_keys_device_placement` as well for device_map
        self._no_split_modules = set(self._no_split_modules or [])
        self._skip_keys_device_placement = set(self._skip_keys_device_placement or [])
        # Current submodel must register the `_keys_to_ignore_on_load_unexpected/missing`
        self._keys_to_ignore_on_load_unexpected = set(self._keys_to_ignore_on_load_unexpected or [])
        self._keys_to_ignore_on_load_missing = set(self._keys_to_ignore_on_load_missing or [])
        self._keys_to_ignore_on_save = set(self._keys_to_ignore_on_save or [])

        # Iterate over children only: as the final model is created, this is enough to gather the properties from all submodels.
        # This works because the way the `__init__` and `post_init` are called on all submodules is depth-first in the graph
        for name, module in self.named_children():
            # Always attach the keys of the children (if the children's config says to NOT tie, then it's empty)
            if tied_keys := getattr(module, "all_tied_weights_keys", None):
                self.all_tied_weights_keys.update({f"{name}.{k}": f"{name}.{v}" for k, v in tied_keys.copy().items()})
            # Record keep_in_fp_32 modules from the children as well
            if keep_fp32 := getattr(module, "_keep_in_fp32_modules", None):
                self._keep_in_fp32_modules.update(keep_fp32)
            if keep_fp32_strict := getattr(module, "_keep_in_fp32_modules_strict", None):
                self._keep_in_fp32_modules_strict.update(keep_fp32_strict)
            # Record `_no_split_modules`/`_skip_keys_device_placement` from the children
            if no_split := getattr(module, "_no_split_modules", None):
                self._no_split_modules.update(no_split)
            if skip_keys := getattr(module, "_skip_keys_device_placement", None):
                self._skip_keys_device_placement.update(skip_keys)
            # Record `_keys_to_ignore_on_load_unexpected/missing` from the children - note that we do not add the name (prefix)
            # of the current module to the child's key, as this is matched by regex anyway and adding prefix could break the regex
            if ignore_unexpected := getattr(module, "_keys_to_ignore_on_load_unexpected", None):
                self._keys_to_ignore_on_load_unexpected.update(ignore_unexpected)
            if ignore_missing := getattr(module, "_keys_to_ignore_on_load_missing", None):
                self._keys_to_ignore_on_load_missing.update(ignore_missing)
            # This one is matched exactly, not by regex, so we need to add the prefix
            if ignore_save := getattr(module, "_keys_to_ignore_on_save", None):
                self._keys_to_ignore_on_save.update({f"{name}.{k}" for k in ignore_save})

        # Maybe initialize the weights and tie the keys
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()


    def _get_dtype_plan(self, dtype: torch.dtype) -> dict:
        """Create the dtype_plan describing modules/parameters that should use the `keep_in_fp32` flag."""
        dtype_plan = {}

        # The _keep_in_fp32_modules flag is only used to avoid bf16 -> fp16 casting precision issues. It was introduced
        # in case of force loading a model that should stay in bf16 in fp16
        # See https://github.com/huggingface/transformers/issues/20287 for details.
        if (
            self._keep_in_fp32_modules is not None and
            dtype == torch.float16
        ):
            dtype_plan.update(
                dict.fromkeys(
                    self._keep_in_fp32_modules,
                    torch.float32
                )
            )

        # The _keep_in_fp32_modules_strict was introduced to always force upcast to fp32, for both fp16 and bf16
        if (self._keep_in_fp32_modules_strict is not None and
            dtype in (torch.float16, torch.bfloat16)
        ):
            dtype_plan.update(
                dict.fromkeys(
                    self._keep_in_fp32_modules_strict,
                    torch.float32
                )
            )

        return dtype_plan
    
    @classmethod
    def get_init_context(
        cls,
        dtype: torch.dtype,
        allow_all_kernels: bool | None
    ):
        # Need to instantiate with correct dtype
        init_contexts = [
            local_torch_dtype(dtype, cls.__name__),
            #init.no_tie_weights(),
            #apply_patches()
        ]
        # Needed as we cannot forward the `allow_all_kernels` arg in the model's __init__
        if allow_all_kernels:
            init_contexts.append(allow_all_hub_kernels())
        else:
            # meta_device_safe_creation_ops patches torch.linspace to default to CPU
            # so that custom models calling .item() during __init__ (e.g. drop-path
            # schedules) don't crash on meta tensors.
            init_contexts.extend([
                torch.device("meta"),
                #init.meta_device_safe_creation_ops()
            ])

        return init_contexts
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        ignore_mismatched_sizes: bool = False,
        use_safetensors: bool | None = None,
        weights_only: bool = True,
        disable_mmap: bool | None = None,
        **kwargs,
    ):
        dtype = kwargs.pop("dtype", None)
        torch_dtype = kwargs.pop("torch_dtype", None)  # kept for BC
        device_map = kwargs.pop("device_map", None)
        
        adapter_kwargs = (kwargs.pop("adapter_kwargs", {}) or {}).copy()
        adapter_name = kwargs.pop("adapter_name", "default")

        device_mesh = kwargs.pop("device_mesh", None)
        
        allow_all_kernels = kwargs.pop("allow_all_kernels", False)
        use_kernels = kwargs.pop("use_kernels", False)
        kernel_config = kwargs.pop("kernel_config", None)
        
        # For BC on torch_dtype argument
        if torch_dtype is not None:
            dtype = dtype if dtype is not None else torch_dtype
        if dtype is None:
            dtype = "auto"
            
        model_kwargs = {}

        device_map = check_and_set_device_map(device_map)  # warn, error and fix the device map
        
        # Get Config
        config = (
            cls.config_class
            .from_pretrained(
                pretrained_model_name_or_path
            )
        )

        # Get tensor file
        files = _resolve_checkpoint_files(
            pretrained_model_name_or_path
        )

        config, dtype = _get_dtype(
            dtype, files, config, weights_only
        )
        config.name_or_path = pretrained_model_name_or_path

        model_init_context = cls.get_init_context(
            dtype, allow_all_kernels
        )
        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        with ContextManagers(model_init_context):
            model = cls(config, *model_args, **model_kwargs)
            #patch_output_recorders(model)
            
        # Create the dtype_plan to potentially use the `keep_in_fp32` flags (this needs to be called on the already
        # instantiated model, as the flags can be modified by instances sometimes)
        dtype_plan = model._get_dtype_plan(dtype)

        # Obtain the weight conversion mapping for this model if any are registered and apply to all submodels recursively
        weight_conversions = get_model_conversion_mapping(
            model
        )
        print(weight_conversions)

        # Prepare the full device map
        if device_map is not None:
            device_map = _get_device_map(model, device_map, max_memory, hf_quantizer)

        # Finalize model weight initialization
        offload_folder = kwargs.pop("offload_folder", None)
        offload_buffers = kwargs.pop("offload_buffers", False)
        
        load_config = LoadStateDictConfig(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            device_map=device_map,
            disk_offload_folder=offload_folder,
            offload_buffers=offload_buffers,
            dtype=dtype,
            dtype_plan=dtype_plan,
            device_mesh=device_mesh,
            weights_only=weights_only,
            weight_mapping=weight_conversions,
            use_safetensors=use_safetensors,
            disable_mmap=disable_mmap,
        )
        loading_info, disk_offload_index = cls._load_pretrained_model(
            model, files, load_config
        )
        loading_info = cls._finalize_model_loading(model, load_config, loading_info)
        model.eval()  # Set model in evaluation mode to deactivate Dropout modules by default
        
        return model
        
        state_dict = load_state_dict(files)

        
        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=False,
        )

        if missing:
            raise RuntimeError(
                f"Missing model weights: {missing}"
            )

        if unexpected:
            raise RuntimeError(
                f"Unexpected model weights: {unexpected}"
            )

        return model

    @staticmethod
    def _load_pretrained_model(
        model: "Model",
        files: list[Path] | None,
        load_config: LoadStateDictConfig,
        expected_keys: list[str] | None = None,
    ) -> tuple[LoadStateDictInfo, dict]:
        """Perform the actual loading of some checkpoints into a `model`, by reading them from disk and dispatching them accordingly."""

        # Model's definition arriving here is final (TP hooks added, quantized layers replaces)
        expected_keys = list(model.state_dict().keys()) if expected_keys is None else expected_keys

        # This offload index if for params explicitly on the "disk" in the device_map
        disk_offload_index = None
        # Prepare parameters offloading if needed
        if load_config.device_map is not None and "disk" in load_config.device_map.values():
            raise NotImplementedError

        error_msgs = []

        all_pointer = set()
        if (
            files is not None and
            str(files[0]).endswith(".safetensors")
        ):
            from safetensors import safe_open

            merged_state_dict = {}
            for file in files:
                # if load_config.disable_mmap or _is_on_hf_mount(file):
                #     with open(file, "rb") as _fh:
                #         merged_state_dict.update(_safe_load_bytes(_fh.read()))
                #     continue
                is_mps = load_config.device_map is not None and any(
                    (d.type if isinstance(d, torch.device) else d) == "mps"
                    for d in load_config.device_map.values()
                )
                # Use pread on MPS (mmap incompatible) and Windows (mmap reserves
                # copy-on-write commit charge for the entire file, exhausting memory
                # for large multi-shard checkpoints).
                if is_mps:
                    backend, device = "pread", "mps"
                elif sys.platform == "win32":
                    backend, device = "pread", "cpu"
                else:
                    backend, device = "mmap", "cpu"
                file_pointer = safe_open(file, framework="pt", device=device, backend=backend)
                all_pointer.add(file_pointer)
                for k in file_pointer.keys():
                    merged_state_dict[k] = file_pointer.get_slice(k)  # don't materialize yet
        # Checkpoints are .bin
        elif files is not None:
            raise NotImplementedError(".bin pytorch")
            merged_state_dict = {}
            for ckpt_file in checkpoint_files:
                merged_state_dict.update(load_state_dict(ckpt_file, disable_mmap=load_config.disable_mmap))
        else:
            raise ValueError("Neither a state dict nor checkpoint files were found.")

        loading_info, disk_offload_index = convert_and_load_state_dict_in_model(
            model=model,
            state_dict=merged_state_dict,
            load_config=load_config,
            disk_offload_index=disk_offload_index,
        )

        # finally close all opened file pointers
        for k in all_pointer:
            k.__exit__(None, None, None)

        return loading_info, disk_offload_index

    @staticmethod
    def _finalize_model_loading(
        model,
        load_config: LoadStateDictConfig,
        loading_info: LoadStateDictInfo
    ) -> LoadStateDictInfo:
        """Perform all post processing operations after having loaded some checkpoints into a model, such as moving
        missing keys from meta device to their expected device, reinitializing missing weights according to proper
        distributions, tying the weights and logging the loading report."""
        try:
            # Marks tied weights as `_is_hf_initialized` to avoid initializing them (it's very important for efficiency)
            model.mark_tied_weights_as_initialized(loading_info)

            # Move missing (and potentially mismatched) keys and non-persistent buffers back to their expected device from
            # meta device (because they were not moved when loading the weights as they were not in the loaded state dict)
            model._move_missing_keys_from_meta_to_device(
                loading_info.missing_and_mismatched(),
                load_config.device_map,
                load_config.device_mesh,
            )

            # Correctly initialize the missing (and potentially mismatched) keys (all parameters without the `_is_hf_initialized` flag)
            model._initialize_missing_keys(load_config.is_quantized)

            # Tie the weights
            model.tie_weights(missing_keys=loading_info.missing_keys, recompute_mapping=False)

            # Adjust missing and unexpected keys
            model._adjust_missing_and_unexpected_keys(loading_info)
        finally:
            log_state_dict_report(
                model=model,
                pretrained_model_name_or_path=load_config.pretrained_model_name_or_path,
                ignore_mismatched_sizes=load_config.ignore_mismatched_sizes,
                loading_info=loading_info,
                logger=logger,
            )

        return loading_info

    def _move_missing_keys_from_meta_to_device(
        self,
        missing_keys: list[str],
        device_map: dict | None,
        device_mesh: "DeviceMeshLike | None",
        hf_quantizer: HfQuantizer | None = None,
    ) -> None:
        """Move missing params/buffers off meta to their target device.

        Loaded weights are handled earlier in `convert_and_load_state_dict_in_model`
        via `DtensorShardOperation` and `set_param_for_module`. This only
        materializes keys that were not loaded (or mismatched) so
        `_initialize_missing_keys` can run proper init on them.
        """
        is_quantized = hf_quantizer is not None

        # In this case we need to move everything back
        # if is_fsdp_enabled() and not is_local_dist_rank_0() and not is_quantized:
        #     for key, param in self.named_parameters():
        #         value = torch.zeros_like(param, device="cpu")
        #         _load_parameter_into_model(self, key, value)
        #     for key, buffer in self.named_buffers():
        #         value = torch.zeros_like(buffer, device="cpu")
        #         _load_parameter_into_model(self, key, value)
        #     return

        # The tied weight keys are in the "missing" usually, but they should not be moved (they will be tied anyway)
        # This is especially important because if they are moved, they will lose the `_is_hf_initialized` flag, and they
        # will be re-initialized for nothing (which can be quite long)
        for key in missing_keys - self.all_tied_weights_keys.keys():
            param = self.get_parameter_or_buffer(key)
            param_device = get_device(device_map, key, valid_torch_device=True)
            value = torch.empty_like(param, device=param_device)
            # For TP, we may need to shard the param
            if is_dtensor(param):
                local = torch.empty(param._local_tensor.shape, dtype=param.dtype, device=param_device)
                value = torch.nn.Parameter(
                    _dtensor_from_local_like(local, param),
                    requires_grad=param.requires_grad,
                )
            _load_parameter_into_model(self, key, value)
        # We need to move back non-persistent buffers as well, as they are not part of loaded weights anyway
        for key, buffer in self.named_non_persistent_buffers():
            buffer_device = get_device(device_map, key, valid_torch_device=True)
            value = torch.empty_like(buffer, device=buffer_device)
            _load_parameter_into_model(self, key, value)

    def _initialize_missing_keys(self, is_quantized: bool) -> None:
        """
        Initialize the missing keys (keys that are part of the model parameters, but were NOT found in the loaded state dicts), according to
        `_initialize_weights`. Indeed, since the corresponding weights are missing from the state dict, they will not be replaced and need to
        be initialized correctly (i.e. weight initialization distribution).

        Also marks non-missing params/buffers with `_is_hf_initialized` and propagates this flag to modules,
        so that `_initialize_weights` can skip fully-initialized modules entirely.
        """
        if is_fsdp_enabled() and not is_local_dist_rank_0():
            # Handle FSDP edge case when using cpu ram efficient loading to ensure it is marked as initialized
            # since it will get its weights broadcasted from rank0
            # We actually need to do that only because we want to re-initialize non-persistent buffers with correct values.
            # Everything else in the state_dict will be gathered from rank0, so we don't need re-initialization.
            # We could simply early return after buffer inits if we had a way to init only the non-persistent buffers
            for key in self.state_dict():
                try:
                    param_or_buffer = self.get_parameter_or_buffer(key)
                    param_or_buffer._is_hf_initialized = True
                except AttributeError:
                    pass  # may happen when handling pre-quantized weights
            self._is_hf_initialized = True

        self.initialize_weights()

    def _adjust_missing_and_unexpected_keys(self, loading_info: LoadStateDictInfo) -> None:
        """Adjust the `missing_keys` and `unexpected_keys` based on current model's exception rules, to avoid
        raising unneeded warnings/errors. This is performed in-place.
        """
        # Old checkpoints may have keys for rotary_emb.inv_freq for each layer, however we moved this buffer to the main model
        # (so the buffer name has changed). Remove them in such a case. This is another exception that was not added to
        # `_keys_to_ignore_on_load_unexpected` as it touches many models -> we add it manually to the existing patterns
        has_inv_freq_buffers = any(buffer.endswith("rotary_emb.inv_freq") for buffer, _ in self.named_buffers())
        additional_unexpected_patterns = {r"rotary_emb\.inv_freq"} if has_inv_freq_buffers else set()
        # Same idea for `position_ids`: used to be a persistent buffer, now `persistent=False` in most models.
        has_position_ids_buffers = any(buffer.endswith("position_ids") for buffer, _ in self.named_buffers())
        if has_position_ids_buffers:
            additional_unexpected_patterns.add(r"(^|\.)position_ids$")

        missing_patterns = self._keys_to_ignore_on_load_missing or set()
        unexpected_patterns = (self._keys_to_ignore_on_load_unexpected or set()) | additional_unexpected_patterns
        ignore_missing_regex, ignore_unexpected_regex = None, None
        if len(missing_patterns) > 0:
            ignore_missing_regex = re.compile("|".join(rf"({pattern})" for pattern in missing_patterns))
        if len(unexpected_patterns) > 0:
            ignore_unexpected_regex = re.compile("|".join(rf"({pattern})" for pattern in unexpected_patterns))

        # Clean-up missing keys
        if ignore_missing_regex is not None:
            loading_info.missing_keys = {
                key for key in loading_info.missing_keys if ignore_missing_regex.search(key) is None
            }

        # Clean-up unexpected keys
        if ignore_unexpected_regex is not None:
            loading_info.unexpected_keys = {
                key for key in loading_info.unexpected_keys if ignore_unexpected_regex.search(key) is None
            }

    def mark_tied_weights_as_initialized(self, loading_info):
        """Adds the `_is_hf_initialized` flag on parameters that will be tied, in order to avoid initializing them
        later as they will be tied (overwritten) anyway.
        This is very important as most embeddings are tied, and they are huge params (vocabularies are often 256k), so
        running inits on them is very costly."""
        for tied_param in getattr(self, "all_tied_weights_keys", {}).keys():
            param = self.get_parameter(tied_param)
            setattr(param, "_is_hf_initialized", True)

        # Some custom code models define module tying (not parameter tying) in their __init__. When modules themselves are shared,
        # weights inside both modules appear in the `state_dict` but only one will appear in the safetensors checkpoints
        # as they are inherently tied because the 2 modules are the same object. In this case, once we load a parameter
        # inside one of the 2 modules, the other will also automatically be loaded and will have the `_is_hf_initialized`
        # flag (because we call `setattr` with the loaded param on the module, which is the same object), but its counterpart
        # will still appear as a missing key as we never get it out of the set (because it appears in the state_dict as well).
        # So we remove it now - otherwise it's considered missing and will be wrongly reinitialized
        # Note: this is never an issue in main Transformers, as we never do module-tying, only parameter-tying, and we know
        # which params are supposed to be tied to which other params
        # if self.is_custom_code():
        #     # Remove those that are already initialized, but appear as missing due to module tying (only if they are not known
        #     # tied weights, i.e. we did not explicitly mark them as initialized just above)
        #     loading_info.missing_keys = {
        #         key
        #         for key in loading_info.missing_keys
        #         if key in self.all_tied_weights_keys
        #         or not getattr(self.get_parameter_or_buffer(key), "_is_hf_initialized", False)
        #     }
        

def _resolve_checkpoint_files(
    pretrained_model_name_or_path: str | Path,
) -> list[Path]:
    # Safetensors, single file.
    try:
        return [
            ensure_file(
                pretrained_model_name_or_path,
                SAFE_WEIGHTS_NAME,
            )
        ]
    except FileNotFoundError:
        pass

    # PyTorch, single file.
    try:
        return [
            ensure_file(
                pretrained_model_name_or_path,
                WEIGHTS_NAME,
            )
        ]
    except FileNotFoundError:
        pass

    raise FileNotFoundError(
        "Could not find model weights. Expected one of: "
        f"{SAFE_WEIGHTS_NAME!r}, {WEIGHTS_NAME!r}"
    )
