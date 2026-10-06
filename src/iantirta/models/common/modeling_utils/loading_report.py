# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of transformers, improved by iantirta.com

import logging
import re
import shutil
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any

from .distributed.pipeline_parallel import PipelineStage

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


PALETTE = {
    "reset": "\x1b[0m",
    "red": "\x1b[31m",
    "yellow": "\x1b[33m",
    "orange": "\x1b[38;5;208m",
    "purple": "\x1b[35m",
    "green": "\x1b[32m",
    "bold": "\x1b[1m",
    "italic": "\x1b[3m",
    "dim": "\x1b[2m",
}


def _style(s, color):
    """Return color/style-formatted input `s` if `sys.stdout` is interactive, e.g. connected to a terminal."""
    if sys.stdout is not None and sys.stdout.isatty():
        return f"{PALETTE[color]}{s}{PALETTE['reset']}"
    else:
        return s


def _get_terminal_width(default=80):
    try:
        return shutil.get_terminal_size().columns
    except Exception:  # noqa: BLE001
        return default


def _pp_report_key_owners(
    stage: PipelineStage, model, skipped_pp_keys: set[str]
) -> tuple[dict[str, int], dict[str, int]]:
    """Map checkpoint keys to their owning PP rank for the load report."""
    prefix = model.base_model_prefix
    num_layers = len(getattr(model, prefix).layers)

    owned = dict.fromkeys(model.state_dict(), stage.pp_rank)
    skipped = {
        key: owner
        for key in skipped_pp_keys
        if (owner := stage.find_rank_for_key(key, num_layers, prefix)) is not None
    }
    return owned, skipped


