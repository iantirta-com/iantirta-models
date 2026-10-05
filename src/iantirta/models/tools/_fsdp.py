from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._torch import is_torch_distributed_available

if TYPE_CHECKING:
    from torch import nn


def is_fsdp_managed_module(module: nn.Module) -> bool:
    """Check if a module is managed by FSDP (1 or 2)."""
    if not is_torch_distributed_available():
        return False

    # FSDP2: attribute set by apply_fsdp2()
    if getattr(module, "_is_fsdp_managed_module", False):
        return True
    # FSDP1: wrapped by FullyShardedDataParallel
    from torch.distributed.fsdp import FullyShardedDataParallel

    return isinstance(module, FullyShardedDataParallel)
