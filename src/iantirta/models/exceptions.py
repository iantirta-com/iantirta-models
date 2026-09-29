# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of huggingface_hub, improved by iantirta.com
from __future__ import annotations


class YetToImplement(NotImplementedError):
    def __init__(self, *args):
        error_message = (
            "Currently iantirta-models is still in developments.\n"
            f"  message: {args}\n"
        )
        super().__init__(error_message)
