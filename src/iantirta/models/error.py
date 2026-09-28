# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of huggingface_hub, improved by iantirta.com
"""error class and utilities."""

from __future__ import annotations


# Currently Not Yet Implemented Error
class CurrentlyNotImplementedError(NotImplementedError):
    def __init__(self, *args):
        error_message = (
            "Currently iantirta-models not yet support "
            f"for {args}\n"
        )
        super().__init__(error_message)
    