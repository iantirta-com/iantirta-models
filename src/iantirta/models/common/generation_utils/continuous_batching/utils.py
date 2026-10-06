import queue
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil, log2
from typing import Any

import torch

from iantirta.models.common.attentions.paged_utils.requests import (
    FutureRequestState,
    RequestState,
    RequestStatus,
)

from ...configuration_utils import PreTrainedConfig


@dataclass
class WorkloadHints:
    """A tiny dataclass containing hints to help choose good continuous batching defaults"""

    max_prompt_length: int = 0
    max_generated_length: int = 0
    num_requests: int = 0


class ThreadLocalCounter(threading.local):
    def __init__(self) -> None:
        self.value = 0

def drain_queue(request_queue: queue.Queue) -> list[RequestState]:
    """Drains a queue and returns a list of RequestStates."""
    new_states: list[RequestState] = []
    while not request_queue.empty():
        try:
            state = request_queue.get_nowait()
            if state is not None:
                new_states.append(state)
        except queue.Empty:
            break
    return new_states
