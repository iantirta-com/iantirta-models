
import logging
from abc import abstractmethod

__all__ = [
    "BaseModelMixin"
]


class BaseModelMixin:
    """
    Wrapper to hold model instances.
    """
    model = None

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None
    ):
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def load_model(self):
        """
        Called to load model flexibly.
        """
        if self.model is not None:
            return
