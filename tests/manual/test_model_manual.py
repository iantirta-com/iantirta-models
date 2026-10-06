
import pytest

from iantirta.models.core.auto import (
    AutoConfig,
    AutoModel,
    AutoModelForCTC,
    AutoModelForMultimodalLM,
)

pytestmark = pytest.mark.manual


def test_mms_model():
    model = AutoModelForCTC.from_pretrained("facebook/mms-1b-all")
    assert model is not None
    from rich import inspect
    inspect(model)
