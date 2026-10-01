import sys

from .vendor import huggingface_hub
from .vendor import transformers

sys.modules.setdefault("huggingface_hub", huggingface_hub)
sys.modules.setdefault("transformers", transformers)
