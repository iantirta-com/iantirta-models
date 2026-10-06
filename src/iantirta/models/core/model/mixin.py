
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from .pretrained import PreTrainedModel


class ModuleUtilsMixin:
    """
    A few utilities for `torch.nn.Modules`, to be used as a mixin.
    """

    @property
    def device(self: "PreTrainedModel") -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return next(param.device for param in self.parameters())

    @property
    def dtype(self: "PreTrainedModel") -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return next(param.dtype for param in self.parameters() if param.is_floating_point())

    def num_parameters(self: "PreTrainedModel", only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]

        is_loaded_in_4bit = getattr(self, "is_loaded_in_4bit", False)
        if is_loaded_in_4bit:
            import bitsandbytes as bnb  # type: ignore

        total_params = 0
        for name, param in self.named_parameters():
            if exclude_embeddings and name in embedding_param_names:
                continue
            if param.requires_grad or not only_trainable:
                # For 4bit models, we need to multiply the number of parameters by 2 as half of the parameters are
                # used for the 4bit quantization (uint8 tensors are stored)
                if is_loaded_in_4bit and isinstance(param, bnb.nn.Params4bit):
                    if hasattr(param, "element_size"):
                        num_bytes = param.element_size()
                    elif hasattr(param, "quant_storage"):
                        num_bytes = param.quant_storage.itemsize
                    else:
                        num_bytes = 1
                    total_params += param.numel() * 2 * num_bytes
                else:
                    total_params += param.numel()

        return total_params


class EmbeddingAccessMixin:
    """
    Base utilities to regroup getters and setters for embeddings.
    Introduces the `input_layer_embed` attribute, which indicates
    where the input embeddings come from and where they
    should be set.
    """

    _input_embed_layer = "embed_tokens"  # default layer that holds input embeddings.

    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the model's input embeddings.

        Returns:
            `nn.Module`: A torch module mapping vocabulary to hidden states.
        """

        name = getattr(self, "_input_embed_layer", "embed_tokens")

        # 1) Direct attribute (most NLP models).
        if (default_embedding := getattr(self, name, None)) is not None:
            return default_embedding
        # 2) Nested embeddings (e.g., self.embeddings.patch_embedding for vision/audio models).
        embeddings = getattr(self, "embeddings", None)
        if embeddings is not None and hasattr(embeddings, name):
            return getattr(embeddings, name)
        # 3) Encoder/decoder wrappers (e.g., `self.model.embed_tokens` or similar overrides).
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, name):
            return getattr(model, name)
        # 4) Multimodal LMs that wrap a text sub-model at `self.language_model`.
        language_model = getattr(self, "language_model", None)
        if (
            language_model is not None
            and language_model is not self
            and hasattr(language_model, "get_input_embeddings")
        ):
            return language_model.get_input_embeddings()

        # 5) recurse once into the registered *base* model (e.g. for encoder/decoder)
        base_model = getattr(self, "base_model", None)
        if base_model is not None and base_model is not self and hasattr(base_model, "get_input_embeddings"):
            return base_model.get_input_embeddings()

        raise NotImplementedError(
            f"`get_input_embeddings` not auto‑handled for {self.__class__.__name__}; please override in the subclass."
        )

    def set_input_embeddings(self, value: nn.Module):
        """Fallback setter that handles **~70%** of models in the code-base.

        Order of attempts:
        1. `self.<_input_embed_layer>` (direct attribute)
        2. `self.embeddings.<_input_embed_layer>` (nested embeddings for vision/audio models)
        3. `self.model.<_input_embed_layer>` (encoder/decoder models)
        4. delegate to the *base model* if one exists
        5. otherwise raise `NotImplementedError` so subclasses still can (and
            should) override for exotic layouts.
        """

        name = getattr(self, "_input_embed_layer", "embed_tokens")
        # 1) Direct attribute (most NLP models)
        if hasattr(self, name):
            setattr(self, name, value)
        # 2) Nested embeddings (e.g., self.embeddings.patch_embedding for vision models)
        elif (embeddings := getattr(self, "embeddings", None)) is not None and hasattr(embeddings, name):
            setattr(embeddings, name, value)
        # 3) encoder/decoder and VLMs like `Gemma3nForConditionalGeneration`
        elif (model := getattr(self, "model", None)) is not None and hasattr(model, name):
            setattr(model, name, value)
        # 4) Multimodal LMs that wrap a text sub-model at `self.language_model`.
        elif (
            (language_model := getattr(self, "language_model", None)) is not None
            and language_model is not self
            and hasattr(language_model, "set_input_embeddings")
        ):
            language_model.set_input_embeddings(value)
        # 5) recurse once into the registered *base* model (e.g. for encoder/decoder)
        elif (
            (base_model := getattr(self, "base_model", None)) is not None
            and base_model is not self
            and hasattr(base_model, "set_input_embeddings")
        ):
            base_model.set_input_embeddings(value)
        else:
            raise NotImplementedError(
                f"`set_input_embeddings` not auto-handled for {self.__class__.__name__}; please override in the subclass."
            )

    def get_output_embeddings(self):
        if not hasattr(self, "lm_head"):
            return None
        try:
            # Speech / vision backbones raise here, so we return None.
            # Legit use of get_input_embs?
            self.get_input_embeddings()
        except NotImplementedError:
            return None
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the model's output embedding, defaulting to setting new_embeddings to lm_head.
        """
        if self.lm_head:
            self.lm_head = new_embeddings
