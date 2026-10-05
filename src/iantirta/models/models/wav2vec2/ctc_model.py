
from torch import nn


class Wav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
    def __init__(self, config, target_lang: str | None = None):
        r"""
        target_lang (`str`, *optional*):
            Language id of adapter weights.
            Adapter weights are stored in the
            format adapter.<lang>.safetensors or
            adapter.<lang>.bin. Only relevant when
            using an instance of [`Wav2Vec2ForCTC`]
            with adapters. Uses 'eng' by default.
        """
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)

        self.target_lang = target_lang

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} "
                "with a configuration that does not define the "
                "vocabulary size of the language model head. "
                "Please instantiate the model as follows: "
                "`Wav2Vec2ForCTC.from_pretrained(..., "
                "vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(
                config, "add_adapter"
            ) and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def tie_weights(self, **kwargs):
        """
        This method overwrites [`~PreTrainedModel.tie_weights`]
        so that adapter weights can be correctly loaded when
        passing `target_lang=...` to `from_pretrained(...)`.

        This method is **not** supposed to be called by
        the user and is prone to be changed in the future.
        """

        if get_torch_context_manager_or_global_device() == torch.device("meta"):
            return

        # Note that `tie_weights` is usually used
        # to tie input and output embedding weights.
        # The method is re-purposed to
        # correctly load adapter layers for Wav2Vec2
        # so that we do not have to introduce a new API to
        # [`PreTrainedModel`]. While slightly hacky,
        # Wav2Vec2 never has to tie input and output embeddings,
        # so that it is ok to repurpose this function here.
        target_lang = self.target_lang

        if target_lang is not None and getattr(
            self.config, "adapter_attn_dim", None
        ) is None:
            raise ValueError(
                f"Cannot pass `target_lang`: {target_lang} "
                "if `config.adapter_attn_dim` is not defined."
            )
        elif target_lang is None and getattr(
            self.config, "adapter_attn_dim", None
        ) is not None:
            logger.info(
                "By default `target_lang` is set to 'eng'."
            )
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)
