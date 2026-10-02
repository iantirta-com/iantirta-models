

class Wav2Vec2ForCTC:

    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        pass