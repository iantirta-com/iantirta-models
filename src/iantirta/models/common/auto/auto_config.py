

_UNSUPPORTED_KWARGS = [
    "revision",
    "token",
    
]

class AutoConfig:
    """ Auto Config Class. """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        **kwargs
    ):
        pass