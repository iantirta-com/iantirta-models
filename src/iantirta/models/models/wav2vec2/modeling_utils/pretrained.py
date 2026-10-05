
from torch import nn
import torch
from iantirta.models.common import PreTrainedModel

import logging

logger = logging.getLogger("iantirta.models.wav2vec2")

WAV2VEC2_ADAPTER_PT_FILE = "adapter.{}.bin"
WAV2VEC2_ADAPTER_SAFE_FILE = "adapter.{}.safetensors"


class Wav2Vec2PreTrainedModel(PreTrainedModel):

    def _get_adapters(self):
        if self.config.adapter_attn_dim is None:
            raise ValueError(f"{self.__class__} has no adapter layers. Make sure to define `config.adapter_attn_dim`.")

        adapter_weights = {}
        for name, module in self.named_modules():
            if isinstance(module, Wav2Vec2AttnAdapterLayer):
                for param_name, param in module.named_parameters():
                    adapter_weights[".".join([name, param_name])] = param

        if isinstance(self, Wav2Vec2ForCTC):
            for name, param in self.lm_head.named_parameters():
                adapter_weights[".".join(["lm_head", name])] = param

        return adapter_weights

    def load_adapter(
        self,
        target_lang: str,
        force_load=True, **kwargs
    ):
        if self.config.adapter_attn_dim is None:
            raise ValueError(
                f"Cannot load_adapter for {target_lang} "
                "if `config.adapter_attn_dim` is not defined."
            )

        if target_lang == self.target_lang and not force_load:
            logger.warning(
                f"Adapter weights are already set to {target_lang}."
            )
            return

        cache_dir = kwargs.pop("cache_dir", None)
        _ = kwargs.pop("force_download", False)
        _ = kwargs.pop("proxies", None)
        _ = kwargs.pop("local_files_only", False)
        _ = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        model_path_or_id = self.config._name_or_path
        state_dict = None

        # 1. Let's first try loading a safetensors adapter weight
        if use_safetensors is not False:
            filepath = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang)

            try:
                weight_path = cached_file(
                    model_path_or_id,
                    filename=filepath,
                    revision=revision,
                    cache_dir=cache_dir,
                )

                state_dict = safe_load_file(weight_path)

            except OSError:
                if use_safetensors:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise

            except Exception:
                # For any other exception, we throw a generic error.
                if use_safetensors:
                    raise OSError(
                        f"Can't load the model for '{model_path_or_id}'. If you were trying to load it"
                        " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{model_path_or_id}' is the correct path to a"
                        f" directory containing a file named {filepath}."
                    )

        # 2. If this didn't work let's try loading a PyTorch adapter weight
        if state_dict is None:
            filepath = WAV2VEC2_ADAPTER_PT_FILE.format(target_lang)

            try:
                weight_path = cached_file(
                    model_path_or_id,
                    filename=filepath,
                    revision=revision,
                    cache_dir=cache_dir,
                )

                check_torch_load_is_safe()
                state_dict = torch.load(
                    weight_path,
                    map_location="cpu",
                    weights_only=True,
                )

            except OSError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                # to the original exception.
                raise

            except ValueError:
                raise

            except Exception:
                # For any other exception, we throw a generic error.
                raise OSError(
                    f"Can't load the model for '{model_path_or_id}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{model_path_or_id}' is the correct path to a"
                    f" directory containing a file named {filepath}."
                )

        adapter_weights = self._get_adapters()
        unexpected_keys = set(state_dict.keys()) - set(adapter_weights.keys())
        missing_keys = set(adapter_weights.keys()) - set(state_dict.keys())

        if len(unexpected_keys) > 0:
            raise ValueError(
                f"The adapter weights {weight_path} "
                f"has unexpected keys: {', '.join(unexpected_keys)}."
            )
        elif len(missing_keys) > 0:
            raise ValueError(
                f"The adapter weights {weight_path} "
                f"has missing keys: {', '.join(missing_keys)}."
            )

        # make sure now vocab size is correct
        target_vocab_size = state_dict["lm_head.weight"].shape[0]
        if target_vocab_size != self.config.vocab_size:
            self.lm_head = nn.Linear(
                self.config.output_hidden_size,
                target_vocab_size,
                device=self.device,
                dtype=self.dtype
            )
            self.config.vocab_size = target_vocab_size

        # make sure that adapter weights
        # are put in exactly the same precision
        # and device placement and overwritten
        # adapter weights
        state_dict = {
            k: v.to(adapter_weights[k])
            for k, v in state_dict.items()
        }
        self.load_state_dict(state_dict, strict=False)

        # set target language correctly
        self.target_lang = target_lang
