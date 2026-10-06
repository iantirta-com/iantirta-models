
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

from torch import nn

from iantirta.models.tools import is_torchdynamo_compiling

_CAN_RECORD_REGISTRY = {}


@dataclass
class OutputRecorder:
    """
    Configuration for recording outputs from a model via hooks.

    Attributes:
        target_class (Type): The class (e.g., nn.Module) to which the hook will be attached.
        index (Optional[int]): If the output is a tuple/list, optionally record only at a specific index.
        layer_name (Optional[str]): Name of the submodule to target (if needed), e.g., "transformer.layer.3.attn".
        class_name (Optional[str]): Name of the class to which the hook will be attached. Could be the suffix of class name in some cases.
        capture_initial_hidden_state  (bool): Whether to prepend the first module's input as the initial hidden state.
    """

    target_class: type[nn.Module]
    index: int = 0
    layer_name: str | None = None
    class_name: str | None = None
    capture_initial_hidden_state: bool = True


class CompileableContextVar:
    """
    Convenience wrapper around a ContextVar for usage with `torch.compile`.
    This behaves exactly as a `ContextVar`, except when compilation is triggered in which case it behaves as a simple
    global variable. This is useful as `torch.compile` cannot trace the `get` method of `ContextVar`. This however means
    that the access to the underlying variable is not thread-safe when compilation is triggered.
    """

    def __init__(self, name):
        self.context_var = ContextVar(name, default=None)
        self.global_var = None
        self.compiling = False

    def get(self):
        # Set was called before and compilation was already detected
        if self.compiling:
            return self.global_var
        else:
            return self.context_var.get()

    def set(self, value):
        if is_torchdynamo_compiling():
            self.global_var = value
            self.compiling = True
            return None
        else:
            return self.context_var.set(value)

    def reset(self, token):
        if self.compiling or token is None:
            self.global_var = None
            self.compiling = False
        else:
            self.context_var.reset(token)


# Thread/context-safe global variable
_active_collector = CompileableContextVar("output_collector")


def capture_outputs(func=None, *, tie_last_hidden_states=True):
    """
    Decorator to intercept specific layer outputs through hooks. The hooks are installed only once and lazily,
    the first time output capture is requested with the `output_xxx` kwargs/config.
    The implementation is fully context/thread safe, except when using `torch.compile`, as dynamo is unable to trace
    through `ContextVar` methods.

    Args:
        tie_last_hidden_states (`bool`, *optional*, defaults to `True`):
            Whether to overwrite `out.hidden_states[-1]` with the `out.last_hidden_state`. This is true for all language models
            and should be toggled off only if `out.hidden_states[-1]` has to be the hidden state before last layer norm, which
            is needed for some vision models (e.g. CLIP, SigLIP). A model config can override this default per-model by setting
            `config.tie_last_hidden_states`.
    """

    def wrapped_fn(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Pop it so that internal modules always return a dict even if False is requested
            return_dict = kwargs.pop("return_dict", getattr(self.config, "return_dict", True))

            # _can_record_outputs is None by default
            capturable_flags = _CAN_RECORD_REGISTRY.get(str(self.__class__)) or {}
            recordable_keys = {
                f"output_{k}": kwargs.get(f"output_{k}", getattr(self.config, f"output_{k}", False))
                for k in capturable_flags
            }
            # For BC as cross-attentions used to be captured with `output_attentions`
            if "cross_attentions" in capturable_flags:
                recordable_keys["output_cross_attentions"] = kwargs.get(
                    "output_attentions", getattr(self.config, "output_attentions", False)
                )
            # The sam model variants need this annoying exception as well...
            if "mask_decoder_attentions" in capturable_flags:
                recordable_keys["output_mask_decoder_attentions"] = kwargs.get(
                    "output_attentions", getattr(self.config, "output_attentions", False)
                )

            collected_outputs = {k.replace("output_", ""): [] for k, v in recordable_keys.items() if v}
            # We accept a list of layer indices as `output_hidden_states`, to capture only specific layer outputs - in this case
            # we need to add the layers to the `collected_outputs`'s dict to tell the hook which ones we need
            if "output_hidden_states" in recordable_keys and isinstance(
                recordable_keys["output_hidden_states"], (list, tuple, set)
            ):
                collected_outputs["_hidden_states_layers"] = set(recordable_keys["output_hidden_states"])
            # Make sure hooks are installed if we need to collect outputs
            if len(collected_outputs) > 0:
                maybe_install_capturing_hooks(self)
            # Let's activate the output collector hooks if needed!
            output_token = _active_collector.set(collected_outputs)

            # Run the forward
            try:
                outputs = func(self, *args, **kwargs)
            # Reset the states
            finally:
                _active_collector.reset(output_token)

            hidden_states_layers = collected_outputs.pop("_hidden_states_layers", None)
            # Inject collected outputs into model output (return everything as tuples for BC)
            for key in collected_outputs:  # noqa: PLC0206
                if key == "hidden_states":
                    tie_last = getattr(self.config, "tie_last_hidden_states", None)
                    tie_last = tie_last_hidden_states if tie_last is None else tie_last
                    if not tie_last or (hidden_states_layers is not None and collected_outputs[key][-1] is None):
                        pass
                    elif hasattr(outputs, "vision_hidden_states"):
                        collected_outputs[key] = collected_outputs[key][:-1]
                        collected_outputs[key].append(outputs.vision_hidden_states)
                    elif hasattr(outputs, "last_hidden_state"):
                        collected_outputs[key] = collected_outputs[key][:-1]
                        collected_outputs[key].append(outputs.last_hidden_state)

                outputs[key] = tuple(collected_outputs[key])

            if return_dict is False:
                outputs = outputs.to_tuple()

            return outputs

        return wrapper

    if func is not None:
        return wrapped_fn(func)
    return wrapped_fn


def can_return_tuple(func):
    """
    Decorator to wrap model method, to call output.to_tuple() if return_dict=False passed as a kwarg or
    return_dict=False is set in the config.

    Note:
        output.to_tuple() convert output to tuple skipping all `None` values.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return_dict = self.config.return_dict if hasattr(self, "config") else True
        return_dict_passed = kwargs.pop("return_dict", return_dict)
        if return_dict_passed is not None:
            return_dict = return_dict_passed
        output = func(self, *args, **kwargs)
        if not return_dict and not isinstance(output, tuple):
            output = output.to_tuple()
        return output

    return wrapper
