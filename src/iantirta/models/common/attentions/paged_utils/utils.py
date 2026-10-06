from iantirta.models.common.configuration_utils import PreTrainedConfig


def find_num_key_value_heads(config: PreTrainedConfig) -> int:
    """Finds the number of key-value heads for the given config."""
    # If the model supports GQA, we leverage it by using the num_key_value_heads attribute
    kv_heads = getattr(config, "num_key_value_heads", None)
    if kv_heads is not None:
        return kv_heads
    # Otherwise, the number of KV heads is the same as the number of attention heads
    kv_heads = getattr(config, "num_attention_heads", None)
    if kv_heads is not None:
        return kv_heads
    raise ValueError(f"num_key_value_heads or num_attention_heads could not be found in the config:\n{config}")


def find_head_dim(config: PreTrainedConfig) -> int:
    """Finds the head dimension for the given config."""
    # If the model has the head_dim attribute, there is nothing to do but return it
    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None:
        return head_dim
    # If it is missing, we may reconstruct it from the hidden size and the number of attention heads
    hidden_size = getattr(config, "hidden_size", None)
    num_attention_heads = getattr(config, "num_attention_heads", None)
    if hidden_size is not None and num_attention_heads is not None:
        return hidden_size // num_attention_heads
    raise ValueError(f"head_dim or (hidden_size and num_attention_heads) could not be found in the config:\n{config}")


def exact_div(a: int, b: int) -> int:
    """Divide an integer a by a integer b and error out if there is a remainder."""
    quotient, remainder = divmod(a, b)
    if remainder:
        raise ValueError(f"Division of {a} by {b} is not exact: {remainder = } != 0")
    return quotient
