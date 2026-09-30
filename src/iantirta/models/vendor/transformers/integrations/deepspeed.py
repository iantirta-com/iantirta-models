# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Integration with Deepspeed
"""

import copy
import importlib.metadata
import importlib.util
import weakref

from ..utils import is_accelerate_available, is_torch_available, logging

if is_torch_available():
    import torch
    from torch import nn


logger = logging.get_logger(__name__)


def is_deepspeed_available():
    package_exists = importlib.util.find_spec("deepspeed") is not None

    # Check we're not importing a "deepspeed" directory somewhere but the actual library by trying to grab the version
    # AND checking it has an author field in the metadata that is HuggingFace.
    if package_exists:
        try:
            _ = importlib.metadata.metadata("deepspeed")
            return True
        except importlib.metadata.PackageNotFoundError:
            return False


# keep the config object global to be able to access it anywhere during TrainingArguments life-cycle
_hf_deepspeed_config_weak_ref = None


def set_hf_deepspeed_config(hf_deepspeed_config_obj):
    # this is a special weakref global object to allow us to get to Deepspeed config from APIs
    # that don't have an easy way to get to the Deepspeed config outside of the Trainer domain.
    global _hf_deepspeed_config_weak_ref
    # will go away automatically when HfDeepSpeedConfig is destroyed (when TrainingArguments is destroyed)
    _hf_deepspeed_config_weak_ref = weakref.ref(hf_deepspeed_config_obj)


def unset_hf_deepspeed_config():
    # useful for unit tests to ensure the global state doesn't leak - call from `tearDown` method
    global _hf_deepspeed_config_weak_ref
    _hf_deepspeed_config_weak_ref = None


def is_deepspeed_zero3_enabled():
    if _hf_deepspeed_config_weak_ref is not None and _hf_deepspeed_config_weak_ref() is not None:
        return _hf_deepspeed_config_weak_ref().is_zero3()
    else:
        return False


def deepspeed_config():
    if _hf_deepspeed_config_weak_ref is not None and _hf_deepspeed_config_weak_ref() is not None:
        return _hf_deepspeed_config_weak_ref().config
    else:
        return None


def initialize_weights_zero3(model):
    """
    DeepSpeed ZeRO-3 variant of `PreTrainedModel.initialize_weights`. Mirrors the `smart_apply`
    dispatch logic but gathers each module's partitioned parameters before calling
    `_initialize_weights`, so initialization operates on full tensors instead of empty shards.
    Only rank 0 performs the actual init.
    """
    import deepspeed
    import torch

    from ..initialization import guard_torch_init_functions
    from ..modeling_utils import PreTrainedModel

    is_remote_code = model.is_remote_code()

    def _apply_zero3(model_or_module, fn):
        for child in model_or_module.children():
            if isinstance(child, PreTrainedModel):
                _apply_zero3(child, child._initialize_weights)
            else:
                _apply_zero3(child, fn)

        params = list(model_or_module.parameters(recurse=False))
        if params:
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                if deepspeed.comm.get_rank() == 0:
                    fn(model_or_module, is_remote_code)
        else:
            fn(model_or_module, is_remote_code)

    with torch.no_grad():
        with guard_torch_init_functions():
            _apply_zero3(model, model._initialize_weights)


def _apply_weight_conversions_to_state_dict(model, state_dict, weight_mapping):
    """
    Apply weight conversions (renaming and merging/splitting operations) to a state dict.
    This is a simplified version that handles the conversion without loading into the model.
    """
    # Check for Tensor Parallelism - weight conversions are not tested with TP
    # TP uses ReplaceWithTensorSlicing which may conflict with our weight conversions
    ds_config = deepspeed_config()
    if ds_config is not None:
        # Check training config (tensor_parallel.autotp_size)
        tp_size = ds_config.get("tensor_parallel", {}).get("autotp_size", 1)
        # Check inference config (inference.tensor_parallel.tp_size)
        inference_config = ds_config.get("inference", {})
        if isinstance(inference_config, dict):
            tp_size = max(tp_size, inference_config.get("tensor_parallel", {}).get("tp_size", 1))
        if tp_size > 1:
            raise NotImplementedError(
                "Weight conversions (e.g., MoE expert fusion) with DeepSpeed Tensor Parallelism "
                "are not yet implemented but support is coming soon. Please disable tensor_parallel "
                "in your DeepSpeed config or convert your checkpoint to the expected format first."
            )

    from ..core_model_loading import WeightConverter, WeightRenaming, dot_natural_key, rename_source_key

    # Preserve metadata from the original state dict
    metadata = getattr(state_dict, "_metadata", None)

    base_model_prefix = model.base_model_prefix

    # Build a meta state dict for matching - only keys/shapes, no actual tensor data
    # This minimizes memory since we don't duplicate the model's parameters
    model_state_dict = {}
    for key, param in model.state_dict().items():
        model_state_dict[key] = torch.empty(param.shape, dtype=param.dtype, device="meta")

    renamings = [entry for entry in weight_mapping if isinstance(entry, WeightRenaming)]
    converters = [entry for entry in weight_mapping if isinstance(entry, WeightConverter)]

    # Fast path: if we only have simple renamings and no converters, we can skip the expensive collection logic
    if len(converters) == 0:
        new_state_dict = {}
        for original_key, tensor in state_dict.items():
            renamed_key, _ = rename_source_key(
                original_key, renamings, [], base_model_prefix=base_model_prefix, meta_state_dict=model_state_dict
            )
            if renamed_key in model_state_dict:
                new_state_dict[renamed_key] = tensor
        # Attach metadata to the new state dict
        if metadata is not None:
            new_state_dict._metadata = metadata
        return new_state_dict

    # Full path: we have WeightConverter operations that require tensor fusion/splitting
    pattern_to_converter = {k: converter for converter in converters for k in converter.source_patterns}

    # Build a mapping of what needs to be converted
    # Sort keys to ensure consistent ordering (important for MoE conversions)
    # Iterate over sorted keys and pop from state_dict to free memory immediately
    conversion_mapping = {}
    new_state_dict = {}
    sorted_keys = sorted(state_dict.keys(), key=lambda k: dot_natural_key(k))
    for original_key in sorted_keys:
        tensor = state_dict.pop(original_key)
        renamed_key, source_pattern = rename_source_key(
            original_key, renamings, converters, base_model_prefix=base_model_prefix, meta_state_dict=model_state_dict
        )

        # Only process if the renamed key is in the model's state dict
        if renamed_key in model_state_dict:
            # If source_pattern is not None, this key needs WeightConverter (e.g., MoE fusion)
            if source_pattern is not None:
                # Create a fresh converter for this layer to hold its tensors
                # Share operations list (lightweight, no large data) but get new collected_tensors
                converter = pattern_to_converter[source_pattern]
                new_converter = WeightConverter(
                    source_patterns=converter.source_patterns,
                    target_patterns=converter.target_patterns,
                    operations=converter.operations,
                )
                mapping = conversion_mapping.setdefault(renamed_key, new_converter)
                mapping.add_tensor(renamed_key, original_key, source_pattern, tensor)
            else:
                # No conversion needed - add tensor directly to new_state_dict
                # (this handles keys like embed_tokens, lm_head, layernorm, attention)
                new_state_dict[renamed_key] = tensor

    # Apply the conversions and build the new state dict
    for renamed_key, mapping in conversion_mapping.items():
        try:
            realized_value = mapping.convert(
                renamed_key,
                model=model,
                config=model.config,
            )
            for target_name, param in realized_value.items():
                param = param[0] if isinstance(param, list) else param
                new_state_dict[target_name] = param
        except Exception as e:
            raise RuntimeError(
                f"Failed to apply weight conversion for '{renamed_key}'. "
                f"This likely means the checkpoint format is incompatible with the current model version. "
                f"Error: {e}"
            ) from e

    # Attach metadata to the new state dict
    if metadata is not None:
        new_state_dict._metadata = metadata

    return new_state_dict


def _load_state_dict_into_zero3_model(model_to_load, state_dict, load_config=None):
    """
    Loads state dict into a model specifically for Zero3, since DeepSpeed does not support the `transformers`
    tensor parallelism API.

    Nearly identical code to PyTorch's `_load_from_state_dict`

    Args:
        model_to_load: The model to load weights into
        state_dict: The state dict containing the weights
        load_config: Optional LoadStateDictConfig containing weight_mapping and other loading options
    """
    # copy state_dict so `_load_state_dict_into_zero3_model` can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # Extract weight_mapping from load_config if provided
    weight_mapping = None
    if load_config is not None:
        weight_mapping = getattr(load_config, "weight_mapping", None)

    # Apply weight conversions if provided
    if weight_mapping is not None and len(weight_mapping) > 0:
        state_dict = _apply_weight_conversions_to_state_dict(model_to_load, state_dict, weight_mapping)
        # Keep the current weight conversion mapping for later saving (in case it was coming directly from the user)
        model_to_load._weight_conversions = weight_mapping

    error_msgs = []
    meta_model_state_dict = model_to_load.state_dict()
    missing_keys = set(meta_model_state_dict.keys())

    prefix_model = getattr(model_to_load, "base_model_prefix", None)
    # take care of the case where in the checkpoint we don't have the prefix
    state_dict = {
        (f"{prefix_model}.{k}" if meta_model_state_dict.get(f"{prefix_model}.{k}") is not None else k): v
        for k, v in state_dict.items()
    }

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, state_dict, prefix="", assign_to_params_buffers=False):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        local_metadata["assign_to_params_buffers"] = assign_to_params_buffers

        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if is_deepspeed_zero3_enabled():
            import deepspeed

            # In sharded models, each shard has only part of the full state_dict, so only gather
            # parameters that are in the current state_dict.
            named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
            params_to_gather = []
            for k in named_parameters:
                if k in state_dict:
                    param = named_parameters[k]
                    # crucial to not init the weight again
                    param._is_hf_initialized = True
                    params_to_gather.append(param)
                    missing_keys.discard(k)

            if len(params_to_gather) > 0:
                # because zero3 puts placeholders in model params, this context
                # manager gathers (unpartitions) the params of the current layer, then loads from
                # the state dict and then re-partitions them again
                with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        module._load_from_state_dict(*args)

            # Buffers are not partitioned by ZeRO-3, load them directly
            named_buffers = dict(module.named_buffers(prefix=prefix[:-1], recurse=False))
            for k, buf in named_buffers.items():
                if k in state_dict and buf is not None:
                    missing_keys.discard(k)
                    with torch.no_grad():
                        buf.copy_(state_dict[k])
                    buf._is_hf_initialized = True

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".", assign_to_params_buffers)

    load(model_to_load, state_dict, assign_to_params_buffers=False)

    return error_msgs, missing_keys


def deepspeed_optim_sched(trainer, hf_deepspeed_config, args, num_training_steps, model_parameters):
    """
    A convenience wrapper that deals with optimizer and lr scheduler configuration.
    """
    from accelerate.utils import DummyOptim, DummyScheduler

    config = hf_deepspeed_config.config

    # Mixing and matching DS schedulers and optimizers is supported unless Offload is enabled in which case it's:
    # 1. DS scheduler + DS optimizer: Yes
    # 2. HF scheduler + HF optimizer: Mostly*
    # 3. DS scheduler + HF optimizer: Mostly*
    # 4. HF scheduler + DS optimizer: Yes
    #
    # Mostly*: All non-native DeepSpeed optimizers that have both CPU and GPU implementation should work (except LAMB)

    optimizer = None
    if "optimizer" in config:
        optimizer = DummyOptim(params=model_parameters)
    else:
        if hf_deepspeed_config.is_offload():
            logger.info(
                "Detected ZeRO Offload and non-DeepSpeed optimizers: This combination should work as long as the"
                " custom optimizer has both CPU and GPU implementation (except LAMB)"
            )

        # ds supports Adam, OneBitAdam, and Lamb optimizers and can import other optimizers from torch.
        # But trainer uses AdamW by default.
        optimizer = trainer.create_optimizer()
        # To use other optimizers requires voiding warranty with: `zero_allow_untested_optimizer`
        config["zero_allow_untested_optimizer"] = True

    lr_scheduler = None
    if "scheduler" in config:
        lr_scheduler = DummyScheduler(optimizer)
    else:
        if isinstance(optimizer, DummyOptim):

            def _lr_scheduler_callable(optimizer):
                # create a shallow copy first, so later modifications do not affect original trainer
                trainer_copy = copy.copy(trainer)
                # at the time _lr_scheduler_callable is called, trainer.lr_scheduler has been set
                # update it to None so that we can re-create a new scheduler
                trainer_copy.lr_scheduler = None
                lr_scheduler = trainer_copy.create_scheduler(
                    num_training_steps=num_training_steps, optimizer=optimizer
                )
                return lr_scheduler

            lr_scheduler = DummyScheduler(optimizer, lr_scheduler_callable=_lr_scheduler_callable)

    return optimizer, lr_scheduler


def deepspeed_init(trainer, num_training_steps, inference=False):
    """
    Init DeepSpeed, after updating the DeepSpeed configuration with any relevant Trainer's args.

    If `resume_from_checkpoint` was passed then an attempt to resume from a previously saved checkpoint will be made.

    Args:
        trainer: Trainer object
        num_training_steps: per single gpu
        resume_from_checkpoint: path to a checkpoint if to resume from after normal DeepSpeedEngine load
        inference: launch in inference mode (no optimizer and no lr scheduler)
        auto_find_batch_size: whether to ignore the `train_micro_batch_size_per_gpu` argument as it's being
            set automatically by the auto batch size finder

    Returns: optimizer, lr_scheduler

    We may use `deepspeed_init` more than once during the life of Trainer, when we do - it's a temp hack based on:
    https://github.com/deepspeedai/DeepSpeed/issues/1394#issuecomment-937405374 until Deepspeed fixes a bug where it
    can't resume from a checkpoint after it did some stepping https://github.com/deepspeedai/DeepSpeed/issues/1612

    """
    from deepspeed.utils import logger as ds_logger

    model = trainer.model
    args = trainer.args

    hf_deepspeed_config = trainer.accelerator.state.deepspeed_plugin.hf_ds_config

    # resume config update - some bits like `model` and `num_training_steps` only become available during train
    hf_deepspeed_config.trainer_config_finalize(args, model, num_training_steps)

    # set the Deepspeed log level consistent with the Trainer
    ds_logger.setLevel(args.get_process_log_level())

    if inference:
        # only Z3 makes sense for the inference
        if not hf_deepspeed_config.is_zero3():
            raise ValueError("ZeRO inference only makes sense with ZeRO Stage 3 - please adjust your config")

        # in case the training config is re-used for inference
        hf_deepspeed_config.del_config_sub_tree("optimizer")
        hf_deepspeed_config.del_config_sub_tree("lr_scheduler")
        optimizer, lr_scheduler = None, None
        model_parameters = None
    else:
        trainer.optimizer = None  # important for when deepspeed_init is used as re-init
        deepspeed_tp_size = hf_deepspeed_config.config.get("tensor_parallel", {}).get("autotp_size", 1)
        if deepspeed_tp_size > 1:
            import deepspeed

            model = deepspeed.tp_model_init(
                model=model,
                tp_size=deepspeed_tp_size,
                dtype=hf_deepspeed_config.dtype(),
                config=hf_deepspeed_config.config,
            )
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer, lr_scheduler = deepspeed_optim_sched(
            trainer, hf_deepspeed_config, args, num_training_steps, model_parameters
        )

    # keep for quick debug:
    # from pprint import pprint; pprint(config)

    return optimizer, lr_scheduler


def deepspeed_load_checkpoint(deepspeed_engine, checkpoint_path, load_module_strict=True):
    # it's possible that the user is trying to resume from model_path, which doesn't necessarily
    # contain a deepspeed checkpoint. e.g. examples just check if the dir exists and assume it's
    # a resume from a checkpoint and not just a local pretrained weight. So we check here if the
    # path contains what looks like a deepspeed checkpoint
    import glob

    deepspeed_checkpoint_dirs = sorted(glob.glob(f"{checkpoint_path}/global_step*"))

    if len(deepspeed_checkpoint_dirs) > 0:
        logger.info(f"Attempting to resume from {checkpoint_path}")
        # this magically updates self.optimizer and self.lr_scheduler
        load_path, _ = deepspeed_engine.load_checkpoint(
            checkpoint_path,
            load_module_strict=load_module_strict,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        if load_path is None:
            raise ValueError(f"[deepspeed] failed to resume from checkpoint {checkpoint_path}")
    else:
        raise ValueError(f"Can't find a valid checkpoint at {checkpoint_path}")


def deepspeed_sp_compute_loss(accelerator, model, inputs, return_outputs, pc):
    """
    Computes the loss under sequence parallelism with `sp_backend="deepspeed"` and `sp_size > 1`.

    Performs weighted loss aggregation across SP ranks, accounting for varying numbers of valid tokens per rank
    (e.g., when some ranks receive only padding or prompt tokens that are masked with -100).

    Args:
        accelerator (`Accelerator`): The accelerator instance with `torch_device_mesh` support.
        model (`torch.nn.Module`): The model to compute the loss for.
        inputs (`dict[str, torch.Tensor | Any]`): The input data for the model. Must include `"shift_labels"` key.
        return_outputs (`bool`): Whether to return the model outputs along with the loss.
        pc (`accelerate.parallelism_config.ParallelismConfig`): The parallelism configuration.

    Returns:
        The loss, or a tuple of `(loss, outputs)` if `return_outputs` is `True`.
    """
    # DeepSpeed SP automatically injects shift_labels into inputs (pre-shifted labels for SP).
    # The model's forward pass receives shift_labels via **kwargs and passes it to the loss function.
    # Both standard transformer models and Liger-patched models handle shift_labels correctly,
    # so we can directly use the computed loss from the model output.
    # See: https://huggingface.co/docs/accelerate/en/concept_guides/sequence_parallelism
    if "labels" not in inputs and "shift_labels" in inputs:
        # DeepSpeed SP Dataloader removes "labels" but we need it, otherwise, we won't compute the loss.
        inputs["labels"] = inputs["shift_labels"]
    outputs = model(**inputs)
    loss = outputs.loss

    # Prefer DeepSpeed SP groups when using Ulysses; otherwise fall back to torch device mesh.
    if pc.sp_backend == "deepspeed" and pc.sp_size > 1:
        from deepspeed.utils import groups

        sp_group = groups._get_sequence_parallel_group()
    elif accelerator.torch_device_mesh is not None:
        sp_group = accelerator.torch_device_mesh["sp"].get_group()
    else:
        raise ValueError(
            "Sequence parallelism is enabled but no SP process group is available. "
            "Ensure torch_device_mesh is initialized or sp_backend='deepspeed' with sp_size > 1."
        )
    # differentiable weighted per-shard-loss aggregation across ranks
    losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=sp_group)
    # special dealing with SFT that has prompt tokens that aren't used in loss computation
    good_tokens = (inputs["shift_labels"] != -100).view(-1).sum()
    good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=sp_group)
    losses_stacked = torch.stack(losses_per_rank)
    good_tokens_stacked = torch.stack(good_tokens_per_rank)
    mask = good_tokens_stacked > 0
    safe_losses = torch.where(mask, losses_stacked, torch.zeros_like(losses_stacked))
    total_loss = (safe_losses * good_tokens_stacked).sum()
    total_good_tokens = good_tokens_stacked.sum()
    loss = total_loss / total_good_tokens.clamp(min=1)

    return (loss, outputs) if return_outputs else loss
