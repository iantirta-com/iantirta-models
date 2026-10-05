import gc
import logging
import queue
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch

from iantirta.models.common.attentions.flash_attention.utils import (
    is_flash_attn_2_available,
    is_flash_attn_3_available,
)
from iantirta.models.common.attentions.utils import is_flash_attention_requested

logger = logging.getLogger(__name__)

# TODO: add the @strict decorator to prevent attributes passed as args rather than kwargs
@dataclass
class ContinuousBatchingConfig:
    """
    Class that holds arguments relative to continuous batching, when using continuous batching through the
    `generate_batch` method or the `continuous_batching_context_manager` context manager.

    Args:
        page_size (`int`, *optional*, defaults to 256):
            The number of tokens stored for each layer inside a (full-attention) page. A block storing the cache of N
            layers has N pages (one per layer), each holding cache for `page_size` tokens for one layer. Default is 256.
        num_blocks (`int`, *optional*):
            Number of blocks in the KV cache. Auto-inferred from GPU memory when `None`.
        max_batch_tokens (`int`, *optional*):
            Maximum number of tokens in a batch. Auto-inferred from GPU memory when `None`.
        max_memory_percent (`float`, *optional*):
            Maximum percentage of free GPU memory (after the model is loaded) to use for the KV cache. When `None`,
            resolved at runtime to 0.9 if there is no logit processing and 0.8 if there is, to leave headroom for
            vocabulary-sized temporary tensors.
        max_requests_per_batch (`int`, *optional*):
            Maximum number of requests per batch. Auto-inferred from workload hints when `None`, with fallback of 1024.
        max_blocks_per_request (`int`, *optional*):
            Maximum blocks per request, used in the `flash_attn_with_kvcache` fast decode path to dimension
            the block table. Setting this to 0 disables the fast decode path. Default is None (auto-inferred).
        allow_block_sharing (`bool`, *optional*, defaults to `True`):
            Whether to allow block sharing for prefix caching. Block sharing can only be allowed, never forced,
            as some models do not support it. Disable if you have few short prompts but long generation lengths.
        use_async_batching (`bool`, *optional*):
            Whether to enable async double-buffering, which removes CPU overhead from the continuous batching
            loop at the cost of doubled VRAM usage. Auto-detected when `None`.
        use_cuda_graph (`bool` or `tuple[bool, bool]`, *optional*):
            Whether to enable CUDA graphs. This can be a tuple of booleans (one for the varlen path and one for the
            decode fast path), a boolean which will apply to both paths, or None (automatically inferred). After calling
            `decide_use_cuda_graphs`, the attribute will be a tuple of booleans. Default is None (automatically inferred).
        q_padding_interval_size (`int`, *optional*, defaults to 0):
            Query padding granularity in tokens for CUDA graphs. Uses a preset from `continuous_api.py` when
            set to 0.
        kv_padding_interval_size (`int`, *optional*, defaults to 0):
            KV padding granularity in tokens for CUDA graphs. Uses a preset from `continuous_api.py` when
            set to 0.
        varlen_compile_config (`CompileConfig`, *optional*):
            CompileConfig for varlen (prefill) path. Default is None (uses generation_config fallback)
            The varlen path handles batches with varying query and KV lengths, often benefiting from dynamic=True.
        decode_compile_config (`CompileConfig`, *optional*):
            CompileConfig for decode (fast) path. Default is None (uses generation_config fallback)
            The decode path handles batches has no dynamic KV length, so static shapes are a better fit.
        default_compile_level (`int`, *optional*, defaults to 0):
            If this is >0 and no compile config is provided for varlen or decode path, a default compile config will be
            provided. The level can go up to 3, and a higher level means more performance but longer warmup time.
        scheduler_type (`str`, *optional*, defaults to `"fifo"`):
            Scheduler type to use.
        safety_margin (`float`, *optional*):
            Safety margin used to limit the amount of offloading. Defaults to None (use class default).
        return_logprobs (`bool`, *optional*, defaults to `False`):
            Whether to return log probabilities along with the generated tokens.
        seed (`int | None`, *optional*):
            An optional seed for generation. If not specified, the internal seed will be set to a random value.
        cpu_offload_space (`float`, *optional*, defaults to 0.0):
            CPU swap space in GiB for KV cache offloading. A pre-allocated pinned CPU buffer of this size is
            created at initialization. When the GPU cache is full, evicted requests' KV caches are copied here
            instead of being discarded. 0 disables offloading (default).
        cpu_offload_space_safety_threshold (`float`, *optional*, defaults to 0.8):
            If `cpu_offload_space` exceeds this fraction of total system RAM, it is clamped to avoid host OOM.
            Set to 1.0 to disable the safety cap. Ignored when psutil is not available.
        max_queue_size (`int`, *optional*, defaults to 0):
            Maximum request queue size for serving. 0 means unlimited.
        per_request_processors (`bool`, *optional*, defaults to `False`):
            Enable per-request logits processor parameters. Default is False.
        drop_unsupported_processors (`bool`, *optional*, defaults to `True`):
            Remove unsupported logits processors instead of erroring. Default is True.
        disable_nccl_graph_mixing (`bool`, *optional*, defaults to `True`):
            Disable NCCL's safety net for parallel graph-captured comms. Never happens in CB and gives TP a perf boost.
        cpu_group_timeout (`float`, *optional*, defaults to 300.0):
            The time (in seconds) after which a CPU communication will timeout and the process will crash. Leave to None
            for no timeout. Default is 300 seconds.
        use_default_compile_configs (`bool | None`, *optional*):
            Deprecated in 5.11: please use default_compile_level instead.
        max_cached_graphs (`int`, *optional*):
            Deprecated in 5.13: maximum number of graph is no longer an issue.
        block_size (`int | None`, *optional*):
            Deprecated in 5.17: now page_size is used instead.
    """

    # The number of tokens stored inside a (full attention) page. A block storing the cache of N layers has N pages, one
    # per layer. Since different page types can hold different number of tokens, this is for a full attention page.
    # Default is 256. Must be at least 4 (for an efficient cache, it should be well above that)
    page_size: int = 256

    # Number of blocks the cache contains. Usually better to leave it as None and be auto inferred.
    num_blocks: int | None = None

    # The maximum number of tokens in a batch. Once the page size is set, this can be auto inferred using GPU size.
    max_batch_tokens: int | None = None

    # The max percentage of free GPU memory (after the model is loaded) to use for the KV cache. If None, auto resolved
    # to 0.9 (no logit processing) or 0.8 (logit processing) to leave headroom for temporary tensors.
    max_memory_percent: float | None = None

    # The maximum number of requests in a batch. Helps limiting the memory footprint of the logits, which scale with the
    # vocabulary size.
    max_requests_per_batch: int | None = None

    # This is only used in the flash_attn_with_kvcache fast decode path to dimension the block table. If it is set to 0,
    # the fast decode path will not be used. Auto-inferred from GPU memory when `None` (default).
    max_blocks_per_request: int | None = None

    # Block sharing can only be allowed, but never forced: some model just do not support it. If you only have a few
    # short prompts, but long generation lengths, you might want to disable block sharing.
    allow_block_sharing: bool = True

    # Enables asynchronous batching. This removes the CPU overhead from the continuous batching loop, at the cost of
    # doubling the VRAM usage. If None, will be automatically detected.
    use_async_batching: bool | None = None

    # Enables cuda graphs. This can be a tuple of booleans (one for the varlen path and one for the decode fast path), a
    # boolean which will apply to both paths, or None (automatically inferred). After calling `decide_use_cuda_graphs`,
    # the attribute will ALWAYS be a tuple of booleans.
    use_cuda_graph: bool | tuple[bool, bool] | None = None

    # If any of these parameters are set to a non-default, CUDA graphs will be used. Otherwise we automatically infer
    # if they should be turned on. Padding interval sizes are in tokens and further explained in the docstring at the
    # top of the continuous_batching/continuous_api.py file.
    q_padding_interval_size: int = 0
    kv_padding_interval_size: int = 0

    # Compile configs for the two execution paths. If None, uses the compile_config from generation_config as fallback.
    varlen_compile_config: CompileConfig | None = None
    decode_compile_config: CompileConfig | None = None
    # Compile level for the executions path, if no compile config is provided for the path. Default is 0 (no compile).
    # Level 1: `mode=default, dynamic=True`
    # Level 2: `mode=max-autotune-no-cudagraphs, dynamic=True`
    # Level 3: `mode=max-autotune-no-cudagraphs, dynamic=False`
    default_compile_level: int = 0

    # Scheduler type. FIFO by default. For all types available, checks SCHEDULER_MAPPING in scheduler.py
    scheduler_type: str = "fifo"
    # Safety margin: if the number of free blocks falls below (safety_margin * num_blocks), then new prefill requests
    # will not be scheduled to prioritize decoding active requests. Defaults to None (use class default).
    safety_margin: float | None = None

    # Whether to generate log probabilities, which is the log of the softmax of the processed logits. If True, the log
    # probabilities will be returned along with the generated tokens in the generation output.
    return_logprobs: bool = False

    # An optional seed for generation. If not specified, the internal seed will be set to a random value.
    seed: int | None = None

    # CPU swap space in GiB for KV cache offloading. When the GPU cache is full and a request must be evicted, its KV
    # cache is copied to this pre-allocated pinned CPU buffer instead of being discarded. Default to 0.0 GiB. You can
    # also set this to None to dimension the pool using only the safety threshold, but this will error out if psutil is
    # not available.
    # TODO: use async transfer and move this to a non-zero value
    cpu_offload_space: float | None = 0.0
    # Safety cap: if cpu_offload_space exceeds this fraction of total system RAM, it is clamped. Set to 0.0 to disable
    # offloading.
    cpu_offload_space_safety_threshold: float = 0.8

    # The parameters below are mostly useful in the context of serving
    max_queue_size: int = 0

    # Enables per-request logits processor parameters. When enabled, each request can specify its own values (e.g.,
    # temperature) via logits_processor_kwargs. When disabled, all requests use the default values.
    per_request_processors: bool = False
    # When True, processors explicitly marked as unsupported are removed with a warning. When False, all processors
    # are kept but warnings are logged for unsupported/unknown ones.
    drop_unsupported_processors: bool = True

    # Disable NCCL's safety net for parallel graph-captured communications. This means it is no longer safe to replay a
    # CUDA graph with NCCL communication at the same time as 1. another CUDA graph with captured comms 2. an eager comm.
    # This is turned on by default because the above never happens in CB and this gives a nice perf boost.
    disable_nccl_graph_mixing: bool = True

    # The time (in seconds) after which a CPU communication will timeout and the process will crash. Leave to None for
    # no timeout. Default is 300 seconds. This exists because dist has a gloo timeout of 30 minutes, which is way too
    # long for almost all use cases.
    cpu_group_timeout: float | None = 300.0

    # Deprecated arguments
    use_default_compile_configs: bool | None = None
    max_cached_graphs: int | None = None
    block_size: int | None = None

    def __post_init__(self):
        # Convert dicts to CompileConfig objects
        if isinstance(self.varlen_compile_config, dict):
            self.varlen_compile_config = CompileConfig(**self.varlen_compile_config)
        if isinstance(self.decode_compile_config, dict):
            self.decode_compile_config = CompileConfig(**self.decode_compile_config)

        # Only turn off graph mixing support if TP is on
        graph_mixing_supported = os.environ.get("NCCL_GRAPH_MIXING_SUPPORT", "1") == "1"
        distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        if self.disable_nccl_graph_mixing and graph_mixing_supported and distributed:
            logger.warning(
                "Setting NCCL_GRAPH_MIXING_SUPPORT = 0 because disable_nccl_graph_mixing is True and WORLD_SIZE > 1."
            )
            os.environ.setdefault("NCCL_GRAPH_MIXING_SUPPORT", "0")

        # Warn about deprecated arguments
        if self.use_default_compile_configs is not None:  # Deprecated in 5.11
            if self.use_default_compile_configs:
                level_msg = "setting default_compile_level to 3. Consider using a lower level for faster warmup time."
                self.default_compile_level = 3
            else:
                level_msg = "setting default_compile_level to 0."
                self.default_compile_level = 0
            logger.warning(
                "use_default_compile_configs is deprecated: please use default_compile_level instead. For backwards "
                f"compatibility, {level_msg}"
            )
        if self.max_cached_graphs is not None:  # Deprecated in 5.13
            logger.warning(
                "max_cached_graphs is deprecated: maximum number of graph is no longer an issue. Deprecated in 5.13."
            )
        if self.block_size is not None:  # Deprecated in 5.17
            logger.warning(
                "block_size is deprecated: please use page_size instead. For backwards compatibility, block_size will "
                "be used as the full attention page size."
            )
            self.page_size = self.block_size

    @property
    def cuda_graph_booleans(self) -> tuple[bool, bool]:
        """The cuda graph booleans for the varlen and decode paths."""
        if self.use_cuda_graph is None:
            return False, False
        if isinstance(self.use_cuda_graph, bool):
            return self.use_cuda_graph, self.use_cuda_graph
        return self.use_cuda_graph

    @property
    def fallback_max_blocks_per_request(self) -> int:
        """Fallback if no user-hint is given and decode path is available."""
        return 32


# Manager Class (User Interface)
class ContinuousBatchingManager:
    """Manager for handling continuous batching of generation requests. It provides a user interface for submitting
    generation requests, retrieving results, and managing the background generation thread. This class should not be
    created directly, but through one of the following entry points (all methods of the `ContinuousMixin` mixin):
    - `init_continuous_batching`
    - `continuous_batching_context_manager`
    - `generate_batch`
    """

    def __init__(
        self,
        model: ProtoPretrainedModel,
        generation_config: GenerationConfig,
        continuous_batching_config: ContinuousBatchingConfig,
        workload_hints: WorkloadHints | None = None,
    ) -> None:
        """Initialize the continuous batching manager.

        Args:
            model: The language model for generation
            generation_config: Configuration for generation parameters
            continuous_batching_config: Configuration for continuous batching parameters
            workload_hints: Workload hints for the continuous batching initialization (optional)
        """
        # Accumulators for request handling
        self.input_queue = queue.Queue(maxsize=continuous_batching_config.max_queue_size)
        self.cancel_queue: queue.Queue[str] = queue.Queue()
        self._request_counter = 0
        self._request_lock = threading.Lock()
        self._wake_up_loop = threading.Event()

        # Processor-related attributes
        self.background_thread_status = BackgroundThreadStatus()
        self.output_router = OutputRouter()
        self.batch_processor: ContinuousBatchProcessor | None = None
        self._generation_thread = None

        # Control flow attributes
        self.warmed_up = False  # Set to True after warmup is completed. Useful for persistent managers.

        # Model-related attributes
        self._original_attn_impl = None  # needs to be set before the model is switched to paged attention
        self.switch_to_cb_friendly_attn(model)
        self.model = model.eval()

        # Generation config related attributes
        self.generation_config = generation_config
        num_return_sequences = getattr(generation_config, "num_return_sequences", None)
        self.num_return_sequences = num_return_sequences if num_return_sequences is not None else 1

        # Initialize TP-related attributes
        self.distributed_helper = DistributedHelper(
            device_mesh=getattr(self.model, "_device_mesh", None),
            cpu_group_timeout=continuous_batching_config.cpu_group_timeout,
            tp_plan=getattr(self.model, "tp_plan", {}),
        )
        self.is_tp_driver = self.distributed_helper.is_tp_driver
        # If TP is on, check if NCCL graph mixing is disabled (helps with performance)
        if continuous_batching_config.disable_nccl_graph_mixing:
            self.distributed_helper.maybe_warn_nccl_graph_mixing()

        # Turn the classic logits processors into a CB-friendly version
        self.logit_processor = ContinuousBatchingLogitsProcessorList(
            logits_processor=self.model._get_logits_processor(generation_config),
            per_request_processors=continuous_batching_config.per_request_processors,
            drop_unsupported_processors=continuous_batching_config.drop_unsupported_processors,
        )

        # Fully resolve the continuous batching config now that we have the model, the config and the logit processor
        self.continuous_batching_config = resolve_continuous_batching_config(
            config=self.model.config.get_text_config(),
            cb_config=continuous_batching_config,
            workload_hints=workload_hints,
            has_logit_processors=self.logit_processor.do_processing,
        )
        # This is an approximation until the cache is created: it will infer the correct value in cache.__init__
        self._use_prefix_sharing = self.continuous_batching_config.allow_block_sharing

    def switch_to_cb_friendly_attn(self, model: ProtoPretrainedModel) -> None:
        """Switch the attn implementation to one that is CB friendly: try to find a flash implementation if flash is
        requested and, in any cases, switch to a paged implementation."""
        # The self._original_attn_impl is set only if the attn implementation is changed (makes this fn idempotent)
        original_attn_impl = model.config._attn_implementation
        target_implem = original_attn_impl

        # Check if flash attention is supported and available
        is_flash = is_flash_attention_requested(requested_attention_implementation=target_implem)
        is_paged = "paged|" in target_implem
        if not is_flash and not is_paged and model._supports_flash_attn:
            # Try to use FA3, then FA2, then give up. Both regular package or kernels is fine.
            if is_flash_attn_3_available(kernels_fallback_ok=True):
                version = 3
            elif is_flash_attn_2_available(kernels_fallback_ok=True):
                version = 2
            else:
                version = None
            # Change and warn
            msg = "Continuous batching is much better when using flash attention."
            if version is not None:
                target_implem = f"flash_attention_{version}"  # no "paged|" prefix here to enter the branch below
                logger.warning(
                    f"{msg} Switching from {original_attn_impl} to {target_implem}. "
                    "If you need to use eager or sdpa, use paged|eager or paged|sdpa as the `attn_implementation`."
                )
            else:
                logger.info(f"{msg} Consider using a flash `attn_implementation` when loading the model.")

        # Switch to a paged implementation (always entered if conversion to flash happened)
        if "paged|" not in target_implem:
            model.set_attn_implementation(f"paged|{target_implem}")
            self._original_attn_impl = original_attn_impl

    def warmup(self) -> None:
        """Pre-capture CUDA graphs for varlen and decode paths by running dummy batches. Initializes the batch
        processor if not already done."""
        if self.batch_processor is None:
            self.batch_processor = self._create_batch_processor()
        self.batch_processor.warmup(self.model)
        self.warmed_up = True

    # --------------------------------------------- CONTROL FLOW METHODS --------------------------------------------- #

    def is_running(self) -> bool:
        """Returns True if the background generation thread has been started and is still alive."""
        return self._generation_thread is not None and self._generation_thread.is_alive()

    def start(self) -> None:
        """Start the background generation thread."""
        if self.is_running():
            logger.warning("Manager thread is already running.")
            return
        self.background_thread_status.clear()
        self._generation_thread = threading.Thread(target=self._run_generation_loop)
        self._generation_thread.start()

    def stop(
        self,
        block: bool = True,
        timeout: float | None = None,
        keep_for_next_session: bool = False,
        hard_stop: bool = False,
    ) -> None:
        """Stop the background generation thread. If the `block` flag is set to True, then this method waits for the
        thread to stop for a maximum time of `timeout` seconds (None means no timeout). If the `keep_for_next_session`
        flag is set to True, then the manager is cached on the model for future use. If the `hard_stop` flag is set,
        the background generation thread will be stopped immediately and pending requests will be failed."""

        # We expect the batch processor to be initialized at this point. Warn otherwise.
        if self.batch_processor is None:
            logger.warning("\nBatch processor was not initialized.")

        # If the manager is not started, warn and return.
        if self._generation_thread is None:
            msg = "Manager not started."
            if keep_for_next_session:
                msg += " Hence the unstarted manager will not be kept for next session."
            logger.warning(msg)
            return

        # Stopping and pausing are conflicting operations: a thread inside a pause cannot stop the manager, because that
        # would deadlock (pause waits for the stop to complete, stop hangs because the loop is paused)
        if self.background_thread_status.is_pause_requested(local=True):
            raise RuntimeError(
                "Cannot stop the manager from inside a pause: the generation loop is paused and cannot exit, so "
                "this would wait forever. Leave the `pause` context before calling `stop`."
            )

        # Signal the background thread to stop
        stop_trigger_time = perf_counter()
        stop_status = BackgroundThreadStatus.HARD_STOP if hard_stop else BackgroundThreadStatus.FLUSH_AND_STOP
        self.background_thread_status.request_stop(stop_status, self.distributed_helper.global_rank)
        # And maybe wait for that to happen
        if block:
            self.join(stop_trigger_time, timeout)

        # If the manager is not being kept for next session, we clear the batch processor
        if not keep_for_next_session:
            self.batch_processor = None
        # Otherwise, we keep the batch processor and cache the manager as a model attribute
        else:
            logger.info("Continuous batching manager will be kept for next session.")
            self.model._cached_continuous_batching_manager = self  # type: ignore

        # Restore the original attention implementation
        if self._original_attn_impl is not None:
            self.model.set_attn_implementation(self._original_attn_impl)
            self._original_attn_impl = None

        # In all cases, a little cleanup is good
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def join(self, stop_trigger_time: float, timeout: float | None = None) -> None:
        """Wait for the background thread to finish. Wait can be capped using the timeout argument (in seconds)."""
        # Early return if the thread is not running
        if self._generation_thread is None:
            return
        # Join (maybe w/ timeout) and check if the thread is still alive afterwards. If it is, then it means the thread
        # is still running despite the stop signal, so we warn the user who might expect otherwise.
        self._generation_thread.join(timeout=timeout)
        if self._generation_thread.is_alive():
            logger.warning(f"Generation thread did not exit after join timeout ({timeout}).")
        else:
            end = perf_counter()
            logger.info(f"Background generation thread stopped after {end - stop_trigger_time:.2f}s.")
            self._generation_thread = None

    def destroy(self) -> None:
        """Terminate the manager and release distributed resources. Safe to call multiple times. After calling this,
        the manager cannot be restarted."""
        if self.is_running():
            self.stop(block=True, keep_for_next_session=False)
        self.distributed_helper.destroy_cpu_comm_group()

    @contextmanager
    def pause(self):
        """A context manager that pauses the generation loop, so the calling thread can use the model, typically to
        update it in place. The thread only enters this context once the loop is paused, and the loop resumes on exit,
        keeping its cache and its in-flight requests: nothing is drained and no request is lost.
        Several threads may hold the pause at the same time, and the loop resumes once the last one leaves.
        If TP is on, all ranks must enter this context, otherwise other ranks will hang forever.
        """
        # Error out if the caller asks for a pause while no generation loop is running
        if not self.is_running():
            raise RuntimeError("Cannot pause generation while no generation loop is running.")

        self.background_thread_status.acquire_pause(self._wake_up_loop)
        try:
            yield
        finally:
            self.background_thread_status.release_pause()

    # ---------------------------- REQUEST SUBMISSION, CANCELLATION AND RETRIEVAL METHODS ---------------------------- #

    def add_request(
        self,
        input_ids: list[int],
        request_id: str | None = None,
        max_new_tokens: int | None = None,
        streaming: bool = False,
        record_timestamps: bool = False,
        eos_token_id: int | list[int] | None = None,
        **logit_processor_kwargs: Any,
    ) -> str | None:
        """Add a new generation request to the queue. If the process is not a TP driver, this is a no-op.

        Args:
            input_ids: Input token IDs to use as prompt
            request_id: Optional custom request ID (auto-generated if None)
            max_new_tokens: Maximum number of new tokens to generate
            streaming: Whether to stream tokens as they're generated
            record_timestamps: Whether to record timestamps for each generated token
            eos_token_id: End-of-sequence token ID(s)
            logit_processor_kwargs: Keyword arguments for the logits processor.

        Returns:
            str | None: The request ID if the process is a TP driver, None otherwise.
        """
        # If this process is not a TP driver, request submission is a no-op
        if not self.is_tp_driver:
            return None
        # If the manager is not accepting new requests, stop here.
        denial_msg = self.background_thread_status.can_accept_new_requests()
        if denial_msg is not None:
            preview = f"{input_ids[:3]}"[:-1] + ", ..., " + f"{input_ids[-3:]}"[1:]
            logger.warning(f"{denial_msg}. Request with ids {preview} will be dropped.")
            return None

        if request_id is None:
            with self._request_lock:
                request_id = f"req_{self._request_counter}"
                self._request_counter += 1
        max_new_tokens = self.generation_config.max_new_tokens if max_new_tokens is None else max_new_tokens
        eos_token_id = self.generation_config.eos_token_id if eos_token_id is None else eos_token_id

        # NOTE: do we want to handle a case when the user wants token ids returned instead of decoded text?
        state = RequestState(
            request_id=request_id,
            initial_tokens=list(input_ids),
            num_children=self.num_return_sequences - 1,
            record_timestamps=record_timestamps,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            streaming=streaming,
            logit_processor_kwargs=logit_processor_kwargs,
        )

        # Use block=True with timeout to handle backpressure if queue is full
        self.input_queue.put(state, block=True, timeout=10)
        self._wake_up_loop.set()
        return request_id

    def add_requests(
        self,
        inputs: list[list[int]],
        max_new_tokens: int | None = None,
        streaming: bool = False,
        record_timestamps: bool = False,
        **logit_processor_kwargs: Any,
    ) -> list[str]:
        """Utility function to batch `add_request` and return their IDs. Check its documentation for more details."""
        # Infer the request ids of all incoming requests
        num_requests = len(inputs)
        with self._request_lock:
            request_ids = [f"req_{i}" for i in range(self._request_counter, self._request_counter + num_requests)]
            self._request_counter += num_requests
        # If there is prefix sharing, we sort the inputs to maximize cache hits but keep the order of the requests
        ids_and_inputs = list(zip(request_ids, inputs))
        if self._use_prefix_sharing:
            ids_and_inputs = sorted(ids_and_inputs, key=lambda x: x[1], reverse=True)
        # EOS determination order: generation config -> model config -> -1 (no EOS)
        eos_token_id = self.generation_config.eos_token_id
        eos_token_id = self.model.config.eos_token_id if eos_token_id is None else eos_token_id
        eos_token_id = -1 if eos_token_id is None else eos_token_id
        # Add requests in order
        for request_id, input_ids in ids_and_inputs:
            self.add_request(
                input_ids=input_ids,
                request_id=request_id,
                max_new_tokens=max_new_tokens,
                streaming=streaming,
                record_timestamps=record_timestamps,
                eos_token_id=eos_token_id,
                **logit_processor_kwargs,
            )
        return request_ids

    def cancel_request(self, request_id: str) -> None:
        """Cancel a request by its ID. If this called from a process that is not a TP driver, it's a no-op: only TP
        driver processes interact with the manager."""
        if self.is_tp_driver:
            self.cancel_queue.put(request_id)
            self._wake_up_loop.set()

    # TODO (remi-or) : handle benchmarking properly when updating / fixing the requeue logic
    # TODO (remi-or) : this NEEDS to get fixed in a future PR -- it's quite wasteful
    def get_result(self, request_id: str | None = None, timeout: float | None = None) -> GenerationOutput | None:
        """Retrieve one result from the output queue. If an ID is provided, returns the first matching request. If a
        timeout is provided, returns None after the timeout (in seconds)."""
        # Stop if the output queue is empty and the bg thread is not going to produce new results (crashed or stopped)
        if self.output_router.output_queue.empty():  # noqa: SIM102
            if self._generation_thread is None or self.background_thread_status.fatal_error is not None:
                return None
        # Otherwise, wait for a result from the output queue
        try:
            result = self.output_router.output_queue.get(block=True, timeout=timeout)
            if request_id is not None and result.request_id != request_id:
                self.output_router.output_queue.put(result)
                return None
            return result
        except queue.Empty:
            return None

    def __iter__(self):
        """Iterate over results as they become available."""
        while self._generation_thread is not None and self._generation_thread.is_alive():
            result = self.get_result(timeout=0.1)
            if result is not None:
                yield result

    def request_id_iter(self, request_id: str) -> Generator[GenerationOutput]:
        """Iterate over results matching a specific request id (blocking).

        Uses the shared output queue with requeue. For high-concurrency serving,
        use :meth:`register_result_handler` instead.
        """
        while self._generation_thread is not None and self._generation_thread.is_alive():
            result = self.get_result(request_id=request_id, timeout=0.1)
            if result is not None:
                yield result
                if result.is_finished():
                    return

    def register_result_handler(self, request_id: str, callback: Callable) -> None:
        """Register a callback for result delivery (streaming or non-streaming).

        The callback is invoked on the event loop via ``call_soon_threadsafe`` each time a result is produced for this
        request. For streaming requests, this happens on every token; for non-streaming, only on completion. The handler
        is automatically cleaned up when the request finishes.

        Args:
            request_id (`str`): The request ID to receive outputs for.
            callback (`callable`): Called with a ``GenerationOutput`` for each result.
        """
        loop = asyncio.get_running_loop()

        def _auto_cleanup(result):
            callback(result)
            if result.is_finished():
                with self.output_router._lock:
                    self.output_router.result_handlers.pop(request_id, None)

        with self.output_router._lock:
            self.output_router.result_handlers[request_id] = (_auto_cleanup, loop)

    # ---------------------------------------- BACKGROUND THREAD ONLY METHODS ---------------------------------------- #

    def _generation_loop_body(self, batch_processor: ContinuousBatchProcessor, bootstrapping: bool) -> bool:
        """Body of the generation loop. Returns True if the loop should continue, False otherwise. Behaves differently
        if this is for bootstrapping an async run: in that case, there is no need to update the batch, and the first
        step should exit the bootstrapping loop."""
        # If some request is available, perform a generation step
        requests_available = batch_processor.prepare_next_batch()
        if requests_available:
            self._generation_step()
            self.current_batch += 1
            if bootstrapping:  # no update when bootstrapping an async batching generation
                return False
            else:
                batch_processor.update_batch()
                return True
        # Stop waiting if the TP group is hard-stopping
        elif self.background_thread_status.tp_status == BackgroundThreadStatus.HARD_STOP or (
            self.background_thread_status.tp_status == BackgroundThreadStatus.FLUSH_AND_STOP
            and not batch_processor.has_pending_requests()
        ):
            return False
        # Otherwise, we wait for new requests and retry
        else:
            self._wake_up_loop.wait(timeout=0.1)  # wait for new requests instead of busy-spinning.
            self._wake_up_loop.clear()
            return True

    def _run_generation_loop(self) -> None:
        """Main processing loop running in the background thread."""
        batch_processor = None

        # Everything is inside this try / except / finally block so we can handle critical errors gracefully
        try:
            # Scope the device for the generation loop (thread-scoped)
            if self.model.device.type == "cuda" and self.model.device.index is not None:
                torch.cuda.set_device(self.model.device)

            # Start the generation loop
            batch_processor = self._create_batch_processor()
            self.batch_processor = batch_processor  # register the batch processor for main thread access
            self.current_batch = 0

            # If using the async API, we bootstrap the first batch w/out update
            if batch_processor.use_async_batching:
                while self._generation_loop_body(batch_processor, bootstrapping=True):
                    pass

            # The loop continues until a stop signal has been broadcasted in the TP group
            while self._generation_loop_body(batch_processor, bootstrapping=False):
                pass

            # In async mode, the last batch's results are still in flight: switch to the right IO pair and process them
            # Also happens for a hard stop, since the results are already available on the device
            if isinstance(batch_processor.inputs_and_outputs, ContinuousBatchingAsyncIOs):
                batch_processor.inputs_and_outputs.current_pair = 1 - batch_processor.inputs_and_outputs.current_pair
                batch_processor.update_batch()

            # This should be a no-op unless a user asked for a hard stop
            error = RuntimeError(
                f"Generation loop finished before this request completed w/ {self.background_thread_status.tp_status = }"
            )
            self._fail_all_remaining_requests(error, batch_processor)

        # All exceptions are caught here so we can shut down the thread and TP group as gracefully as possible
        except Exception as e:
            logger.error(f"Error in generation loop: {e}", exc_info=True)
            self._handle_critical_error(e, batch_processor)
        finally:
            self.background_thread_status.mark_as_stopped()
            logger.info("Generation loop finished and background thread exited successfully.")

    def _generation_step(self) -> None:
        """Perform a single generation step. This is mostly cuda graphed"""
        if self.batch_processor is None:
            raise RuntimeError("Tried to perform a generation step before the batch processor was initialized.")
        self.batch_processor._generation_step(self.model)

    def _create_batch_processor(self) -> ContinuousBatchProcessor:
        """Create a new batch processor. If an already initialized batch processor exists, it is reset and returned."""
        # Early return if a batch processor exists already
        batch_processor = getattr(self, "batch_processor", None)
        if isinstance(batch_processor, ContinuousBatchProcessor):
            batch_processor.reset()
            return batch_processor

        # Create the PagedAttentionCache
        supports_logits_to_keep = getattr(self.model, "_supports_logits_to_keep", None)
        paged_attention_cache = PagedAttentionCache(
            config=self.model.config.get_text_config(),
            continuous_batching_config=self.continuous_batching_config,
            device=self.model.device,
            distributed_helper=self.distributed_helper,
            dtype=self.model.dtype,
            model_supports_logits_to_keep=callable(supports_logits_to_keep) and supports_logits_to_keep(),
        )
        # Update the approximation now that we know if there is prefix sharing
        self._use_prefix_sharing = paged_attention_cache.use_prefix_sharing
        # And update continuous batching config now that we have concrete values
        update_cb_config_after_cache_creation(
            cb_config=self.continuous_batching_config,
            num_blocks=paged_attention_cache.num_blocks,
            max_batch_tokens=paged_attention_cache.max_batch_tokens,
        )

        # Disable the decode path if the model has sliding window attention (TODO)
        if SLIDING_ATTENTION in paged_attention_cache.cache_allocators:
            self.continuous_batching_config.max_blocks_per_request = 0

        # Retrieve the scheduler class
        scheduler_type = self.continuous_batching_config.scheduler_type
        scheduler_cls = SCHEDULER_MAPPING.get(scheduler_type, None)
        if scheduler_cls is None:
            logger.warning(f"Scheduler '{scheduler_type}' not found. Defaulting to FIFO.")
            scheduler_cls = FIFOScheduler
        # Instantiate the actual scheduler
        scheduler = scheduler_cls(
            cache=paged_attention_cache,
            safety_margin=self.continuous_batching_config.safety_margin,
            max_requests_per_batch=self.continuous_batching_config.max_requests_per_batch,
        )

        # Create the batch processor
        batch_processor = ContinuousBatchProcessor(
            cache=paged_attention_cache,
            config=self.model.config.get_text_config(),
            generation_config=self.generation_config,
            continuous_batching_config=self.continuous_batching_config,
            logit_processor=self.logit_processor,
            input_queue=self.input_queue if self.is_tp_driver else None,
            cancel_queue=self.cancel_queue if self.is_tp_driver else None,
            output_router=self.output_router,
            background_thread_status=self.background_thread_status,
            model_device=self.model.device,
            model_dtype=self.model.dtype,
            scheduler=scheduler,
            distributed_helper=self.distributed_helper,
        )
        return batch_processor

    def _handle_critical_error(self, error: Exception, batch_processor: ContinuousBatchProcessor | None) -> None:
        """Handle critical errors that terminate the generation loop."""
        # Request a hard stop., only on this rank. Other ranks will be notified at the comm
        self.background_thread_status.request_stop(
            status=BackgroundThreadStatus.HARD_STOP, global_rank=self.distributed_helper.global_rank
        )
        # Fail all remaining requests
        self._fail_all_remaining_requests(error, batch_processor)
        # After failing the remaining requests (and so retrieving their partial outputs), record the fatal error
        self.background_thread_status.record_fatal_error(error)
        # Communicate to other ranks in the TP group that the group is stopping (they could have not crashed)
        # Since the other processes need to reach the collective, it may take a few seconds to complete.
        self.distributed_helper.tp_all_reduce_state(0, BackgroundThreadStatus.HARD_STOP)

    def _fail_all_remaining_requests(self, error: Exception, batch_processor: ContinuousBatchProcessor | None) -> None:
        """Fail all remaining requests in the input queue and active requests."""
        # Fail pending requests in input queue
        try:
            while True:
                req_data = self.input_queue.get_nowait()
                if batch_processor is not None:
                    batch_processor._handle_request_error(error, req_data)
                else:
                    self.output_router.fail_and_deliver(req_data, error)
        except queue.Empty:
            pass
        # Fail active and waiting requests
        if batch_processor is not None:
            batch_processor.fail_all_requests(error)


class ContinuousMixin:
    """Mixin class for models to add continuous batching capabilities. Continuous batching has three entry points:
    - `init_continuous_batching`, which is the actual entry point for continuous batching
    - `continuous_batching_context_manager`, which itself is a wrapper around `init_continuous_batching`
    - `generate_batch`, which is really a wrapper around `continuous_batching_context_manager`

    They are defined in this order. Any change made to any of those three entry points should be reflected in the other
    two.
    """

    generation_config: GenerationConfig

    @torch.no_grad()
    def init_continuous_batching(
        self,
        generation_config: GenerationConfig | None = None,
        continuous_batching_config: ContinuousBatchingConfig | None = None,
        workload_hints: WorkloadHints | None = None,
    ) -> ContinuousBatchingManager:
        """Initialize a manager for continuous batching inference.

        Args:
            generation_config: An optional generation configuration, which may contain a CompileConfig object
            continuous_batching_config: An optional continuous batching configuration
            workload_hints: Optional WorkloadHints to help the continuous batching manager make better decisions for
                default values
        Returns:
            `ContinuousBatchingManager`: The manager instance to add requests and retrieve results.
        """
        # Mandatory attributes
        if not hasattr(self, "config") or not hasattr(self, "device") or not hasattr(self, "dtype"):
            raise AttributeError("Model must have 'config', 'device', and 'dtype' attributes.")

        # If a persistent manager is found we return it
        cached_manager = getattr(self, "_cached_continuous_batching_manager", None)
        if isinstance(cached_manager, ContinuousBatchingManager):
            logger.info(
                "Cached continuous batching manager found: it will be re-used instead of creating a new one. If you"
                " want to create a new manager, you should call `destroy_cached_continuous_batching_manager` first."
            )
            cached_manager.switch_to_cb_friendly_attn(self)  # might have switched in .stop
            return cached_manager

        # Retrieve generation config
        gen_config = generation_config if generation_config is not None else self.generation_config
        if gen_config is None:
            raise ValueError("A GenerationConfig must be provided or set in the model.")
        # Warn about EOS
        if gen_config.eos_token_id is None:
            logger.warning("`eos_token_id` not set in GenerationConfig. Setting to -1 (disabled).")
            gen_config.eos_token_id = -1

        # Retrieve continuous batching config, or create it if none is provided
        if continuous_batching_config is None:
            if isinstance(getattr(gen_config, "continuous_batching_config", None), ContinuousBatchingConfig):
                logger.warning(
                    "Passing ContinuousBatchingConfig through GenerationConfig is deprecated. Please pass it separately"
                    " using the continuous_batching_config kwarg."
                )
                continuous_batching_config = gen_config.continuous_batching_config
            else:
                continuous_batching_config = ContinuousBatchingConfig()

        # Create and return the manager
        return ContinuousBatchingManager(
            model=self,
            generation_config=gen_config,
            continuous_batching_config=continuous_batching_config,
            workload_hints=workload_hints,
        )

    def destroy_cached_continuous_batching_manager(self) -> None:
        """Destroy the cached continuous batching manager and free GPU resources."""
        cached_manager = getattr(self, "_cached_continuous_batching_manager", None)
        if isinstance(cached_manager, ContinuousBatchingManager):
            cached_manager.destroy()
            delattr(self, "_cached_continuous_batching_manager")

    @contextmanager
    @torch.no_grad()
    def continuous_batching_context_manager(
        self,
        generation_config: GenerationConfig | None = None,
        block: bool = True,
        timeout: float | None = None,
        continuous_batching_config: ContinuousBatchingConfig | None = None,
        persistent_manager: bool = False,
        warmup: bool = True,
        workload_hints: WorkloadHints | None = None,
    ) -> Generator[ContinuousBatchingManager]:
        """A context manager to safely use the continuous batching manager. Arguments are similar to the ones of
        `init_continuous_batching`, except for:
            - block: whether to block the thread when stopping the manager. Default is True.
            - timeout: maximum time to wait for the thread to stop. Default is None (no timeout).
            - warmup: whether to pre-capture CUDA graphs at the largest sizes before running. Default is True.
        """
        manager = self.init_continuous_batching(
            generation_config=generation_config,
            continuous_batching_config=continuous_batching_config,
            workload_hints=workload_hints,
        )
        if warmup and not manager.warmed_up:
            # TODO: have a progress bar for the warmup as well, like other inference engine
            logger.info("Warming up for continuous batching...")
            start = perf_counter()
            manager.warmup()
            logger.info(f"Warming up completed in {perf_counter() - start:.2f}s.")
        manager.start()
        try:
            yield manager
        finally:
            # This is a dummy log needed for the logs of stop to show. It won't show.
            logger.debug("Continuous batching loop finished")
            manager.stop(block=block, timeout=timeout, keep_for_next_session=persistent_manager)
            if not persistent_manager:
                manager.destroy()

    # TODO: support streaming
    @torch.no_grad()
    def generate_batch(
        self,
        inputs: list[list[int]],
        generation_config: GenerationConfig | None = None,
        continuous_batching_config: ContinuousBatchingConfig | None = None,
        record_timestamps: bool = False,
        progress_bar: bool = False,
        persistent_manager: bool = False,
        warmup: bool = True,
        **kwargs,
    ) -> dict[str, GenerationOutput]:
        """Generate sequences for a batch of prompts using continuous batching.

        Args:
            inputs: List of input token sequences (prompts)
            generation_config: Optional generation configuration
            continuous_batching_config: Optional continuous batching configuration
            record_timestamps: If set to true, the requests will have a timestamp for each token generated
            progress_bar: If set to true, a progress bar will be displayed
            persistent_manager: whether to persist the manager after the generation is finished. Default is False.
            warmup: whether to pre-capture CUDA graphs before processing requests. Default is True.
        Returns:
            `dict[str, GenerationOutput]`: a dictionary of request ids to GenerationOutput objects
        """
        # If no input are provided, return an empty dictionary
        if not inputs:
            return {}

        # If the logger level is less than DEBUG, disable the progress bar
        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.info("Progress bar is disabled when logger level is less than DEBUG")
            progress_bar = False

        # Compute the total number of requests
        gen_cfg = self.generation_config if generation_config is None else generation_config
        num_return_sequences = gen_cfg.num_return_sequences if gen_cfg.num_return_sequences is not None else 1
        num_requests = len(inputs) * num_return_sequences

        # Extract max_new_tokens from kwargs because it's the only expected kwarg
        max_new_tokens = kwargs.pop("max_new_tokens", None)
        max_new_tokens = gen_cfg.max_new_tokens if max_new_tokens is None else max_new_tokens

        # Compute workload hints
        workload_hints = WorkloadHints(
            max_prompt_length=max(len(input_ids) for input_ids in inputs),
            max_generated_length=max_new_tokens if max_new_tokens is not None else 0,
            num_requests=num_requests,
        )
        if persistent_manager:
            logger.warning(
                "Since you passed `persistent_manager=True`, the manager will be kept alive after the generation is "
                "finished. However, it was sized specifically for the requests passed in `generate_batch`. If you plan "
                "to reuse the manager for a very different workload, you might want to create a new manager instead."
            )

        # Prepare context managers for the main loop
        manager_cm = self.continuous_batching_context_manager(
            generation_config=generation_config,
            continuous_batching_config=continuous_batching_config,
            block=True,
            timeout=5,
            persistent_manager=persistent_manager,
            warmup=warmup,
            workload_hints=workload_hints,
        )
        logging_cm = logging_redirect_tqdm([logger])
        pbar_cm = tqdm(
            total=num_requests,
            disable=(not progress_bar),
            desc=f"Solving {num_requests} requests",
            unit="request",
        )

        # Main loop
        results = {}
        finished_count = 0
        request_ids = []  # if the manager fails before add_requests returns, request_ids should not be unbounded
        with manager_cm as manager, logging_cm, pbar_cm as pbar:
            try:
                request_ids = manager.add_requests(
                    inputs=inputs, max_new_tokens=max_new_tokens, record_timestamps=record_timestamps
                )
                while finished_count < num_requests:
                    result = manager.get_result(timeout=1)
                    if result:
                        req_id = result.request_id
                        if result.is_finished():
                            results[req_id] = result
                            finished_count += 1
                            pbar.update(1)
                    elif not manager.is_running():
                        logger.error("Generation thread terminated unexpectedly.")
                        # This helps get some information in stdout
                        print("Returning results of generate_batch despite unexpected termination.")
                        break

            except Exception as e:
                logger.error(f"Error during batch generation: {e}", exc_info=True)  # noqa: G201

        # Re-order requests to match the order of the inputs, forked children right after their parent
        reordered_results = {}
        missing_keys, failed_keys = [], []
        for request_id in request_ids:
            # If there are multiple return sequences, taken it into account
            selected_ids = [f"{request_id}__child#{i}" for i in range(num_return_sequences - 1)]
            selected_ids.append(request_id)
            # Add the parent and child IDs to the list
            for selected_id in selected_ids:
                result = results.get(selected_id)
                if result is not None:
                    reordered_results[selected_id] = result
                    if result.error is not None:
                        failed_keys.append(selected_id)
                else:
                    missing_keys.append(selected_id)

        if missing_keys:
            logger.error(f"Requests {missing_keys} not found in results.")
        if failed_keys:
            logger.error(f"Requests {failed_keys} failed during generation.")
        return reordered_results
