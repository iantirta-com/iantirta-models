
import re
from contextlib import contextmanager
from functools import lru_cache

from packaging import version

from iantirta.models.tools._deps import _is_package_available, _make_compile_constant

KERNELS_MIN_VERSION = "0.17.0"
KERNELS_MAX_VERSION = "0.18.0"

_MISSING_KERNELS_MESSAGE = (
    "`kernels` is either not installed or uses an incompatible version. Please install a compatible version "
    f"({KERNELS_MIN_VERSION} <= version < {KERNELS_MAX_VERSION}), e.g. `pip install kernels=={KERNELS_MIN_VERSION}`"
)


@lru_cache
@_make_compile_constant
def is_kernels_available(MIN_VERSION: str = KERNELS_MIN_VERSION, MAX_VERSION: str = KERNELS_MAX_VERSION) -> bool:
    is_available, kernels_version = _is_package_available("kernels", return_version=True)
    viable_version = False
    if kernels_version != "N/A":
        viable_version = version.parse(kernels_version) >= version.parse(MIN_VERSION) and version.parse(
            kernels_version
        ) < version.parse(MAX_VERSION)
    return is_available and viable_version


if is_kernels_available():
    pass
else:
    # Stub to make decorators in transformers work when `kernels`
    # is not installed.
    def use_kernel_forward_from_hub(*args, **kwargs):
        def decorator(cls):
            return cls

        return decorator


def is_kernel(attn_implementation: str | None) -> bool:
    """Check whether `attn_implementation` matches a kernel pattern from the hub."""
    return (
        attn_implementation is not None
        and re.search(r"^[^/:]+/[^/:]+(?:@[^/:]+)?(?::[^/:]+)?$", attn_implementation) is not None
    )


def kernelize(model: "PreTrainedModel", mode: "Mode | None" = None):
    """Temporarily register hidden kernel wrappers so `kernelize` can discover and replace them."""
    if not is_kernels_available():
        raise ImportError(_MISSING_KERNELS_MESSAGE)

    mode = Mode.INFERENCE if not model.training else Mode.TRAINING if mode is None else mode
    device = Device(type=get_device_type(model.device))

    if model.kernel_config is not None:
        inherit_mapping = not model.kernel_config.use_local_kernel and model.kernel_config.inherit_mapping
        with use_kernel_mapping(model.kernel_config.kernel_mapping, inherit_mapping=inherit_mapping):
            _kernels_kernelize(model, device=device, mode=mode)
    else:
        _kernels_kernelize(model, device=device, mode=mode)

    model._use_kernels = True


# Whether to allow hub kernels coming from untrusted repos, i.e. repos outside `kernels-community`
ALLOW_ALL_KERNELS = False


@contextmanager
def allow_all_hub_kernels():
    """
    Context manager used to adjust the value of the global `ALLOW_HUB_KERNELS`. This is needed, as this argument
    cannot be forwarded directly to the `__init__` of the models, where we set the attention implementation.
    """
    global ALLOW_ALL_KERNELS

    try:
        ALLOW_ALL_KERNELS = True

        yield
    finally:
        # Set back the original
        ALLOW_ALL_KERNELS = False


def make_parent_class_for_kernel_fusion(
    parent_cls: type,
    child_names: list[str],
    kernel_cls: type,
) -> type:
    """
    Create a new class that inherits from `parent_cls` and fuses the child modules specified in `child_names
    with the provided `kernel_cls`.
    The first child in `child_names` will be replaced with the `kernel_cls`, and the rest will be replaced with
    `nn.Identity()` to keep the same interface.
    """
    original_init = parent_cls.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        children = [getattr(self, name) for name in child_names]
        kernel_instance = kernel_cls(*children)
        setattr(self, child_names[0], kernel_instance)
        for name in child_names[1:]:
            setattr(self, name, nn.Identity())

    patched_cls = type(f"Fused{parent_cls.__name__}", (parent_cls,), {"__init__": patched_init})
    patched_cls.__qualname__ = f"Fused{parent_cls.__qualname__}"
    return patched_cls


def register_kernel_replacements_and_fusions(
    cls: "type[PreTrainedModel]",
    config: "PretrainedConfig",
    kernel_config: "KernelConfig",
) -> None:
    if not hasattr(cls, "config_class") or not hasattr(cls.config_class, "model_type"):
        raise ValueError(f"Model {cls.__name__} has no config_class or model_type.")
    model_type = cls.config_class.model_type

    patch_mapping: dict[str, type] = {}
    new_mapping: dict = {}

    # We might need to instantiate the model on meta device.
    # We do it lazily, only if we encounter a fused kernel.
    meta_model = None

    for layer_name, hub_repo in kernel_config.kernel_mapping.items():
        if isinstance(hub_repo, (str, tuple)):
            hub_repo = {None: hub_repo}

        if isinstance(hub_repo, dict):
            if len(hub_repo.values()) != 1:
                raise ValueError(
                    f"Expected exactly one kernel repo regardless of device/mode specificity, got {hub_repo}"
                )
        else:
            raise ValueError(f"Invalid hub repo {hub_repo!r} for layer {layer_name!r}")

        hub_repo = next(iter(hub_repo.values()))

        # Infer metadata (revision/version/trust_remote_code)
        if isinstance(hub_repo, tuple):
            repo_str, metadata = hub_repo

            revision = metadata.get("revision", None)
            version = metadata.get("version", None)
            trust_remote_code = metadata.get("trust_remote_code", False) or ALLOW_ALL_KERNELS
            metadata = {"version": version} if version is not None else {"revision": revision}
            metadata |= {"trust_remote_code": trust_remote_code}

            final_repo = (repo_str, metadata)
        else:
            repo_str = hub_repo
            metadata = {"version": 1, "trust_remote_code": ALLOW_ALL_KERNELS}
            final_repo = (repo_str, metadata)

        repo_id, _, layer_name_in_repo = repo_str.partition(":")
        if not repo_id or not layer_name_in_repo:
            raise ValueError(f"Invalid kernel repo string {repo_str!r} for layer {layer_name!r}")

        if kernel_config.use_local_kernel:
            repo = LocalLayerRepository(
                repo_path=Path(repo_id),
                layer_name=layer_name_in_repo,
            )
        else:
            repo = LayerRepository(
                repo_id=repo_id,
                layer_name=layer_name_in_repo,
                **metadata,
            )

        kernel_cls = repo.load()

        if kernel_cls is None:
            raise ValueError(f"Could not load kernel class from hub_repo={hub_repo!r}")

        kernel_mod = sys.modules.get(kernel_cls.__module__)
        layout_cls = getattr(kernel_mod, f"{kernel_cls.__name__}Layout", None) if kernel_mod else None

        if layout_cls is not None and "forward" not in layout_cls.__dict__:

            @functools.wraps(kernel_cls.forward)
            def _noop_forward(self, *args, **kwargs):
                pass

            layout_cls.forward = _noop_forward

        # Case 1: no fusion.
        if isinstance(layer_name, str):
            # No layout class: stateless kernel, leave for kernels.kernelize.
            if layout_cls is None:
                new_mapping[layer_name] = final_repo
                continue

            # Register the layout class as a monkey patch for the parent module containing the target layer.
            layout_cls.kernel_layer_name = kernel_cls.__name__
            patch_mapping[layer_name] = layout_cls

            # Keep the original repo string so kernelize can replace the layout's forward.
            new_mapping[kernel_cls.__name__] = final_repo

        # Case 2: fusion.
        elif isinstance(layer_name, tuple):
            if layout_cls is None:
                raise ValueError(
                    f"Fused kernel {kernel_cls.__name__!r} requires a companion layout class "
                    f"named '{kernel_cls.__name__}Layout' in the same module."
                )

            layout_cls.kernel_layer_name = kernel_cls.__name__

            glob_patterns = [item[1] for item in layer_name]
            parent_patterns = [p.rsplit(".", 1)[0] for p in glob_patterns]

            if len(set(parent_patterns)) != 1:
                raise ValueError(
                    f"All patterns for a fused kernel must share the same parent module, got {glob_patterns}"
                )

            parent_pattern = parent_patterns[0].replace("*", r"\w+")
            child_names = [p.rsplit(".", 1)[1] for p in glob_patterns]

            if meta_model is None:
                with torch.device("meta"):
                    meta_model = cls(config)

            matched_any = False
            for name, module in meta_model.named_modules():
                if not re.fullmatch(parent_pattern, name):
                    continue
                if not all(hasattr(module, child) for child in child_names):
                    raise ValueError(
                        f"Module {name!r} does not have the expected child modules {child_names} required for "
                        f"the fused kernel {kernel_cls.__name__!r}"
                    )
                matched_any = True
                module_cls = type(module)
                patch_mapping[module_cls.__name__] = make_parent_class_for_kernel_fusion(
                    module_cls, child_names, layout_cls
                )

            if not matched_any:
                raise ValueError(
                    f"No module matched pattern {parent_pattern!r} for fused kernel {kernel_cls.__name__!r}. "
                    f"Provide the full dotted path from the model root."
                )

        register_patch_mapping(patch_mapping, overwrite=True)

        if hasattr(layout_cls, "conversion_mapping"):
            existing = get_checkpoint_conversion_mapping(model_type)
            transforms = list(layout_cls.conversion_mapping)
            if existing is not None:
                transforms = existing + transforms
            register_checkpoint_conversion_mapping(model_type, transforms, overwrite=True)

        new_mapping[kernel_cls.__name__] = final_repo

    kernel_config.kernel_mapping = new_mapping
