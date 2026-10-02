
import importlib
import logging

logger = logging.getLogger(__name__)


PACKAGE_DISTRIBUTION_MAPPING = importlib.metadata.packages_distributions()


def _candidate_distribution_names(pkg_name: str) -> list[str]:
    """Distribution names to try for the
    import name `pkg_name`,most likely first.

    The distribution name may differ from the
    import name (`PIL` is imported, but `pillow` is distributed),
    and `packages_distributions()` maps one to the other
    -- but only for wheels shipping a `top_level.txt` on
    Python < 3.12, which `torch` >= 2.14 does not.
    So keep the import name itself as a candidate.
    """
    # Per PEP 503, underscores and hyphens are equivalent in package names.
    normalized_pkg_name = pkg_name.replace("_", "-")
    distributions = PACKAGE_DISTRIBUTION_MAPPING.get(pkg_name, [])
    candidates = [
        *(
            name
            for name in (
                normalized_pkg_name,
                pkg_name
            )
            if name in distributions
        ),
        *distributions,
        normalized_pkg_name,
        pkg_name,
    ]
    # de-duplicate, keeping first-seen order
    return list(dict.fromkeys(candidates))


def _is_package_available(
    pkg_name: str,
    return_version: bool = False
) -> tuple[bool, str]:
    """Check if `pkg_name` exist, and optionally try to get its version"""

    spec = importlib.util.find_spec(pkg_name)
    package_exists = spec is not None
    package_version = "N/A"

    if package_exists and return_version:
        for distribution_name in _candidate_distribution_names(pkg_name):
            try:
                package_version = importlib.metadata.version(distribution_name)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        else:
            # No metadata under any candidate name
            # (editable install without a `dist-info`, for example).
            # Last resort: importing defeats the
            # lazy imports these checks guard, costing every
            # `import transformers` the package's whole import tree.
            package = importlib.import_module(pkg_name)
            package_version = getattr(package, "__version__", "N/A")
            # No version + no __file__ means a
            # namespace package (PEP 420) shadowing on sys.path,
            # not a real install.
            if (
                package_version == "N/A"
                and getattr(
                    package, "__file__", None
                ) is None
            ):
                package_exists = False
        logger.debug(f"Detected {pkg_name} version: {package_version}")

    if return_version:
        return package_exists, package_version
    else:
        return package_exists, None


def _make_compile_constant(fn):
    """Mark `fn`'s result as a trace-time constant,
    so `torch.compile` inlines it instead of tracing it.

    This is `torch._dynamo.assume_constant_result`,
    spelled without importing torch: this module is what
    decides whether torch is installed, so it must never import it
    (and doing so would pull torch into `import transformers`,
    which is deliberately torch-free).

    Apply it *under* `@lru_cache`, not above:
    dynamo steps past the cache wrapper and only reads the
    marker on the function it actually traces.

    Only for helpers whose answer is fixed for the lifetime
    of the process — an install probe, a hardware capability,
    an environment variable. Never for a runtime query such as
    `is_cuda_stream_capturing`, where inlining a value that legitimately
    changes would silently bake a transient into the graph.
    """
    setattr(fn, "_dynamo_marked_constant", True)
    return fn
