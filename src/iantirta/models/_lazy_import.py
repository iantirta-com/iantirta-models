
import importlib
import importlib.machinery
import logging
import operator
import os
import re
import sys
from collections import OrderedDict
from enum import Enum
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any

from packaging import version

from .tools._deps import _is_package_available

logger = logging.getLogger(__name__)


BACKENDS_MAPPING = OrderedDict([])


BACKENDS_T = frozenset[str]
IMPORT_STRUCTURE_T = dict[BACKENDS_T, dict[str, set[str]]]


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but
    only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(
        self,
        name: str,
        module_file: str,
        import_structure: IMPORT_STRUCTURE_T,
        module_spec: importlib.machinery.ModuleSpec | None = None,
        extra_objects: dict[str, object] | None = None,
        explicit_import_shortcut: dict[str, list[str]] | None = None,
    ):
        super().__init__(name)

        self._object_missing_backend = {}
        self._explicit_import_shortcut = (
            explicit_import_shortcut
            if explicit_import_shortcut
            else {}
        )

        if any(isinstance(key, frozenset) for key in import_structure):
            self._modules = set()
            self._class_to_module = {}
            self.__all__ = []

            _import_structure = {}

            for backends, module in import_structure.items():
                missing_backends = []

                # This ensures that if a module is importable,
                # then all other keys of the module are importable.
                # As an example, in module.keys() we might have the following:
                #
                # dict_keys(['models.nllb_moe.configuration_nllb_moe',
                # 'models.sew_d.configuration_sew_d'])
                #
                # with this, we don't only want to be able to import
                # these explicitly, we want to be able to import
                # every intermediate module as well.
                # Therefore, this is what is returned:
                #
                # {
                #     'models.nllb_moe.configuration_nllb_moe',
                #     'models.sew_d.configuration_sew_d',
                #     'models',
                #     'models.sew_d', 'models.nllb_moe'
                # }

                module_keys = set(
                    chain(*[[
                        k.rsplit(".", i)[0]
                        for i in range(k.count(".") + 1)
                    ] for k in list(module.keys())])
                )

                for backend in backends:
                    if backend in BACKENDS_MAPPING:
                        callable, _ = BACKENDS_MAPPING[backend]
                    else:
                        if any(key in backend for key in ["=", "<", ">"]):
                            backend = Backend(backend)
                            callable = backend.is_satisfied
                        else:
                            raise ValueError(
                                "Backend should be defined in the "
                                f"BACKENDS_MAPPING. Offending backend: {backend}.\n"
                                f"From module_file: {module_file}.\n"
                                f"All backends: {backends!r}.\n"
                                f"Import Structure: {import_structure!r}.\n"
                            )

                    try:
                        if not callable():
                            missing_backends.append(backend)
                    except (ModuleNotFoundError, RuntimeError):
                        missing_backends.append(backend)

                self._modules = self._modules.union(module_keys)

                for key, values in module.items():
                    if missing_backends:
                        self._object_missing_backend[key] = missing_backends

                    for value in values:
                        self._class_to_module[value] = key
                        if missing_backends:
                            self._object_missing_backend[value] = missing_backends
                    _import_structure.setdefault(key, []).extend(values)

                # Needed for autocompletion in an IDE
                self.__all__.extend(module_keys | set(chain(*module.values())))

            self.__file__ = module_file
            self.__spec__ = module_spec
            self.__path__ = [os.path.dirname(module_file)]
            self._objects = {} if extra_objects is None else extra_objects
            self._name = name
            self._import_structure = _import_structure

        # This can be removed once every exportable
        # object has a `require()` require.
        else:
            self._modules = set(import_structure.keys())
            self._class_to_module = {}
            for key, values in import_structure.items():
                for value in values:
                    self._class_to_module[value] = key
            # Needed for autocompletion in an IDE
            self.__all__ = (
                list(import_structure.keys())
                + list(chain(*import_structure.values()))
            )
            self.__file__ = module_file
            self.__spec__ = module_spec
            self.__path__ = [os.path.dirname(module_file)]
            self._objects = {} if extra_objects is None else extra_objects
            self._name = name
            self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = list(super().__dir__())
        # The elements of self.__all__ that are submodules may or
        # may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the
        # elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        import_error_message = (
            f"Could not import module '{name}'. "
            "Are this object's requirements defined correctly? "
            "Set the logging verbosity to DEBUG for the original import error."
        )
        if name in self._objects:
            return self._objects[name]
        if name in self._object_missing_backend:
            missing_backends = self._object_missing_backend[name]

            # Backward-compat fallback: before the image processor refactoring, the base
            # `<Model>ImageProcessor` name referred to the PIL/slow backend. After the refactoring
            # it refers to the TorchvisionBackend (which requires torchvision). So if torchvision
            # is not installed, transparently fall back to `<Model>ImageProcessorPil` and warn once.
            if "torchvision" in missing_backends and name.endswith("ImageProcessor"):
                pil_name = f"{name}Pil"
                if pil_name in self._class_to_module and pil_name not in self._object_missing_backend:
                    try:
                        pil_module = self._get_module(self._class_to_module[pil_name])
                        pil_value = getattr(pil_module, pil_name)
                        logger.warning_once(
                            f"`{name}` requires torchvision (not installed); falling back to `{pil_name}` "
                            f"for backward compatibility. Install torchvision to use the default backend, "
                            f"or import `{pil_name}` directly to silence this warning."
                        )
                        setattr(self, name, pil_value)
                        return pil_value
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"Could not load PIL fallback {pil_name}: {e}")

            class Placeholder(metaclass=DummyObject):
                _backends = missing_backends

                def __init__(self, *args, **kwargs):
                    requires_backends(self, missing_backends)

                def call(self, *args, **kwargs):
                    pass

            Placeholder.__name__ = name

            if name not in self._class_to_module:
                module_name = f"iantirta.models.{name}"
            else:
                module_name = self._class_to_module[name]
                if not module_name.startswith("iantirta.models."):
                    module_name = f"iantirta.models.{module_name}"

            Placeholder.__module__ = module_name

            value = Placeholder
        elif name in self._class_to_module:
            try:
                module = self._get_module(self._class_to_module[name])
                value = getattr(module, name)
            except (ModuleNotFoundError, RuntimeError, AttributeError) as e:
                # V5: If trying to import a *TokenizerFast symbol, transparently fall back to the
                # non-Fast symbol from the same module when available. This lets us keep only one
                # backend tokenizer class while preserving legacy public names.
                if name.endswith("TokenizerFast"):
                    fallback_name = name[:-4]
                    # Prefer importing the module that declares the fallback symbol if known
                    try:
                        if fallback_name in self._class_to_module:
                            fb_module = self._get_module(self._class_to_module[fallback_name])
                            fallback_value = getattr(fb_module, fallback_name)
                        else:
                            module = self._get_module(self._class_to_module[name])
                            fallback_value = getattr(module, fallback_name)
                        setattr(self, fallback_name, fallback_value)
                        value = fallback_value
                    except Exception:  # noqa: BLE001
                        # If we can't find the fallback here, try converter logic as a last resort
                        # before giving up
                        value = None
                        # Try converter mapping for Fast tokenizers that don't exist
                        if value is None and name.endswith("TokenizerFast"):
                            lookup_name = name[:-4]
                            try:
                                from ..convert_slow_tokenizer import (
                                    SLOW_TO_FAST_CONVERTERS,
                                )

                                if lookup_name in SLOW_TO_FAST_CONVERTERS:
                                    converter_class = SLOW_TO_FAST_CONVERTERS[lookup_name]
                                    converter_base_name = converter_class.__name__.replace("Converter", "")
                                    preferred_tokenizer_name = f"{converter_base_name}Tokenizer"

                                    candidate_names = [preferred_tokenizer_name]
                                    for tokenizer_name, tokenizer_converter in SLOW_TO_FAST_CONVERTERS.items():
                                        if tokenizer_converter is converter_class and tokenizer_name != lookup_name:  # noqa: SIM102
                                            if tokenizer_name not in candidate_names:
                                                candidate_names.append(tokenizer_name)

                                    # Try to import the preferred candidate directly
                                    import importlib

                                    for candidate_name in candidate_names:
                                        base_tokenizer_class = None

                                        # Try to derive module path from tokenizer name (e.g., "AlbertTokenizer" -> "albert")
                                        # Remove "Tokenizer" suffix and convert to lowercase
                                        if candidate_name.endswith("Tokenizer"):
                                            model_name = candidate_name[:-10].lower()  # Remove "Tokenizer"
                                            module_path = f"iantirta.models.{model_name}.tokenization_{model_name}"
                                            try:
                                                module = importlib.import_module(module_path)
                                                base_tokenizer_class = getattr(module, candidate_name)
                                            except Exception:  # noqa: BLE001
                                                logger.debug(f"{module_path} does not have {candidate_name} defined.")

                                        # Fallback: try via _class_to_module
                                        if base_tokenizer_class is None and candidate_name in self._class_to_module:
                                            try:
                                                alias_module_name = self._class_to_module[candidate_name]
                                                alias_module = self._get_module(alias_module_name)
                                                base_tokenizer_class = getattr(alias_module, candidate_name)
                                            except Exception:  # noqa: BLE001
                                                logger.debug(
                                                    f"{alias_module_name} does not have {candidate_name} defined"
                                                )

                                        # If we still don't have base_tokenizer_class, skip this candidate
                                        if base_tokenizer_class is None:
                                            logger.debug(f"skipping candidate {candidate_name}")
                                            continue

                                        # If we got here, we have base_tokenizer_class
                                        value = base_tokenizer_class

                                        setattr(self, candidate_name, base_tokenizer_class)
                                        if lookup_name != candidate_name:
                                            setattr(self, lookup_name, value)
                                        setattr(self, name, value)
                                        break
                            except Exception as alias_error:  # noqa: BLE001
                                logger.debug(f"Could not create tokenizer alias: {alias_error}")

                        if value is None:
                            logger.debug(f"Original import error for '{name}': {e}")
                            raise ModuleNotFoundError(import_error_message) from e
                else:
                    logger.debug(f"Original import error for '{name}': {e}")
                    raise ModuleNotFoundError(import_error_message) from e

        elif name in self._modules:
            try:
                value = self._get_module(name)
            except (ModuleNotFoundError, RuntimeError) as e:
                logger.debug(f"Original import error for '{name}': {e}")
                raise ModuleNotFoundError(import_error_message) from e
        else:
            # V5: If a *TokenizerFast symbol is requested but not present in the import structure,
            # try to resolve to the corresponding non-Fast symbol's module if available.
            if name.endswith("TokenizerFast"):
                fallback_name = name[:-4]
                if fallback_name in self._class_to_module:
                    try:
                        fb_module = self._get_module(self._class_to_module[fallback_name])
                        value = getattr(fb_module, fallback_name)
                        setattr(self, fallback_name, value)
                        setattr(self, name, value)
                        return value
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"Could not load fallback {fallback_name}: {e}")
            # V5: Handle *ImageProcessorFast backward compatibility
            # Similar to TokenizerFast, but for image processors
            if name.endswith("ImageProcessorFast"):
                fallback_name = name[:-4]  # Remove "Fast"
                if fallback_name in self._class_to_module:
                    logger.warning_once(
                        f"`{name}` is deprecated. The `Fast` suffix for image processors has been removed; "
                        f"use `{fallback_name}` instead."
                    )
                    if fallback_name in self._object_missing_backend:
                        # The Fast alias has no entry in the import structure, so `requires_backends` on
                        # the real class never runs. Handle the missing backend explicitly here, otherwise
                        # `_get_module` swallows the ImportError and the caller gets an AttributeError.
                        # Do not fall through to the PIL fallback since a legacy "Fast" image processor was explicitly requested.
                        missing_backends = self._object_missing_backend[fallback_name]

                        class Placeholder(metaclass=DummyObject):
                            _backends = missing_backends

                            def __init__(self, *args, **kwargs):
                                requires_backends(self, missing_backends)

                            def call(self, *args, **kwargs):
                                pass

                        Placeholder.__name__ = fallback_name
                        module_name = self._class_to_module[fallback_name]
                        Placeholder.__module__ = (
                            module_name if module_name.startswith("iantirta.models.") else f"iantirta.models.{module_name}"
                        )
                        setattr(self, name, Placeholder)
                        return Placeholder
                    try:
                        fb_module = self._get_module(self._class_to_module[fallback_name])
                        value = getattr(fb_module, fallback_name)
                        setattr(self, fallback_name, value)
                        setattr(self, name, value)
                        return value
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"Could not load fallback {fallback_name}: {e}")
            # V5: If a tokenizer class doesn't exist, check if it should alias to another tokenizer
            # via the converter mapping (e.g., FNetTokenizer -> AlbertTokenizer via AlbertConverter)
            value = None
            if name.endswith(("Tokenizer", "TokenizerFast")):
                # Strip "Fast" suffix for converter lookup if present
                lookup_name = name[:-4] if name.endswith("TokenizerFast") else name

                try:
                    # Lazy import to avoid circular dependencies
                    from ..convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS

                    # Check if this tokenizer has a converter mapping
                    if lookup_name in SLOW_TO_FAST_CONVERTERS:
                        converter_class = SLOW_TO_FAST_CONVERTERS[lookup_name]

                        # Find which tokenizer class uses the same converter (reverse lookup)
                        # Prefer the tokenizer that matches the converter name pattern
                        # (e.g., AlbertConverter -> AlbertTokenizer)
                        converter_base_name = converter_class.__name__.replace("Converter", "")
                        preferred_tokenizer_name = f"{converter_base_name}Tokenizer"

                        # Try preferred tokenizer first
                        candidate_names = [preferred_tokenizer_name]
                        # Then try all other tokenizers with the same converter
                        for tokenizer_name, tokenizer_converter in SLOW_TO_FAST_CONVERTERS.items():
                            if tokenizer_converter is converter_class and tokenizer_name != lookup_name:  # noqa: SIM102
                                if tokenizer_name not in candidate_names:
                                    candidate_names.append(tokenizer_name)

                        # Try to import one of the candidate tokenizers
                        for candidate_name in candidate_names:
                            if candidate_name in self._class_to_module:
                                try:
                                    alias_module = self._get_module(self._class_to_module[candidate_name])
                                    base_tokenizer_class = getattr(alias_module, candidate_name)
                                    value = base_tokenizer_class

                                    # Cache both names for future imports
                                    setattr(self, candidate_name, base_tokenizer_class)
                                    if lookup_name != candidate_name:
                                        setattr(self, lookup_name, value)
                                    setattr(self, name, value)
                                    break
                                except Exception:  # noqa: BLE001, S112
                                    # If this candidate fails, try the next one
                                    continue
                            else:
                                # Candidate not in _class_to_module - might need recursive resolution
                                # Try importing it directly to trigger lazy loading
                                try:
                                    # Try to get it from iantirta.models.vendor.transformers module to trigger lazy loading
                                    transformers_module = sys.modules.get("iantirta.models")
                                    if transformers_module and hasattr(transformers_module, candidate_name):
                                        base_tokenizer_class = getattr(transformers_module, candidate_name)
                                        value = base_tokenizer_class

                                        if lookup_name != candidate_name:
                                            setattr(self, lookup_name, value)
                                        setattr(self, name, value)
                                        break
                                except Exception:  # noqa: BLE001, S112
                                    continue
                except (ImportError, AttributeError):
                    pass

            if value is None:
                for key, values in self._explicit_import_shortcut.items():
                    if name in values:
                        value = self._get_module(key)
                        break

            if value is None:
                raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            print(
                f"\n!! FAILED IMPORT"
                f"\n   package: {self.__name__}"
                f"\n   module:  {module_name}"
                f"\n   error:   {type(e).__name__}: {e}"
            )
            raise

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))


class VersionComparison(Enum):
    EQUAL = operator.eq
    NOT_EQUAL = operator.ne
    GREATER_THAN = operator.gt
    LESS_THAN = operator.lt
    GREATER_THAN_OR_EQUAL = operator.ge
    LESS_THAN_OR_EQUAL = operator.le

    @staticmethod
    def from_string(version_string: str) -> "VersionComparison":
        string_to_operator = {
            "=": VersionComparison.EQUAL,
            "==": VersionComparison.EQUAL,
            "!=": VersionComparison.NOT_EQUAL,
            ">": VersionComparison.GREATER_THAN,
            "<": VersionComparison.LESS_THAN,
            ">=": VersionComparison.GREATER_THAN_OR_EQUAL,
            "<=": VersionComparison.LESS_THAN_OR_EQUAL,
        }

        return string_to_operator[version_string]


@lru_cache
def split_package_version(package_version_str) -> tuple[str, str, str]:
    pattern = r"([a-zA-Z0-9_-]+)([!<>=~]+)([0-9.]+)"
    match = re.match(pattern, package_version_str)
    if match:
        return (match.group(1), match.group(2), match.group(3))
    else:
        raise ValueError(f"Invalid package version string: {package_version_str}")


class Backend:
    def __init__(self, backend_requirement: str):
        self.package_name, self.version_comparison, self.version = split_package_version(backend_requirement)

        if self.package_name not in BACKENDS_MAPPING:
            raise ValueError(
                f"Backends should be defined in the BACKENDS_MAPPING. Offending backend: {self.package_name}"
            )

    def get_installed_version(self) -> str:
        """Return the currently installed version of the backend"""
        is_available, current_version = _is_package_available(self.package_name, return_version=True)
        if not is_available:
            raise RuntimeError(f"Backend {self.package_name} is not available.")
        return current_version

    def is_satisfied(self) -> bool:
        return VersionComparison.from_string(self.version_comparison).value(
            version.parse(self.get_installed_version()), version.parse(self.version)
        )

    def __repr__(self) -> str:
        return f'Backend("{self.package_name}", {VersionComparison[self.version_comparison]}, "{self.version}")'

    @property
    def error_message(self):
        return (
            f"{{0}} requires the {self.package_name} library version {self.version_comparison}{self.version}. That"
            f" library was not found with this version in your environment."
        )


def fetch__all__(file_content) -> list[str]:
    """
    Returns the content of the __all__ variable in the file content.
    Returns None if not defined, otherwise returns a list of strings.
    """

    if "__all__" not in file_content:
        return []

    start_index = None
    lines = file_content.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("__all__"):
            start_index = index

    # There is no line starting with `__all__`
    if start_index is None:
        return []

    lines = lines[start_index:]

    if not lines[0].startswith("__all__"):
        raise ValueError(
            "fetch__all__ accepts a list of lines, with "
            "the first line being the __all__ variable declaration"
        )

    # __all__ is defined on a single line
    if lines[0].endswith("]"):
        return [
            obj.strip("\"' ")
            for obj in lines[0].split("=")[1].strip(" []").split(",")
        ]

    # __all__ is defined on multiple lines
    else:
        _all: list[str] = []
        for __all__line_index in range(1, len(lines)):
            if lines[__all__line_index].strip() == "]":
                return _all
            else:
                _all.append(lines[__all__line_index].strip("\"', "))

        return _all


@lru_cache
def create_import_structure_from_path(module_path):
    """
    This method takes the path to a file/a folder and returns the import structure.
    If a file is given, it will return the import structure of the parent folder.

    Import structures are designed to be digestible by `_LazyModule` objects. They are
    created from the __all__ definitions in each files as well as the `@require` decorators
    above methods and objects.

    The import structure allows explicit display of the required backends for a given object.
    These backends are specified in two ways:

    1. Through their `@require`, if they are exported with that decorator. This `@require` decorator
       accepts a `backend` tuple kwarg mentioning which backends are required to run this object.

    2. If an object is defined in a file with "default" backends, it will have, at a minimum, this
       backend specified. The default backends are defined according to the filename:

       - If a file is named like `modeling_*.py`, it will have a `torch` backend
       - If a file is named like `tokenization_*_fast.py`, it will have a `tokenizers` backend
       - If a file is named like `image_processing*_fast.py`, it will have a `torchvision` + `torch` backend

    Backends serve the purpose of displaying a clear error message to the user in case the backends are not installed.
    Should an object be imported without its required backends being in the environment, any attempt to use the
    object will raise an error mentioning which backend(s) should be added to the environment in order to use
    that object.

    Here's an example of an input import structure at the src.transformers.models level:

    {
        'albert': {
            frozenset(): {
                'configuration_albert': {'AlbertConfig'}
            },
            frozenset({'tokenizers'}): {
                'tokenization_albert_fast': {'AlbertTokenizer'}
            },
        },
        'align': {
            frozenset(): {
                'configuration_align': {'AlignConfig', 'AlignTextConfig', 'AlignVisionConfig'},
                'processing_align': {'AlignProcessor'}
            },
        },
        'altclip': {
            frozenset(): {
                'configuration_altclip': {'AltCLIPConfig', 'AltCLIPTextConfig', 'AltCLIPVisionConfig'},
                'processing_altclip': {'AltCLIPProcessor'},
            }
        }
    }
    """
    import_structure = {}

    if os.path.isfile(module_path):
        module_path = os.path.dirname(module_path)

    adjacent_modules = []

    with os.scandir(module_path) as entries:
        for entry in entries:
            if entry.name == "__pycache__":
                continue
            if entry.is_dir():
                import_structure[entry.name] = create_import_structure_from_path(entry.path)
            elif not entry.name.startswith(("convert_", "modular_")):
                adjacent_modules.append(entry.name)

    # We're only taking a look at files different from __init__.py
    # We could theoretically require things directly from the __init__.py
    # files, but this is not supported at this time.
    if "__init__.py" in adjacent_modules:
        adjacent_modules.remove("__init__.py")

    module_requirements = {}
    for module_name in adjacent_modules:
        # Only modules ending in `.py` are accepted here.
        if not module_name.endswith(".py"):
            continue

        with open(os.path.join(module_path, module_name), encoding="utf-8") as f:
            file_content = f.read()

        # Remove the .py suffix
        module_name = module_name[:-3]

        previous_line = ""
        previous_index = 0

        # Some files have some requirements by default.
        # For example, any file named `modeling_xxx.py`
        # should have torch as a required backend.
        base_requirements = ()
        # for check, requirements in BASE_FILE_REQUIREMENTS.items():
        #     if check(module_name, file_content):
        #         base_requirements = requirements
        #         break

        # Objects that have a `@require` assigned to them will get exported
        # with the backends specified in the decorator as well as the file backends.
        exported_objects = set()
        if "@requires" in file_content:
            lines = file_content.split("\n")
            for index, line in enumerate(lines):
                # This allows exporting items with other decorators. We'll take a look
                # at the line that follows at the same indentation level.
                if line.startswith((" ", "\t", "@", ")")) and not line.startswith("@requires"):
                    continue

                # Skipping line enables putting whatever we want between the
                # requires() call and the actual class/method definition.
                # This is what enables having # Copied from statements, docs, etc.
                skip_line = False

                if "@requires" in previous_line:
                    skip_line = False

                    # Backends are defined on the same line as requires
                    if "backends" in previous_line:
                        try:
                            backends_string = previous_line.split("backends=")[1].split("(")[1].split(")")[0]
                        except IndexError:
                            raise ValueError(
                                f"Couldn't parse backends for @requires decorator in file {module_name}:{previous_line}"
                            )
                        backends = tuple(sorted([b.strip("'\",") for b in backends_string.split(", ") if b]))

                    # Backends are defined in the lines following requires, for example such as:
                    # @requires(
                    #     backends=(
                    #             "sentencepiece",
                    #             "torch",
                    #     )
                    # )
                    #
                    # or
                    #
                    # @requires(
                    #     backends=(
                    #             "sentencepiece",
                    #     )
                    # )
                    elif "backends" in lines[previous_index + 1]:
                        backends = []
                        for backend_line in lines[previous_index:index]:
                            if "backends" in backend_line:
                                backend_line = backend_line.split("=")[1]
                            if '"' in backend_line or "'" in backend_line:
                                if ", " in backend_line:
                                    backends.extend(backend.strip("()\"', ") for backend in backend_line.split(", "))
                                else:
                                    backends.append(backend_line.strip("()\"', "))

                            # If the line is only a ')', then we reached the end of the backends and we break.
                            if backend_line.strip() == ")":
                                break
                        backends = tuple(backends)

                    # No backends are registered for requires
                    else:
                        backends = ()

                    backends = frozenset(backends + base_requirements)
                    if backends not in module_requirements:
                        module_requirements[backends] = {}
                    if module_name not in module_requirements[backends]:
                        module_requirements[backends][module_name] = set()

                    if not line.startswith("class") and not line.startswith("def"):
                        skip_line = True
                    else:
                        start_index = 6 if line.startswith("class") else 4
                        object_name = line[start_index:].split("(")[0].strip(":")
                        module_requirements[backends][module_name].add(object_name)
                        exported_objects.add(object_name)

                if not skip_line:
                    previous_line = line
                    previous_index = index

        # All objects that are in __all__ should be exported by default.
        # These objects are exported with the file backends.
        if "__all__" in file_content:
            for _all_object in fetch__all__(file_content):
                if _all_object not in exported_objects:
                    backends = frozenset(base_requirements)
                    if backends not in module_requirements:
                        module_requirements[backends] = {}
                    if module_name not in module_requirements[backends]:
                        module_requirements[backends][module_name] = set()

                    module_requirements[backends][module_name].add(_all_object)

    import_structure = {**module_requirements, **import_structure}
    return import_structure


def spread_import_structure(nested_import_structure):
    """
    This method takes as input an unordered import structure and brings the required backends at the top-level,
    aggregating modules and objects under their required backends.

    Here's an example of an input import structure at the src.transformers.models level:

    {
        'albert': {
            frozenset(): {
                'configuration_albert': {'AlbertConfig'}
            },
            frozenset({'tokenizers'}): {
                'tokenization_albert_fast': {'AlbertTokenizer'}
            },
        },
        'align': {
            frozenset(): {
                'configuration_align': {'AlignConfig', 'AlignTextConfig', 'AlignVisionConfig'},
                'processing_align': {'AlignProcessor'}
            },
        },
        'altclip': {
            frozenset(): {
                'configuration_altclip': {'AltCLIPConfig', 'AltCLIPTextConfig', 'AltCLIPVisionConfig'},
                'processing_altclip': {'AltCLIPProcessor'},
            }
        }
    }

    Here's an example of an output import structure at the src.transformers.models level:

    {
        frozenset({'tokenizers'}): {
            'albert.tokenization_albert_fast': {'AlbertTokenizer'}
        },
        frozenset(): {
            'albert.configuration_albert': {'AlbertConfig'},
            'align.processing_align': {'AlignProcessor'},
            'align.configuration_align': {'AlignConfig', 'AlignTextConfig', 'AlignVisionConfig'},
            'altclip.configuration_altclip': {'AltCLIPConfig', 'AltCLIPTextConfig', 'AltCLIPVisionConfig'},
            'altclip.processing_altclip': {'AltCLIPProcessor'}
        }
    }

    """

    def propagate_frozenset(unordered_import_structure):
        frozenset_first_import_structure = {}
        for _key, _value in unordered_import_structure.items():
            # If the value is not a dict but a string, no need for custom manipulation
            if not isinstance(_value, dict):
                frozenset_first_import_structure[_key] = _value

            elif any(isinstance(v, frozenset) for v in _value):
                for k, v in _value.items():
                    if isinstance(k, frozenset):
                        # Here we want to switch around _key and k to propagate k upstream if it is a frozenset
                        if k not in frozenset_first_import_structure:
                            frozenset_first_import_structure[k] = {}
                        if _key not in frozenset_first_import_structure[k]:
                            frozenset_first_import_structure[k][_key] = {}

                        frozenset_first_import_structure[k][_key].update(v)

                    else:
                        # If k is not a frozenset, it means that the dictionary is not "level": some keys (top-level)
                        # are frozensets, whereas some are not -> frozenset keys are at an unknown depth-level of the
                        # dictionary.
                        #
                        # We recursively propagate the frozenset for this specific dictionary so that the frozensets
                        # are at the top-level when we handle them.
                        propagated_frozenset = propagate_frozenset({k: v})
                        for r_k, r_v in propagated_frozenset.items():
                            if isinstance(_key, frozenset):
                                if r_k not in frozenset_first_import_structure:
                                    frozenset_first_import_structure[r_k] = {}
                                if _key not in frozenset_first_import_structure[r_k]:
                                    frozenset_first_import_structure[r_k][_key] = {}

                                # _key is a frozenset -> we switch around the r_k and _key
                                frozenset_first_import_structure[r_k][_key].update(r_v)
                            else:
                                if _key not in frozenset_first_import_structure:
                                    frozenset_first_import_structure[_key] = {}
                                if r_k not in frozenset_first_import_structure[_key]:
                                    frozenset_first_import_structure[_key][r_k] = {}

                                # _key is not a frozenset -> we keep the order of r_k and _key
                                frozenset_first_import_structure[_key][r_k].update(r_v)

            else:
                frozenset_first_import_structure[_key] = propagate_frozenset(_value)

        return frozenset_first_import_structure

    def flatten_dict(_dict, previous_key=None):
        items = []
        for _key, _value in _dict.items():
            _key = f"{previous_key}.{_key}" if previous_key is not None else _key
            if isinstance(_value, dict):
                items.extend(flatten_dict(_value, _key).items())
            else:
                items.append((_key, _value))
        return dict(items)

    # The tuples contain the necessary backends. We want these first, so we propagate them up the
    # import structure.
    ordered_import_structure = nested_import_structure

    # 6 is a number that gives us sufficient depth to go through all files and foreseeable folder depths
    # while not taking too long to parse.
    for i in range(6):
        ordered_import_structure = propagate_frozenset(ordered_import_structure)

    # We then flatten the dict so that it references a module path.
    flattened_import_structure = {}
    for key, value in ordered_import_structure.copy().items():
        if isinstance(key, str):
            del ordered_import_structure[key]
        else:
            flattened_import_structure[key] = flatten_dict(value)

    return flattened_import_structure


@lru_cache
def define_import_structure(
    module_path: str,
    prefix: str | None = None
) -> IMPORT_STRUCTURE_T:
    """
    This method takes a module_path as input and creates an import structure digestible by a _LazyModule.

    Here's an example of an output import structure at the src.transformers.models level:

    {
        frozenset({'tokenizers'}): {
            'albert.tokenization_albert_fast': {'AlbertTokenizer'}
        },
        frozenset(): {
            'albert.configuration_albert': {'AlbertConfig'},
            'align.processing_align': {'AlignProcessor'},
            'align.configuration_align': {'AlignConfig', 'AlignTextConfig', 'AlignVisionConfig'},
            'altclip.configuration_altclip': {'AltCLIPConfig', 'AltCLIPTextConfig', 'AltCLIPVisionConfig'},
            'altclip.processing_altclip': {'AltCLIPProcessor'}
        }
    }

    The import structure is a dict defined with frozensets as keys, and dicts of strings to sets of objects.

    If `prefix` is not None, it will add that prefix to all keys in the returned dict.
    """
    import_structure = create_import_structure_from_path(module_path)
    spread_dict = spread_import_structure(import_structure)

    if prefix is None:
        return spread_dict
    else:
        spread_dict = {
            k: {
                f"{prefix}.{kk}": vv
                for kk, vv in v.items()
            }
            for k, v in spread_dict.items()
        }
        return spread_dict
