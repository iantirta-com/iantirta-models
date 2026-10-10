

from functools import lru_cache
import os
from types import ModuleType


class LazyModule(ModuleType):
    def __init__(self, name: str, module_path: str, import_structure):
        super().__init__(name)

@lru_cache
def create_import_structure_from_path(module_path: str):
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
            
    print(adjacent_modules)

@lru_cache
def define_import_structure(module_path: str, prefix: str | None = None):
    import_structure = create_import_structure_from_path(module_path)

if __name__ == "__main__":
    from rich import inspect as i
    _file = globals()["__file__"]
    i(_file)
    i(LazyModule("hello", _file, define_import_structure(_file)), all=True)