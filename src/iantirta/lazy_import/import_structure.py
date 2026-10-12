from functools import lru_cache
from pprint import pprint
import os
import ast
import time
from rich import inspect as i
from timeit import timeit
import re


# This looks for: __all__ = [ or ( ...anything... ] or )
__ALL__PATTERN_RE = re.compile(r'__all__\s*=\s*[\[\(](.*?)[\]\)]', re.DOTALL)


def fetch__all__(file_content) -> list[str]:
    """
    Returns the content of the __all__ variable in the file content.
    Returns None if not defined, otherwise returns a list of strings.
    """

    if "__all__" not in file_content:
        return []

    match = __ALL__PATTERN_RE.search(file_content)
    if not match:
        # There is no pattern matching `__all__`
        return []

    # This finds anything surrounded by single or double quotes
    return re.findall(r'["\'](.*?)["\']', match.group(1))


@lru_cache
def create_import_structure_from_path(
    module_path: str
):
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
            elif entry.name.endswith(".py"):
                adjacent_modules.append(entry.name)

    # We're only taking a look at files different from __init__.py
    # We could theoretically require things directly from the __init__.py
    # files, but this is not supported at this time.
    if "__init__.py" in adjacent_modules:
        adjacent_modules.remove("__init__.py")

    
    all_modules = {}
    for module_name in adjacent_modules:
        # Only modules ending in `.py` are accepted here.
        if not module_name.endswith(".py"):
            continue

        with open(os.path.join(module_path, module_name), encoding="utf-8") as f:
            file_content = f.read()

        # Remove the .py suffix
        module_name = module_name[:-3]

        # All objects that are in __all__ should be exported by default.
        # These objects are exported with the file backends.
        exported_objects = set()
        if "__all__" in file_content:
            for _all_object in fetch__all__(file_content):
                if _all_object not in exported_objects:
                    if module_name not in all_modules:
                        all_modules[module_name] = set()

                    all_modules[module_name].add(_all_object)

    import_structure = {**all_modules, **import_structure}
    return import_structure

@lru_cache
def define_import_structure(
    module_path: str,
    prefix: str | None = None
):
    import_structure = create_import_structure_from_path(module_path)
    i(import_structure)

def benchmark_extractors(module_path: str, iterations: int = 1000):
    """Benchmarks both functions across all Python files in the directory."""
    path = Path(module_path)
    if path.is_file():
        path = path.parent

    py_files = list(path.glob("**/*.py"))
    file_contents = []
    
    # Pre-load file contents to avoid disk I/O noise in timing
    t1, t2 = 0, 0
    _setup="from __main__ import fetch__all__, re_fetch__all__"
    params = {"setup": _setup, "number": iterations}
    for p in py_files:
        if p.name == "__init__.py":
            continue
        content = p.read_text(encoding="utf-8")
        if "__all__" in content:
            t2 += timeit("re_fetch__all__(content)", **params, globals=locals(),)
            t1 += timeit("fetch__all__(content)", **params, globals=locals(),)
            
    
    for name, content in file_contents:
        res_ori = fetch__all__(content)
        res_re = re_fetch__all__(content)
        if res_ori != res_re:
            print(f"[Warning] Mismatch in {name}: Original={res_ori} vs Regex={res_re}")

    print(f"Original Method : {t1:.4f} seconds")
    print(f"Regex Method    : {t2:.4f} seconds")
    
    if t2 < t1:
        speedup = (t1 / t2) if t2 > 0 else 0
        print(f"Winner: Regex is ~{speedup:.2f}x faster!")
    else:
        print("Winner: Original method was faster.")


if __name__ == "__main__":
    from pathlib import Path
    test_module_path = Path("./src/iantirta/models/").expanduser().resolve()
    pprint(test_module_path)
    define_import_structure(str(test_module_path))
    #benchmark_extractors(str(test_module_path), iterations=2000)
    
    # assert itd1(str(test_module_path)) == itd2(str(test_module_path))
    # t1 = timeit.timeit("itd1(str(test_module_path))", globals=globals(), number=20)
    # t2 = timeit.timeit('itd2(test_module_path)', globals=globals(), number=20)
    # #t3 = timeit.timeit("entries = []; with os.scandir(test_module_path) as d: for entry in d: if entry.name.endswith('.py'): entries.append(entry.name)", globals=globals(), number=20)
    # print(t1)
    # print(t2)
