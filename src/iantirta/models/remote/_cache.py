


import os
from pathlib import Path
import shutil
import tempfile

from .hf import repo_folder_name

CACHE_DIR = Path("~/.cache/iantirta").expanduser().resolve()
CACHEDIR_TAG_CONTENT = (
    "Signature: 8a477f597d28d172789f06886806bc55\n"
    "# This file is a cache directory tag created by huggingface_hub.\n"
    "# For information about cache directory tags, see:\n"
    "#\thttps://bford.info/cachedir/\n"
)

_symlink_support: dict[Path, bool] = {}


def create_cache_tag(cache_dir: Path) -> None:
    """Create a CACHEDIR.TAG file in ``cache_dir`` if one does not already exist.

    The tag follows the `Cache Directory Tagging Standard <http://www.brynosaurus.com/cachedir/>`_
    so that backup tools can recognize and skip cache directories.
    """
    tag_path = cache_dir / "CACHEDIR.TAG"
    if not tag_path.exists():
        try:
            tag_path.write_text(CACHEDIR_TAG_CONTENT)
        except OSError:
            pass


def supports_symlink(
    cache_dir: str | Path = CACHE_DIR,
) -> bool:
    cache_dir = Path(cache_dir).expanduser().resolve()

    if cache_dir in _symlink_support:
        return _symlink_support[cache_dir]

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            dir=cache_dir
        ) as directory:
            source = Path(directory) / "source"
            target = Path(directory) / "target"

            source.touch()

            try:
                target.symlink_to(source)
            except OSError:
                _symlink_support[cache_dir] = False
            else:
                _symlink_support[cache_dir] = True

    except ValueError:
        _symlink_support[cache_dir] = os.name != "nt"

    return _symlink_support[cache_dir]


def create_pointer(
    source: Path,
    destination: Path,
    *,
    move_source: bool = False,
) -> None:
    """Create a snapshot pointer.

    Symlinks are preferred. A normal copy is used when symlinks
    are unavailable.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        destination.unlink()
    except FileNotFoundError:
        pass

    source = source.absolute()
    destination = destination.absolute()
    if supports_symlink(destination.parent):
        try:
            relative_source = os.path.relpath(source, destination.parent,)
            os.symlink(relative_source, destination,)
            return
        except PermissionError:
            pass
        except OSError:
            pass

    if move_source:
        shutil.move(source, destination,)
    else:
        shutil.copyfile(source, destination, )


def pointer_path(
    storage_dir: Path,
    filename: str,
    revision: str,
) -> Path:
    snapshot = storage_dir / "snapshots"
    path = snapshot / revision / Path(filename)

    snapshot_abs = Path(os.path.abspath(snapshot))
    path_abs = Path(os.path.abspath(path))

    if snapshot_abs not in path_abs.parents:
        raise ValueError("Invalid snapshot path.")

    return path


def try_to_load_from_cache(
    repo_id: str,
    filename: str,
    cache_dir: str | Path = CACHE_DIR,
    revision: str = "main",
    repo_type: str = "model",
) -> str | None:
    if repo_type != "model":
        raise NotImplementedError()

    cache_dir = Path(cache_dir).expanduser().resolve()

    repo_cache = cache_dir / repo_folder_name(
        repo_id=repo_id, repo_type=repo_type
    )
    if not repo_cache.is_dir():
        # No cache for this model
        return None

    refs_dir = repo_cache / "refs"
    snapshots_dir = repo_cache / "snapshots"
    no_exist_dir = repo_cache / ".no_exist"

    # Resolve refs (for instance to convert main to the associated commit sha)
    revision_file = refs_dir / revision
    if revision_file.is_file():
        revision = revision_file.read_text().strip()

    if (no_exist_dir / revision / filename).is_file():
        return None

    if not snapshots_dir.exists():
        return None

    snapshots_dir = snapshots_dir / revision
    if not snapshots_dir.is_dir():
        return None

    if (cached_file := (snapshots_dir / filename)).is_file():
        return str(cached_file)

    return None
