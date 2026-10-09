


from functools import cached_property
import os
from pathlib import Path
import uuid
from dataclasses import dataclass


@dataclass
class CachedFile:

    filename: str
    cache_dir: Path = Path("~/.cache/iantirta").expanduser().resolve()
    
    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value: str | Path | None = None):
        self._cache_dir = Path(value).expanduser().resolve() or self.cache_dir
    
    @cached_property
    def locks_dir(self) -> Path:
        return self.cache_dir / ".locks"

    @cached_property
    def locks_path(self) -> Path:
        return self.locks_dir / f"{uuid.uuid4().hex[:8]}.lock"

    @cached_property
    def storage_dir(self) -> Path:
        return self.cache_dir / "Downloaded"

    @property
    def filename(self) -> str:
        return self._relative_filename

    @filename.setter
    def filename(self, value: str) -> None:
        self._relative_filename = os.path.join(*value.split("/"))
    
    @cached_property
    def pointer_path(self) -> Path:
        """Symlink pointer path"""
        return self.storage_dir / self.filename

    @staticmethod
    def as_extended_path(path: str | Path, max_length: int = 255) -> Path:
        r"""Return `path` in its Windows extended-length form if it is too long, unchanged otherwise.

        Some Windows versions do not allow for paths longer than 255 characters (247 for directories, i.e. MAX_PATH minus
        room for an 8.3 file name). In this case, we must specify them as extended paths by using the `\\?\` prefix, which
        only works on absolute paths. Network shares take the `\\?\UNC\server\share\...` form: prefixing them verbatim
        would produce an invalid `\\?\\\server\...` path.
        """
        path = Path(path)
        if os.name != "nt":
            return path
        absolute_path = str(path.absolute())
        if len(absolute_path) <= max_length or absolute_path.startswith("\\\\?\\"):
            return path
        if absolute_path.startswith("\\\\"):  # UNC share: `\\server\share\...` => `\\?\UNC\server\share\...`
            return Path("\\\\?\\UNC\\") / absolute_path[2:]
        return Path("\\\\?\\") / absolute_path

    def _cache_download(self,):
        pass
