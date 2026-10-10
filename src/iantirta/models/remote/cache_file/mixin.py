


from contextlib import contextmanager
import errno
import os
import stat
import tempfile
import uuid
from dataclasses import InitVar, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar
import shutil
import logging
import warnings
from . import cache_constant

logger = logging.getLogger(__name__)


class CachedFile:
    """ Can be a direct cache dir storage or a filename cache path
    """
    _cache_dir: Path = cache_constant.DEFAULT_CACHE_DIR
    _cached_files: dict[str, Any] = {}

    _symlink_supported_dir: dict[str, Any] = {}

    def __init__(self, *, filename: str | None = None, cache_dir: str | Path | None = None) -> None:
        
        self.filename = filename
        self.cache_dir = cache_dir
        self.register()

    def register(self):
        self._cached_files[self.make_cache_key(self.cache_dir)] = self

    @classmethod
    def make_cache_key(cls, cache_dir: str | Path | None) -> tuple:
        if not cache_dir:
            cache_dir = cls._cache_dir
        cache_dir = str(cache_dir)
        return (cache_dir,)

    @classmethod
    def get_cached_file(cls, *, cache_dir: str | Path | None, filename: str | None) -> "CachedFile":
        cache_key = cls.make_cache_key(cache_dir=cache_dir)
        cache_file = cls._cached_files.get(cache_key, None)
        
        if not cache_file:
            cache_file = cls(filename=filename, cache_dir=cache_dir)
        
        if filename is not None:
            cache_file.filename = filename
        
        return cache_file

    @staticmethod
    def create_cache_dir_tag(cache_dir: str | Path) -> bool:
        """Create a CACHEDIR.TAG file in ``cache_dir`` if one does not already exist.

        The tag follows the `Cache Directory Tagging Standard <http://www.brynosaurus.com/cachedir/>`_
        so that backup tools can recognize and skip cache directories.
        """
        cache_dir = Path(cache_dir)
        if not (tag_path := cache_dir / "CACHEDIR.TAG").exists():
            try:
                tag_path.write_text(cache_constant.CACHEDIR_TAG_CONTENT)
            except OSError:
                pass

    @property
    def cache_dir(self) -> Path:
        return Path(self._cache_dir).expanduser().resolve()

    @cache_dir.setter
    def cache_dir(self, value: str | Path | None):
        self._cache_dir = Path(value).expanduser().resolve() if value is not None else self._cache_dir
        self.create_cache_dir_tag(self.cache_dir)

    @property
    def filename(self) -> str:
        return self._relative_filename

    @filename.setter
    def filename(self, value: str | None = None):
        self._relative_filename = str(Path(value)) if value is not None else None

    @property
    def locks_dir(self) -> Path:
        return self.cache_dir / ".locks"

    @property
    def lock_path(self) -> Path:
        return self.as_extended_path(self.locks_dir / f"{uuid.uuid4().hex[:8]}.lock")

    @property
    def storage_dir(self) -> Path:
        return self.cache_dir / "Downloaded"

    @property
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

    @staticmethod
    def as_striped_path(path: str | Path) -> Path:
        """Return an absolute path without the Windows extended-length prefix."""
        path_str = os.fspath(path)
        if path_str[:8].lower() == "\\\\?\\unc\\":
            path_str = f"\\\\{path_str[8:]}"
        elif path_str.startswith("\\\\?\\"):
            path_str = path_str[4:]
        return Path(os.path.abspath(path_str))

    @staticmethod
    def force_copy(src: str | Path, dst: str | Path) -> None:
        """Copy file from src to dst.

        If `shutil.copy2` fails, fallback to `shutil.copyfile`.
        """
        src = str(src)
        dst = str(dst)
        try:
            # Copy file with metadata and permission
            # Can fail e.g. if dst is an S3 mount
            shutil.copy2(src, dst)
        except OSError:
            # Copy only file content
            shutil.copyfile(src, dst)

    @staticmethod
    def force_move(src: str | Path, dst: str | Path) -> None:
        """Replace `dst` with `src`.

        Some mounts reject replace-over-existing: stage the new file next to `dst`, move the old entry aside
        and restore it if the final move fails.
        """
        src = Path(src)
        dst = Path(dst)
        try:
            os.replace(str(src), str(dst))
        except OSError:
            staged_dst = dst.with_name(f".{dst.name}.{uuid.uuid4().hex[:8]}.new")
            backup_dst = dst.with_name(f".{dst.name}.{uuid.uuid4().hex[:8]}.old")
            backup_holds_previous_entry = False
            try:
                shutil.move(str(src), str(staged_dst), copy_function=CachedFile.force_copy)
                os.rename(str(dst), str(backup_dst))
                backup_holds_previous_entry = True
                try:
                    shutil.move(str(staged_dst), str(dst), copy_function=CachedFile.force_copy)
                except OSError:
                    try:
                        if os.path.lexists(str(dst)):
                            os.unlink(str(dst))
                        os.rename(str(backup_dst), str(dst))
                        backup_holds_previous_entry = False
                    except OSError as restore_error:
                        raise OSError(
                            f"Could not restore previous destination '{dst}' from '{backup_dst}'"
                        ) from restore_error
                    raise
                try:
                    backup_dst.unlink()
                    backup_holds_previous_entry = False
                except OSError as cleanup_error:
                    logger.warning(f"Could not remove previous destination backup '{backup_dst}': {cleanup_error}")
            finally:
                staged_dst.unlink(missing_ok=True)
                if not backup_holds_previous_entry:
                    backup_dst.unlink(missing_ok=True)

    @staticmethod
    def chmod_and_move(src: str | Path, dst: str | Path) -> None:
        """Set correct permission before moving a blob from tmp directory to cache dir.

        Do not take into account the `umask` from the process as there is no convenient way
        to get it that is thread-safe.
        """
        src = Path(src)
        dst = Path(dst)
        tmp_file = src.parent / f"tmp_{uuid.uuid4()}"
        try:
            tmp_file.touch()
            cache_dir_mode = Path(tmp_file).stat().st_mode
            os.chmod(str(src), stat.S_IMODE(cache_dir_mode))
        except OSError as e:
            logger.warning(
                f"Could not set the permissions on the file '{src}'. Error: {e}.\nContinuing without setting permissions."
            )
        finally:
            try:
                tmp_file.unlink()
            except OSError:
                # fails if `tmp_file.touch()` failed => do nothing
                # See https://github.com/huggingface/huggingface_hub/issues/2359
                pass

        if os.path.lexists(str(dst)):
            # Replace the entry so a force download never writes through a shared symlink.
            CachedFile.force_move(src, dst)
        else:
            shutil.move(str(src), str(dst), copy_function=CachedFile.force_copy)

    @staticmethod
    def support_symlink(folder: str | Path) -> bool:
        folder = Path(folder)
        if str(folder) not in CachedFile._symlink_supported_dir:
            folder.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=str(folder)) as tmpdir:
                src = Path(tmpdir) / "dummy_file_src"
                dst = Path(tmpdir) / "dummy_file_dst"
                src.touch()

                relative_src = os.path.relpath(str(src), start=os.path.dirname(str(dst)))
                try:
                    os.symlink(relative_src, str(dst))
                    CachedFile._symlink_supported_dir[str(folder)] = True
                except OSError:
                    CachedFile._symlink_supported_dir[str(folder)] = False

        return CachedFile._symlink_supported_dir[str(folder)]

    @staticmethod
    def create_symlink(src: str | Path, dst: str | Path, *, new_blob: bool = False) -> None:
        """Create a symbolic link named dst pointing to src."""
        src = Path(src).expanduser().absolute()
        dst = Path(dst).expanduser().absolute()
        try:
            dst.unlink()
        except OSError:
            pass

        try:
            relative_src = os.path.relpath(str(src), str(dst.parent))
        except ValueError:
            relative_src = None

        try:
            commonpath = os.path.commonpath([str(src), str(dst)])
            _support_symlink = CachedFile.support_symlink(commonpath)
        except ValueError:
            _support_symlink = os.name != "nt"
        except PermissionError:
            _support_symlink = CachedFile.support_symlink(dst.parent)
        except OSError as e:
            # OS error (errno=30) means that the commonpath is readonly on Linux/MacOS.
            if e.errno == errno.EROFS:
                _support_symlink = CachedFile.support_symlink(dst.parent)
            else:
                raise

        if _support_symlink:
            src_rel_or_abs = relative_src or str(src)
            logger.debug(f"Creating pointer from {src_rel_or_abs} to {dst}")
            try:
                os.symlink(str(src_rel_or_abs), str(dst))
                return
            except FileExistsError:
                if dst.is_symlink():
                    # `abs_dst` already exists and is a symlink to the `abs_src` blob. It is most likely that the file has
                    # been cached twice concurrently (exactly between `os.remove` and `os.symlink`). Do nothing.
                    return
                else:
                    # Very unlikely to happen. Means a file `dst` has been created exactly between `os.remove` and
                    # `os.symlink` and is not a symlink to the `abs_src` blob file. Raise exception.
                    raise
            except PermissionError:
                # Permission error means src and dst are not in the same volume (e.g. download to local dir) and symlink
                # is supported on both volumes but not between them. Let's just make a hard copy in that case.
                pass

        if new_blob:
            logger.debug(f"Symlink not supported. Moving file from {src} to {dst}")
            shutil.move(src, dst, copy_function=CachedFile.force_copy)
        else:
            logger.debug(f"Symlink not supported. Copying file from {src} to {dst}")
            shutil.copyfile(src, dst)

    @staticmethod
    def is_regular_file(path: str | Path) -> bool:
        # checks if path is a regular file without following symlinks
        path = Path(path)
        try:
            return stat.S_ISREG(path.lstat().st_mode)
        except OSError:
            return False

    @staticmethod
    def is_dir(path: str | Path) -> bool:
        # checks if path is a regular directory without following symlinks
        path = Path(path)
        try:
            return stat.S_ISDIR(path.lstat().st_mode)
        except OSError:
            return False

    @staticmethod
    def _shared_blob_mode(cache_dir: str | Path) -> int:
        """Return a read-only mode accessible to users who can traverse the cache root."""
        if os.name == "nt":
            # Windows uses ACLs; 0444 would set the read-only attribute and block replacement and GC.
            return 0o666
        try:
            cache_dir = Path(cache_dir)
            cache_mode = stat.S_IMODE(cache_dir.stat().st_mode)
        except OSError:
            return 0o400
        return 0o400 | (0o040 if cache_mode & stat.S_IXGRP else 0) | (0o004 if cache_mode & stat.S_IXOTH else 0)

    @staticmethod
    def _shared_directory_mode(cache_dir: str | Path) -> int:
        """Mirror cache-root access and inheritance bits on newly created store directories."""
        try:
            cache_mode = stat.S_IMODE(Path(cache_dir).stat().st_mode)
        except OSError:
            return 0o700
        return cache_mode & (0o777 | stat.S_ISGID | stat.S_ISVTX)

    @staticmethod
    def _repair_shared_directory_mode(path: str | Path, cache_dir: str | Path) -> None:
        path = Path(path)
        expected_mode = CachedFile._shared_directory_mode(cache_dir)
        if stat.S_IMODE(path.lstat().st_mode) != expected_mode:
            path.chmod(expected_mode)

    @staticmethod
    def check_disk_space(expected_size: int, target_dir: str | Path) -> None:
        target_dir = Path(target_dir)
        for path in [target_dir] + list(target_dir.parents):  # first check target_dir, then each parents one by one
            try:
                target_dir_free = shutil.disk_usage(path).free
                if target_dir_free < expected_size:
                    warnings.warn(
                        "Not enough free disk space to download the file. "
                        f"The expected file size is: {expected_size / 1e6:.2f} MB. "
                        f"The target location {target_dir} only has {target_dir_free / 1e6:.2f} MB free disk space."
                    )
                return
            except OSError as e:  # raise on anything: file does not exist or space disk cannot be checked
                logger.warning(f"Cannot check disk space, Error: {e}")


if __name__ == "__main__":
    from rich import inspect
    file1 = CachedFile(filename=".")
    file2 = CachedFile()
    file3 = CachedFile(cache_dir="~/123")
    file4 = CachedFile(cache_dir="~/123", filename="hello.py")
    inspect(file2)