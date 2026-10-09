
import copy
import logging
import os
import secrets
import shutil
import stat
import uuid
from functools import cached_property, lru_cache
from pathlib import Path
from urllib.parse import quote

import requests

from ..cache_file import CachedFile
from ..http import HTTPHeader, HTTPMixin
from . import _xet, errors, hf_constant
from .types import (
    _BUCKET_ID_FROM_URL_RE,
    _COMMIT_HASH_RE,
    _JOB_ID_FROM_URL_RE,
    _MARKER_TMP_RE,
    _REPO_DIR_RE,
    _REPO_ID_FROM_RESOLVE_URL_RE,
    _REPO_ID_FROM_URL_RE,
    _REPO_URL_SUBPATHS,
    BUCKET_API_RE,
    REPO_API_RE,
)

logger = logging.getLogger(__name__)


class HFHTTPApi(HTTPMixin):
    endpoint = "https://huggingface.co"

    _WARNED_TOPICS = set()  # noqa: RUF012

    def _warn_on_warning_headers(self, headers: dict[str, str]) -> None:
        server_warnings = []
        for k, v in headers.items():
            if k.lower() == "x-hf-warning":
                server_warnings.append(v)
        
        for warning in server_warnings:
            topic, message = warning.split(";", 1) if ";" in warning else ("", warning)
            topic = topic.strip()
            if topic not in self._WARNED_TOPICS:
                message = message.strip()
                if message:
                    self._WARNED_TOPICS.add(topic)
                    self.logger.warning(message)

    def _format_exc(self, error_type: errors.HFHubHTTPError, custom_message: str, response: requests.Response, **kwargs):
        server_errors = []
        
        if from_headers := response.headers.get("X-Error-Message"):
            server_errors.append(from_headers)

        try:
            data = response.json()
            error = data.get("error")
            error_description = data.get("error_description")
            if error is not None:
                if isinstance(error, list):
                    # Case {'error': ['my error 1', 'my error 2']}
                    server_errors.extend(error)
                elif error_description is not None:
                    # OAuth-style case {'error': 'invalid_grant', 'error_description': 'my description'}
                    server_errors.append(f"{error}: {error_description}")
                else:
                    # Case {'error': 'my error'}
                    server_errors.append(error)
            elif error_description is not None:
                # Case {'error_description': 'my description'} (no 'error' field)
                server_errors.append(error_description)

            errors = data.get("errors")
            if errors is not None:
                # Case {'errors': [{'message': 'my error 1'}, {'message': 'my error 2'}]}
                for error in errors:
                    if "message" in error:
                        server_errors.append(error["message"])

        except requests.exceptions.JSONDecodeError:
            content_type = response.headers.get("Content-Type", "")
            if response.text and "html" not in content_type.lower():
                server_errors.append(response.text)

        # Strip all server messages
        server_errors = [str(line).strip() for line in server_errors if str(line).strip()]

        # Deduplicate server messages (keep order)
        # taken from https://stackoverflow.com/a/17016257
        server_errors = list(dict.fromkeys(server_errors))

        # Format server error
        server_message = "\n".join(server_errors)

        # Add server error to custom message
        final_error_message = custom_message
        if server_message and server_message.lower() not in custom_message.lower():
            if "\n\n" in custom_message:
                final_error_message += "\n" + server_message
            else:
                final_error_message += "\n\n" + server_message

        # Prepare Request ID message
        request_id = ""
        request_id_message = ""
        for header, label in (
            ("x-request-id", "Request ID"),
            ("X-Amzn-Trace-Id", "Amzn Trace ID"),
            ("x-amz-cf-id", "Amz CF ID"),
        ):
            value = response.headers.get(header)
            if value:
                request_id = str(value)
                request_id_message = f" ({label}: {value})"
                break
    
        # Add Request ID
        if request_id and request_id.lower() not in final_error_message.lower():
            if "\n" in final_error_message:
                newline_index = final_error_message.index("\n")
                final_error_message = (
                    final_error_message[:newline_index] + request_id_message + final_error_message[newline_index:]
                )
            else:
                final_error_message += request_id_message
    
        # Return
        err = error_type(final_error_message.strip(), response=response, server_message=server_message or None)
        for k, v in kwargs.items():
            setattr(err, k, v)
        return err

    def _parse_repo_info_from_url(self, url: str)  -> tuple[str | None, str | None]:
        if match := _REPO_ID_FROM_RESOLVE_URL_RE.search(url):
            return hf_constant.REPO_TYPES_MAPPING[match.group(1) or "models"], match.group(2)
        match = _REPO_ID_FROM_URL_RE.search(url)
        if not match:
            return None, None
        repo_type = hf_constant.REPO_TYPES_MAPPING.get(match.group(1))
        first, second = match.group(2), match.group(3)
        if second and second not in _REPO_URL_SUBPATHS:
            repo_id = f"{first}/{second}"
        else:
            repo_id = first
        return repo_type, repo_id
    
    def _parse_bucket_id_from_url(self, url: str) -> str | None:
        """Extract bucket_id (namespace/name) from a bucket API URL."""
        match = _BUCKET_ID_FROM_URL_RE.search(url)
        return match.group(1) if match else None

    def _parse_job_id_from_url(self, url: str) -> str | None:
        """Extract the job_id from a (scheduled) job API URL, if present."""
        match = _JOB_ID_FROM_URL_RE.search(url)
        return match.group(1) if match else None

    def raise_for_status(self, res: requests.Response, endpoint_name: str | None = None) -> None:
        try:
            self._warn_on_warning_headers(res.headers)
        except Exception as e:
            self.logger.debug(f"Failed to parse warning headers: {e}", exc_info=True)

        try:
            res.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if res.status_code // 100 == 3:
                return # Redirects

            error_code = res.headers.get("X-Error-Code")
            error_message = res.headers.get("X-Error-Message")

            request_url = (
                str(res.request.url)
                if res.request is not None
                and res.request.url is not None
                else None
            )
            repo_type, repo_id = self._parse_repo_info_from_url(request_url) if request_url else (None, None)

            if error_code == "RevisionNotFound":
                raise self._format_exc(
                    errors.RevisionNotFoundError,
                    f"{res.status_code} Client Error.\n\nRevision Not Found for url: {res.url}",
                    res,
                    repo_type=repo_type,
                    repo_id=repo_id,
                ) from e
            elif error_code == "EntryNotFound":
                raise self._format_exc(
                    errors.RemoteEntryNotFoundError,
                    f"{res.status_code} Client Error.\nEntry Not Found for url: {res.url}",
                    res,
                    repo_type=repo_type,
                    repo_id=repo_id,
                ) from e
            elif error_code == "GatedRepo":
                raise self._format_exc(
                    errors.GatedRepoError,
                    f"{res.status_code} Client Error.\nCannot access gated repo for url: {res.url}",
                    res,
                    repo_type=repo_type,
                    repo_id=repo_id,
                ) from e
            elif error_message == "Access to this resource is disabled.":
                raise self._format_exc(
                    errors.DisabledRepoError,
                    f"{res.status_code} Client Error.\nCannot access repository for url: {res.url}.\nAccess to this resource is disabled.",
                    res,
                ) from e
            elif (
                error_code == "RepoNotFound"
                and request_url is not None
                and BUCKET_API_RE.search(request_url) is not None
            ):
                raise self._format_exc(
                    errors.BucketNotFoundError,
                    f"{res.status_code} Client Error.\nBucket Not Found for url: {res.url}.\nPlease make sure you specified the correct bucket id (namespace/name).\nIf the bucket is private, make sure you are authenticated and your token has the required permissions.",
                    res,
                    bucket_id=self._parse_bucket_id_from_url(request_url)
                ) from e
            elif (
                res.status_code == 404
                and request_url is not None
                and (job_id := self._parse_job_id_from_url(request_url)) is not None
            ):
                raise self._format_exc(
                    errors.JobNotFoundError,
                    f"{res.status_code} Client Error.\nJob Not Found for url: {res.url}.\nPlease make sure you specified the correct job ID and namespace.",
                    res,
                    job_id=job_id
                ) from e
            elif error_code == "RepoNotFound" or (
                res.status_code == 401
                and error_message != "Invalid credentials in Authorization header"
                and request_url is not None
                and REPO_API_RE.search(request_url) is not None
            ):
                # 401 is misleading as it is returned for:
                #    - private and gated repos if user is not authenticated
                #    - missing repos
                # => for now, we process them as `RepoNotFound` anyway.
                # See https://gist.github.com/Wauplin/46c27ad266b15998ce56a6603796f0b9
                raise self._format_exc(
                    errors.RepositoryNotFoundError,
                    f"{res.status_code} Client Error.\nRepository Not Found for url: {res.url}.\nPlease make sure you specified the correct `repo_id` and `repo_type`.\nIf you are trying to access a private or gated repo, make sure you are authenticated and your token has the required permissions.",
                    res,
                    repo_type=repo_type,
                    repo_id=repo_id,
                ) from e
            elif res.status_code == 400:
                message = (
                    f"\n\nBad request for {endpoint_name} endpoint:" if endpoint_name is not None else "\n\nBad request:"
                )
                raise self._format_exc(
                    errors.HFHubHTTPError,
                    message,
                    res,
                ) from e
            elif res.status_code == 403:
                raise self._format_exc(
                    errors.HFHubHTTPError,
                    f"\n\n{res.status_code} Forbidden: {error_message}.\nCannot access content at: {res.url}.\nMake sure your token has the correct permissions.",
                    res,
                ) from e
            elif res.status_code == 429:
                header_metadata = HTTPHeader(res.headers)
                ratelimit_info = header_metadata.ratelimit_info
                if ratelimit_info is not None and ratelimit_info.remaining == 0:
                    message = f"\n\n429 Too Many Requests: you have reached your '{ratelimit_info.resource_type}' rate limit."
                    message += f"\nRetry after {ratelimit_info.reset_in_seconds} seconds"
                    if ratelimit_info.limit is not None and ratelimit_info.window_seconds is not None:
                        message += (
                            f" ({ratelimit_info.remaining}/{ratelimit_info.limit} requests remaining"
                            f" in current {ratelimit_info.window_seconds}s window)."
                        )
                    else:
                        message += "."
                    message += f"\nUrl: {res.url}."
                else:
                    message = f"\n\n429 Too Many Requests for url: {res.url}."
                raise self._format_exc(
                    errors.HFHubHTTPError,
                    message,
                    res,
                ) from e
            elif res.status_code == 416:
                range_header = res.request.headers.get("Range")
                raise self._format_exc(
                    errors.HFHubHTTPError,
                    f"{e} Requested range: {range_header}. Content-Range: {res.headers.get('Content-Range')}.",
                    res,
                ) from e

            # Convert `HTTPError` into a `HfHubHTTPError` to display request information
            # as well (request id and/or server error message)
            raise self._format_exc(
                errors.HFHubHTTPError,
                str(e),
                res,
            ) from e

    def hf_hub_url(
        self,
        repo_id: str,
        filename: str,
        *,
        repo_type: str,
        revision: str = "main",
    ) -> str:
        if not revision:
            revision = "main"
        
        if repo_type in hf_constant.REPO_TYPES_URL_PREFIXES:
            repo_id = hf_constant.REPO_TYPES_URL_PREFIXES[repo_type] + repo_id

        return f"{repo_id}/resolve/{quote(revision, safe='')}/{quote(filename)}"

class HFCachedFile(CachedFile):
    
    _cache_dir = CachedFile._cache_dir / "hf_hub"
    
    def __init__(
        self,
        *,
        repo_id: str,
        repo_type: str,
        revision: str | None = None,
        etag: str | None = None,
        **kwargs
    ) -> None:
        self._revision: str | None = None

        self.repo_id = repo_id
        self.repo_type = repo_type
        self.revision = revision
        self.etag = etag
        super().__init__(**kwargs)

    def register(self):
        self._cached_files[self.make_cache_key(repo_id=self.repo_id, repo_type=self.repo_type, cache_dir=self.cache_dir)] = self
    
    @classmethod
    def make_cache_key(cls, repo_id: str, repo_type: str, **kwargs) -> tuple:
        key = super().make_cache_key(**kwargs)
        return key + (repo_id, repo_type,)

    @classmethod
    def get_cached_file(
        cls,
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
        etag: str | None = None,
        cache_dir: str | Path | None = None,
        filename: str | None = None,
    ) -> "HFCachedFile":
        cache_key = cls.make_cache_key(repo_id=repo_id, repo_type=repo_type, cache_dir=cache_dir)
        cache_file = cls._cached_files.get(cache_key, None)

        if not cache_file:
            cache_file = cls(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=filename,
                cache_dir=cache_dir,
                revision=revision,
                etag=etag,
            )
        else:
            # Copy it so that its not using the original object
            cache_file = copy.deepcopy(cache_file)
        
        if filename is not None:
            cache_file.filename = filename

        if revision is not None:
            cache_file.revision = revision

        if etag is not None:
            cache_file.etag = etag
        
        return cache_file

    @staticmethod
    @lru_cache(maxsize=128)
    def _format_repo_folder_name(repo_id: str, repo_type: str) -> str:
        return "--".join([f"{repo_type}s", *repo_id.split("/")])

    @cached_property
    def repo_folder_name(self) -> str:
        return self._format_repo_folder_name(self.repo_id, self.repo_type)

    @property
    def revision_is_commit(self):
        if not self.revision:
            raise OSError(f"Revision is required to know its resolved but get: {self.revision}")
        return _COMMIT_HASH_RE.fullmatch(self.revision) is not None

    @property
    def revision(self) -> str | None:
        return self._revision

    @revision.setter
    def revision(self, commit_hash: str | None) -> None:
        if self._revision and self._revision != commit_hash and (
            not self.ref_path.exists() or commit_hash != self.ref_path.read_text()
        ):
            self.ref_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.ref_path.with_name(f"{self.ref_path.name}.{uuid.uuid4().hex[:8]}.tmp")
            tmp_path.write_text(commit_hash)
            os.replace(tmp_path, self.ref_path)
        self._revision = commit_hash

    @property
    def ref_path(self) -> Path:
        if not self.revision:
            raise OSError(f"Revision is required to get ref_path but get: {self.revision}")
        return self.storage_dir / "refs" / self.revision

    @property
    def lock_path(self) -> Path:
        if not self.etag:
            raise OSError(f"Etag is required to create the lock_path but got: {self.etag}")
        return self.as_extended_path(self.cache_dir / self.repo_folder_name / f"{self.etag}.lock")

    @property
    def storage_dir(self) -> Path:
        return self.cache_dir / self.repo_folder_name

    @property
    def snapshot_dir(self) -> Path:
        return self.storage_dir / "snapshots"

    @property
    def pointer_path(self) -> Path:
        """Symlink pointer path"""
        if not self.revision:
            raise OSError(f"Revision is required to get pointer_path but get: {self.revision}")
        if not self.filename:
            raise OSError(f"Filename is required to get pointer_path but get: {self.filename}")
        # [WARNING !!!] Don't use resolve on Symlink
        pointer_path = self.snapshot_dir / self.revision / self.filename
        if self.snapshot_dir.absolute() not in pointer_path.absolute().parents:
            raise ValueError(
                "Invalid pointer path: cannot create pointer path in snapshot folder if"
                f" `storage_folder='{self.storage_dir}'`, `revision='{self.revision}'` and"
                f" `relative_filename='{self.filename}'`."
            )
        return

    @property
    def blob_path(self) -> Path:
        if not self.etag:
            raise OSError(f"Etag is required to create the lock_path but got: {self.etag}")
        return self.as_extended_path(self.storage_dir / "blobs" / self.etag)

    @property
    def no_exist_file_path(self) -> Path:
        if not self.revision:
            raise OSError(f"Revision is needed to get no exist file path but got: {self.revision}")
        if not self.filename:
            raise OSError(f"Filename is needed to get no exist file path but got: {self.filename}")
        return self.storage_dir / ".no_exist" / self.revision / self.filename

    def cache_no_exists(self):
        """Will only cache on non-existant file from server."""
        if not self.no_exist_file_path.exists():
            try:
                self.no_exist_file_path.parent.mkdir(parents=True, exist_ok=True)
                self.no_exist_file_path.touch()
            except OSError as e:
                logger.error(
                    f"Could not cache non-existence of file. Will ignore error and continue. Error: {e}"
                )

    # Shared Store Xet
    @property
    def shared_blob_dir(self) -> Path:
        return self.cache_dir / hf_constant.SHARED_BLOBS_DIR_NAME

    @property
    def shared_blob_path(self) -> Path:
        if not self.is_xet_hash_valid:
            raise ValueError(f"Invalid Xet file hash: '{self.xet_hash}'.")
        return self.shared_blob_dir / self.xet_hash[:2] / self.xet_hash
    
    @property
    def is_xet_hash_valid(self) -> bool:
        if self.xet_hash is None:
            return False
        return _xet._XET_HASH_RE.fullmatch(self.xet_hash) is not None
    
    @property
    def xet_hash(self) -> str:
        return self._xet_hash

    @xet_hash.setter
    def xet_hash(self, value: str | None):
        self._xet_hash = value

    @property
    def relative_blob_path(self) -> str | None:
        blob_path = self.as_striped_path(self.blob_path)
        cache_dir = self.as_striped_path(self.cache_dir)
        try:
            relative_path = blob_path.relative_to(cache_dir)
        except ValueError:
            return None
        if (len(relative_path.parts) != 3
            or relative_path.parts[1] != "blobs"
            or not relative_path.parts[2]
            or _REPO_DIR_RE.fullmatch(relative_path.parts[0]) is None
        ):
            return None
        relative_str = relative_path.as_posix()
        return None if "\n" in relative_str or "\r" in relative_str else relative_str

    def get_shared_blob_prefix_dir(self) -> Path | None:
        if not self._ensure_shared_blobs_dir():
            return None
        prefix_dir = self.shared_blob_path.parent
        try:
            prefix_dir.mkdir(exist_ok=True)
        except OSError as e:
            logger.debug(f"Could not create shared blob prefix directory '{prefix_dir}': {e}")
            return None
        if not self.is_dir(prefix_dir):
            logger.debug(f"Refusing to use non-directory shared blob prefix '{prefix_dir}'.")
            return None
        try:
            self._repair_shared_directory_mode(prefix_dir, self.cache_dir)
        except OSError as e:
            logger.debug(f"Could not set shared blob prefix permissions on '{prefix_dir}': {e}")
            return None
        return prefix_dir

    @staticmethod
    def is_shared_blob_dir(path: str | Path) -> bool:
        path = Path(path)
        marker_path = path / hf_constant.SHARED_BLOBS_MARKER_NAME
        if not CachedFile.is_dir(path) or not CachedFile.is_regular_file(marker_path):
            return False
        try:
            return marker_path.read_text() == f"{hf_constant.SHARED_BLOBS_LAYOUT_VERSION}"
        except OSError:
            return False

    def _cleanup_abandoned_marker_temps(self) -> bool:
        """Remove leftover marker temporaries.

        Returns whether the directory is empty afterwards, i.e. safe to mark. Any foreign content returns False.
        """
        expected_content = f"{hf_constant.SHARED_BLOBS_LAYOUT_VERSION}\n"
        entries = list(self.shared_blob_dir.iterdir())
        for entry in entries:
            if _MARKER_TMP_RE.fullmatch(entry.name) is None or not self.is_regular_file(entry):
                return False
            try:
                if not expected_content.startswith(entry.read_text()):
                    return False
            except (OSError, UnicodeError):
                return False
        for entry in entries:
            entry.unlink(missing_ok=True)
        return not any(self.shared_blob_dir.iterdir())

    def _ensure_shared_blobs_dir(self) -> bool:
        """Create and mark the shared store, refusing to adopt an unmarked directory."""
        marker_path = self.shared_blob_dir / hf_constant.SHARED_BLOBS_MARKER_NAME
        try:
            self.shared_blob_dir.mkdir(exist_ok=True)
            if self.is_shared_blob_dir(self.shared_blob_dir):
                return True
            if not self.is_dir(self.shared_blob_dir) or not self._cleanup_abandoned_marker_temps():
                logger.debug(f"Refusing to use unmarked shared blob directory '{self.shared_blob_dir}'.")
                return False
            self._repair_shared_directory_mode(self.shared_blob_dir, self.cache_dir)

            tmp_marker = marker_path.with_name(f"{marker_path.name}.{secrets.token_hex(4)}.tmp")
            try:
                tmp_marker.write_text(f"{hf_constant.SHARED_BLOBS_LAYOUT_VERSION}\n")
                tmp_marker.chmod(self._shared_blob_mode(self.cache_dir))
                os.replace(tmp_marker, marker_path)
            finally:
                tmp_marker.unlink(missing_ok=True)
            return self.is_shared_blob_dir(self.shared_blob_dir)
        except OSError as e:
            logger.debug(f"Could not initialize shared blob directory '{self.shared_blob_dir}': {e}")
            return False

    def is_shared_blob_usable(self, expected_size: int | None) -> bool:
        if expected_size is None:
            return False
        try:
            store_stat = self.shared_blob_path.lstat()
        except OSError:
            return False
        if not stat.S_ISREG(store_stat.st_mode):
            return False
        if store_stat.st_size != expected_size:
            logger.warning(
                f"Shared blob '{self.shared_blob_path}' has an unexpected size "
                f"({store_stat.st_size} instead of {expected_size}). Not using it."
            )
            return False
        if not os.access(self.shared_blob_path, os.R_OK):
            logger.warning(f"Shared blob '{self.shared_blob_path}' is not readable. Not using it.")
            return False
        return True

    def get_shared_blob_temp_symlink(self) -> Path:
        tmp_link = self.blob_path.with_name(f".{self.blob_path.name}.{secrets.token_hex(4)}.shared")
        relative_target = os.path.relpath(
            self.as_striped_path(self.shared_blob_path), start=self.as_striped_path(self.blob_path).parent
        )
        os.symlink(relative_target, str(tmp_link))
        return tmp_link

    def store_manifest_refs(self) -> None:
        manifest_path = self.shared_blob_path.with_name(f"{self.shared_blob_path.name}.refs")
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(manifest_path, flags, 0o666)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError(f"Shared blob manifest is not a regular file: '{manifest_path}'.")
            if hasattr(os, "fchmod"):
                try:
                    os.fchmod(fd, 0o666)
                except OSError:
                    pass
            line = f"{self.relative_blob_path}\n".encode()
            if os.write(fd, line) != len(line):
                raise OSError(f"Could not append a complete reference to '{manifest_path}'.")
            os.fsync(fd)
        finally:
            os.close(fd)
    
    def is_shared_blob_exist(self, expected_size: int | None) -> bool:
        """Materialize `blobs/<etag>` as a symlink to an existing store entry, if any.
    
        The reference manifest is flushed before the symlink becomes visible. Failures are
        best-effort misses and leave the regular download path untouched.
        """
        if not self.is_xet_hash_valid or not self.is_shared_blob_dir(self.shared_blob_dir) or expected_size is None:
            return False

        if self.relative_blob_path is None:
            return False

        tmp_link: Path | None = None
        lock_path = self.shared_blob_path.with_name(f"{self.shared_blob_path.name}.lock")
        try:
            with self.FileLock(lock_path):
                if not self.is_shared_blob_usable(expected_size):
                    return False

                tmp_link = self.get_shared_blob_temp_symlink()
                self.store_manifest_refs()
                os.replace(str(tmp_link), str(self.blob_path))
        except OSError as e:
            logger.debug(f"Could not symlink '{self.blob_path}' from shared blob store: {e}")
            return False
        finally:
            if tmp_link is not None:
                tmp_link.unlink(missing_ok=True)
            
        logger.debug(f"Blob '{self.blob_path}' reused from shared blob store (no download needed).")
        return True

    def _prepare_shared_blob_permissions(self, prefix_dir: Path) -> None:
        """Make a payload immutable and readable according to the shared cache policy."""
        blob_mode = self._shared_blob_mode(self.cache_dir)
        os.chmod(str(self.blob_path), blob_mode)
        if os.name == "nt" or not hasattr(os, "chown"):
            return
        target_gid = prefix_dir.stat().st_gid
        if self.blob_path.stat().st_gid == target_gid:
            return
        try:
            os.chown(str(self.blob_path), -1, target_gid)
        except OSError:
            # Without other-read, a wrong group makes the entry unreadable to other users: fall back to repo-local.
            if blob_mode & stat.S_IRGRP and not blob_mode & stat.S_IROTH:
                raise

    def store_shared_blob(self, expected_size: int | None, replace_existing: bool = False) -> bool:
        """Move a fresh Xet download into the store and replace its repo blob with a symlink.

        Best-effort: on failure the repo blob remains (or is restored as) a regular local
        file. Returns whether the repo blob was successfully shared.
        """
        if not self.is_xet_hash_valid or not expected_size:
            return False

        prefix_dir = self.get_shared_blob_prefix_dir()
        if self.relative_blob_path is None or prefix_dir is None:
            return False

        tmp_link: Path | None = None
        lock_path = self.shared_blob_path.with_name(f"{self.shared_blob_path.name}.lock")
        blob_moved = False
        try:
            with self.FileLock(lock_path):
                tmp_link = self.get_shared_blob_temp_symlink()
                store_is_usable = not replace_existing and self.is_shared_blob_usable(expected_size)
                self.store_manifest_refs()

                if not store_is_usable:
                    self._prepare_shared_blob_permissions(prefix_dir)
                    os.replace((self.blob_path), str(self.shared_blob_path))
                    blob_moved = True

                try:
                    os.replace(str(tmp_link), str(self.blob_path))
                except OSError:
                    if blob_moved:
                        shutil.copyfile(str(self.shared_blob_path), str(self.blob_path))  # restore under the lock, before GC can run
                        blob_moved = False
                    raise

        except OSError as e:
            logger.debug(f"Could not publish '{self.blob_path}' to shared blob store: {e}")
            if blob_moved:
                raise OSError(f"Could not restore repo blob '{self.blob_path}' after shared-store failure") from e
            return False

        finally:
            if tmp_link is not None:
                tmp_link.unlink(missing_ok=True)

        logger.debug(f"Blob '{self.blob_path}' published to shared blob store.")
        return True

if __name__ == "__main__":
    from rich import inspect
    repo_id = "test/iantirta"
    repo_type = "model"
    file1 = HFCachedFile(repo_type=repo_type, repo_id=repo_id, revision="main", filename="hello.py")
    file2 = HFCachedFile(repo_type=repo_type, repo_id=repo_id,)
    file3 = HFCachedFile(repo_type=repo_type, repo_id=repo_id, cache_dir="~/123")
    file4 = HFCachedFile(repo_type=repo_type, repo_id=repo_id, cache_dir="~/123", filename="hello.py")
    inspect(file1)
