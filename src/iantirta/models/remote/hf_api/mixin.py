
from functools import cached_property, lru_cache
from pathlib import Path
import os
import uuid

import requests

from ..cache_file import CachedFile
from ..http import HTTPHeader, HTTPMixin
from . import errors, hf_constant
from .types import (
    _BUCKET_ID_FROM_URL_RE,
    _JOB_ID_FROM_URL_RE,
    _REPO_ID_FROM_RESOLVE_URL_RE,
    _REPO_ID_FROM_URL_RE,
    _REPO_URL_SUBPATHS,
    BUCKET_API_RE,
    REPO_API_RE,
    _COMMIT_HASH_RE,
)


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


class HFCachedFIle(CachedFile):
    
    cache_dir = CachedFile.cache_dir / "hf_hub"
    repo_id: str | None = None
    repo_type: str | None = None
    revision: str | None = None
    etag: str | None = None
    
    _relative_filename: str | None = None

    def _repo_folder_name(self, repo_id: str, repo_type: str) -> str:
        return "--".join([f"{repo_type}s", *repo_id.split("/")])

    @property
    def revision_is_commit(self):
        return _COMMIT_HASH_RE.fullmatch(self.revision)

    @property
    def revision(self):
        return self.revision

    @revision.setter
    def revision(self, commit_hash: str):
        self.revision = commit_hash
        if self.revision != commit_hash:
            if not self.ref_path.exists() or commit_hash != self.ref_path.read_text():
                tmp_path = self.ref_path.with_name(f"{self.ref_path.name}.{uuid.uuid4().hex[:8]}.tmp")
                tmp_path.write_text(commit_hash)
                os.replace(tmp_path, self.ref_path)

    @property
    def ref_path(self) -> Path:
        ref_path = self.storage_dir / "refs" / self.revision
        ref_path.mkdir(parents=True, exist_ok=True)
        return ref_path

    @cached_property
    def locks_path(self) -> Path:
        return self.cache_dir / self._repo_folder_name(self.repo_id, self.repo_type) / f"{self.etag}.lock"

    @property
    def storage_dir(self) -> Path:
        return self.cache_dir / self._repo_folder_name(self.repo_id, self.repo_type)

    @cached_property
    def snapshot_dir(self) -> Path:
        return self.storage_dir / "snapshots"

    @cached_property
    def pointer_path(self) -> Path:
        """Symlink pointer path"""
        # [WARNING !!!] Don't use resolve on Symlink
        pointer_path = self.snapshot_dir / self.revision / self.filename
        if self.snapshot_dir.absolute() not in pointer_path.absolute().parents:
            raise ValueError(
                "Invalid pointer path: cannot create pointer path in snapshot folder if"
                f" `storage_folder='{self.storage_dir}'`, `revision='{self.revision}'` and"
                f" `relative_filename='{self.filename}'`."
            )
        return
