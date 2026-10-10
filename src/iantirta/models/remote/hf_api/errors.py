import requests

# FILE METADATA ERRORS

class FileMetadataError(OSError):
    """Error triggered when the metadata of a file on the Hub cannot be retrieved (missing ETag or commit_hash).

    Inherits from `OSError` for backward compatibility.
    """


## Base Class For HF HUB Error

class HFHubHTTPError(requests.HTTPError, OSError):
    def __init__(
        self,
        message: str,
        *,
        response: requests.Response,
        server_message: str | None = None,
    ):
        self.request_id = (
            response.headers.get("x-request-id")
            or response.headers.get("X-Amzn-Trace-Id")
            or response.headers.get("x-amz-cf-id")
        )
        self.server_message = server_message
        super().__init__(message, response=response)

    
    def append_to_message(self, additional_message: str) -> None:
        """Append additional information to the `HfHubHTTPError` initial message."""
        self.args = (self.args[0] + additional_message,) + self.args[1:]

    @classmethod
    def _reconstruct_hf_hub_http_error(
        cls,
        message: str,
        response: requests.Response,
        server_message: str | None
    ) -> "HFHubHTTPError":
        return cls(message, response=response, server_message=server_message)

    def __reduce_ex__(self, protocol):
        """Fix pickling of Exception subclass with kwargs. We need to override __reduce_ex__ of the parent class"""
        return (self.__class__._reconstruct_hf_hub_http_error, (str(self), self.response, self.server_message))


# REVISION ERROR

class RevisionNotFoundError(HFHubHTTPError):

    repo_id: str | None = None
    repo_type: str | None = None


class RevisionResolutionError(Exception):
    """
    Raised by [`HfApi.resolve_revision`] when a revision cannot be resolved to a commit hash: the Hub could not be
    reached (offline mode, connection error, timeout, Hub downtime, ...) and no matching entry was found in the
    local cache.
    """


# REPOSITORY ERRORS

class RepositoryNotFoundError(HFHubHTTPError):
    """
    Raised when trying to access a hf.co URL with an invalid repository name, or
    with a private repo name the user does not have access to.
    """

    repo_id: str | None = None
    repo_type: str | None = None


class GatedRepoError(RepositoryNotFoundError):
    """
    Raised when trying to access a gated repository for which the user is not on the
    authorized list.

    Note: derives from `RepositoryNotFoundError` to ensure backward compatibility.
    """


class DisabledRepoError(HFHubHTTPError):
    """
    Raised when trying to access a repository that has been disabled by its author.
    """


# ENTRY ERRORS

class EntryNotFoundError(Exception):
    """
    Raised when entry not found, either locally or remotely.
    """


class RemoteEntryNotFoundError(HFHubHTTPError, EntryNotFoundError):

    repo_id: str | None = None
    repo_type: str | None = None


# BUCKET ERRORS

class BucketNotFoundError(HFHubHTTPError):
    """
    Raised when trying to access a bucket that does not exist.

    Attributes:
        bucket_id (`str` or `None`):
            The bucket id (namespace/name) that was not found, if it could be determined from the request URL.
    """

    bucket_id: str | None = None


# JOB ERRORS

class JobNotFoundError(HFHubHTTPError):
    """
    Raised when trying to access a Job that does not exist.

    Attributes:
        job_id (`str`):
            The job id that was not found.
    """

    job_id: str
