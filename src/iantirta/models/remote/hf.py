

def repo_folder_name(
    repo_id: str,
    repo_type: str = "model"
) -> str:
    """Return a serialized version of a hf.co
    repo name and type, safe for disk storage
    as a single non-nested folder.

    Example: models--julien-c--EsperBERTo-small
    """
    return "--".join(
        [f"{repo_type}s", *repo_id.split("/")]
    )
