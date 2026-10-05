import os
import warnings
from collections.abc import Callable, Iterable, Mapping
from contextlib import redirect_stdout
from dataclasses import dataclass, fields
from io import BytesIO
from typing import NewType, Union
from urllib.parse import urlparse

import numpy as np

from iantirta.models.tools._torch import is_torch_available, is_torchvision_available
from iantirta.models.tools._vision import is_vision_available
from iantirta.models.tools.tensor import is_numpy_array, is_torch_tensor

from .image_utils import is_valid_image

if is_vision_available():
    import PIL.Image

    if is_torchvision_available():
        from torchvision import io as torchvision_io

if is_torch_available():
    import torch


URL = NewType("URL", str)
Path = NewType("Path", str)

VideoInput = Union[
    list["PIL.Image.Image"],
    np.ndarray,
    "torch.Tensor",
    list[np.ndarray],
    list["torch.Tensor"],
    list[list["PIL.Image.Image"]],
    list[list[np.ndarray]],
    list[list["torch.Tensor"]],
    URL,
    list[URL],
    list[list[URL]],
    Path,
    list[Path],
    list[list[Path]],
]



def is_valid_video_frame(frame):
    return isinstance(frame, PIL.Image.Image) or (
        (is_numpy_array(frame) or is_torch_tensor(frame)) and frame.ndim == 3
    )


def is_valid_video(video):
    if not isinstance(video, (list, tuple)):
        return (is_numpy_array(video) or is_torch_tensor(video)) and video.ndim == 4
    return video and all(is_valid_video_frame(frame) for frame in video)


def is_batched_video(videos):
    if isinstance(videos, (list, tuple)):
        return is_valid_video(videos[0])
    elif (is_numpy_array(videos) or is_torch_tensor(videos)) and videos.ndim == 5:
        return True
    return False


def convert_pil_frames_to_video(videos: list[VideoInput]) -> list[Union[np.ndarray, "torch.Tensor"]]:
    """
    Given a batch of videos, converts each video to a 4D array. If video is already in array type,
    it is simply returned. We assume that all inputs in the list are in the same format, based on the type of the first element.

    Args:
        videos (`VideoInput`):
            Video inputs to turn into a list of videos.
    """

    if not (isinstance(videos[0], (list, tuple)) and is_valid_image(videos[0][0])):
        return videos

    video_converted = []
    for video in videos:
        video = [np.array(frame) for frame in video]
        video = np.stack(video)
        video_converted.append(video)
    return video_converted


def make_batched_videos(videos) -> list[Union[np.ndarray, "torch.Tensor", "URL", "Path"]]:
    """
    Ensure that the input is a list of videos. If the input is a single video, it is converted to a list of length 1.
    If the input is a batch of videos, it is converted to a list of 4D video arrays. Videos passed as list `PIL.Image`
    frames are converted to 4D arrays.

    We assume that all inputs in the list are in the same format, based on the type of the first element.

    Args:
        videos (`VideoInput`):
            Video inputs to turn into a list of videos.
    """
    # Early exit for deeply nested list of image frame paths. We shouldn't flatten them
    try:
        if isinstance(videos[0][0], (list, tuple)) and isinstance(videos[0][0][0], str):
            return [image_paths for sublist in videos for image_paths in sublist]
    except (IndexError, TypeError):
        pass

    if is_batched_video(videos):
        return convert_pil_frames_to_video(list(videos))
    elif isinstance(videos, str) or is_valid_video(videos):
        return convert_pil_frames_to_video([videos])
    # only one frame passed, thus we unsqueeze time dim
    elif is_valid_image(videos):
        if isinstance(videos, PIL.Image.Image):
            videos = np.array(videos)
        return [videos[None, ...]]
    elif not isinstance(videos, (list, tuple)):
        raise ValueError(
            f"Invalid video input. Expected either a list of video frames or an input of 4 or 5 dimensions, but got"
            f" type {type(videos)}."
        )

    # Recursively flatten any nested structure
    flat_videos_list = []
    for item in videos:
        if isinstance(item, str) or is_valid_video(item):
            flat_videos_list.append(item)
        elif isinstance(item, (list, tuple)) and item:
            flat_videos_list.extend(make_batched_videos(item))

    flat_videos_list = convert_pil_frames_to_video(flat_videos_list)
    return flat_videos_list
