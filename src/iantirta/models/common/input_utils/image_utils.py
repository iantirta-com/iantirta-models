import base64
import os
from collections.abc import Iterable
from dataclasses import astuple, dataclass, fields
from io import BytesIO
from typing import Any, Union

import numpy as np

from iantirta.models.tools._torch import is_torch_available, is_torchvision_available
from iantirta.models.tools._vision import is_vision_available
from iantirta.models.tools.tensor import is_numpy_array, is_torch_tensor
from iantirta.models.tools.types import ExplicitEnum

if is_vision_available():
    import PIL.Image
    import PIL.ImageOps


if is_torch_available():
    import torch


ImageInput = Union[
    "PIL.Image.Image", np.ndarray, "torch.Tensor", list["PIL.Image.Image"], list[np.ndarray], list["torch.Tensor"]
]


class ChannelDimension(ExplicitEnum):
    FIRST = "channels_first"
    LAST = "channels_last"


def is_pil_image(img):
    return is_vision_available() and isinstance(img, PIL.Image.Image)


def is_valid_image(img):
    return is_pil_image(img) or is_numpy_array(img) or is_torch_tensor(img)


def is_valid_list_of_images(images: list):
    return images and all(is_valid_image(image) for image in images)


def make_flat_list_of_images(
    images: list[ImageInput] | ImageInput,
    expected_ndims: int = 3,
) -> ImageInput:
    """
    Ensure that the output is a flat list of images. If the input is a single image, it is converted to a list of length 1.
    If the input is a nested list of images, it is converted to a flat list of images.
    Args:
        images (`Union[list[ImageInput], ImageInput]`):
            The input image.
        expected_ndims (`int`, *optional*, defaults to 3):
            The expected number of dimensions for a single input image.
    Returns:
        list: A list of images or a 4d array of images.
    """
    # If the input is a nested list of images, we flatten it
    if (
        isinstance(images, (list, tuple))
        and all(isinstance(images_i, (list, tuple)) for images_i in images)
        and all(is_valid_list_of_images(images_i) or not images_i for images_i in images)
    ):
        return [img for img_list in images for img in img_list]

    if isinstance(images, (list, tuple)) and is_valid_list_of_images(images):
        if is_pil_image(images[0]) or images[0].ndim == expected_ndims:
            return images
        if images[0].ndim == expected_ndims + 1:
            return [img for img_list in images for img in img_list]

    if is_valid_image(images):
        if is_pil_image(images) or images.ndim == expected_ndims:
            return [images]
        if images.ndim == expected_ndims + 1:
            return list(images)

    raise ValueError(f"Could not make a flat list of images from {images}")
