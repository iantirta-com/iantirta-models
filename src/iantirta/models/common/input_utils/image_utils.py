import base64
import logging
import os
from collections import defaultdict
from collections.abc import Collection, Iterable
from dataclasses import astuple, dataclass, fields
from io import BytesIO
from math import ceil
from typing import Any, Optional, Union

import numpy as np

from iantirta.models.tools._torch import is_torch_available, is_torchvision_available
from iantirta.models.tools._vision import is_vision_available
from iantirta.models.tools.tensor import is_numpy_array, is_torch_tensor, to_numpy
from iantirta.models.tools.types import ExplicitEnum, TensorType

if is_vision_available():
    import PIL
    import PIL.Image
    import PIL.ImageOps

    PILImageResampling = PIL.Image.Resampling


if is_torchvision_available():
    from torchvision.io import ImageReadMode, decode_image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import pil_to_tensor

    pil_torch_interpolation_mapping = {
        PILImageResampling.NEAREST: InterpolationMode.NEAREST_EXACT,
        PILImageResampling.BOX: InterpolationMode.BOX,
        PILImageResampling.BILINEAR: InterpolationMode.BILINEAR,
        PILImageResampling.HAMMING: InterpolationMode.HAMMING,
        PILImageResampling.BICUBIC: InterpolationMode.BICUBIC,
        PILImageResampling.LANCZOS: InterpolationMode.LANCZOS,
    }
    # Create inverse mapping: InterpolationMode -> PILImageResampling
    torch_pil_interpolation_mapping = {v: k for k, v in pil_torch_interpolation_mapping.items()}
else:
    pil_torch_interpolation_mapping = {}
    torch_pil_interpolation_mapping = {}


if is_torch_available():
    import torch


logger = logging.getLogger(__name__)


ImageInput = Union[
    "PIL.Image.Image", np.ndarray, "torch.Tensor", list["PIL.Image.Image"], list[np.ndarray], list["torch.Tensor"]
]


class ChannelDimension(ExplicitEnum):
    FIRST = "channels_first"
    LAST = "channels_last"


class AnnotationFormat(ExplicitEnum):
    COCO_DETECTION = "coco_detection"
    COCO_PANOPTIC = "coco_panoptic"


AnnotationType = dict[str, int | str | list[dict]]


def is_pil_image(img):
    return is_vision_available() and isinstance(img, PIL.Image.Image)


class ImageType(ExplicitEnum):
    PIL = "pillow"
    TORCH = "torch"
    NUMPY = "numpy"


def get_image_type(image):
    if is_pil_image(image):
        return ImageType.PIL
    if is_torch_tensor(image):
        return ImageType.TORCH
    if is_numpy_array(image):
        return ImageType.NUMPY
    raise ValueError(f"Unrecognized image type {type(image)}")


def is_valid_image(img):
    return is_pil_image(img) or is_numpy_array(img) or is_torch_tensor(img)


def is_valid_list_of_images(images: list):
    return images and all(is_valid_image(image) for image in images)


def concatenate_list(input_list):
    if isinstance(input_list[0], list):
        return [item for sublist in input_list for item in sublist]
    elif isinstance(input_list[0], np.ndarray):
        return np.concatenate(input_list, axis=0)
    elif isinstance(input_list[0], torch.Tensor):
        return torch.cat(input_list, dim=0)


def valid_images(imgs):
    # If we have an list of images, make sure every image is valid
    if isinstance(imgs, (list, tuple)):
        for img in imgs:
            if not valid_images(img):
                return False
    # If not a list of tuple, we have been given a single image or batched tensor of images
    elif not is_valid_image(imgs):
        return False
    return True


def is_batched(img):
    if isinstance(img, (list, tuple)):
        return is_valid_image(img[0])
    return False


def is_scaled_image(image: np.ndarray) -> bool:
    """
    Checks to see whether the pixel values have already been rescaled to [0, 1].
    """
    if image.dtype == np.uint8:
        return False

    # It's possible the image has pixel values in [0, 255] but is of floating type
    return np.min(image) >= 0 and np.max(image) <= 1


def make_list_of_images(images, expected_ndims: int = 3) -> list[ImageInput]:
    """
    Ensure that the output is a list of images. If the input is a single image, it is converted to a list of length 1.
    If the input is a batch of images, it is converted to a list of images.

    Args:
        images (`ImageInput`):
            Image or batch of images to turn into a list of images.
        expected_ndims (`int`, *optional*, defaults to 3):
            Expected number of dimensions for a single input image. If the input image has a different number of
            dimensions, an error is raised.
    """
    if is_batched(images):
        return images

    # Either the input is a single image, in which case we create a list of length 1
    if is_pil_image(images):
        # PIL images are never batched
        return [images]

    if is_valid_image(images):
        if images.ndim == expected_ndims + 1:
            # Batch of images
            images = list(images)
        elif images.ndim == expected_ndims:
            # Single image
            images = [images]
        else:
            raise ValueError(
                f"Invalid image shape. Expected either {expected_ndims + 1} or {expected_ndims} dimensions, but got"
                f" {images.ndim} dimensions."
            )
        return images
    raise ValueError(
        f"Invalid image type. Expected either PIL.Image.Image, numpy.ndarray, or torch.Tensor, but got {type(images)}."
    )


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


def make_nested_list_of_images(
    images: list[ImageInput] | ImageInput,
    expected_ndims: int = 3,
) -> list[ImageInput]:
    """
    Ensure that the output is a nested list of images.
    Args:
        images (`Union[list[ImageInput], ImageInput]`):
            The input image.
        expected_ndims (`int`, *optional*, defaults to 3):
            The expected number of dimensions for a single input image.
    Returns:
        list: A list of list of images or a list of 4d array of images.
    """
    # If it's a list of batches, it's already in the right format
    if (
        isinstance(images, (list, tuple))
        and all(isinstance(images_i, (list, tuple)) for images_i in images)
        and all(is_valid_list_of_images(images_i) or not images_i for images_i in images)
    ):
        return images

    # If it's a list of images, it's a single batch, so convert it to a list of lists
    if isinstance(images, (list, tuple)) and is_valid_list_of_images(images):
        if is_pil_image(images[0]) or images[0].ndim == expected_ndims:
            return [images]
        if images[0].ndim == expected_ndims + 1:
            return [list(image) for image in images]

    # If it's a single image, convert it to a list of lists
    if is_valid_image(images):
        if is_pil_image(images) or images.ndim == expected_ndims:
            return [[images]]
        if images.ndim == expected_ndims + 1:
            return [list(images)]

    raise ValueError("Invalid input type. Must be a single image, a list of images, or a list of batches of images.")


def to_numpy_array(img) -> np.ndarray:
    if not is_valid_image(img):
        raise ValueError(f"Invalid image type: {type(img)}")

    if is_vision_available() and isinstance(img, PIL.Image.Image):
        return np.array(img)
    return to_numpy(img)


def infer_channel_dimension_format(
    image: np.ndarray, num_channels: int | tuple[int, ...] | None = None
) -> ChannelDimension:
    """
    Infers the channel dimension format of `image`.

    Args:
        image (`np.ndarray`):
            The image to infer the channel dimension of.
        num_channels (`int` or `tuple[int, ...]`, *optional*, defaults to `(1, 3)`):
            The number of channels of the image.

    Returns:
        The channel dimension of the image.
    """
    num_channels = num_channels if num_channels is not None else (1, 3)
    num_channels = (num_channels,) if isinstance(num_channels, int) else num_channels

    if image.ndim == 3:
        first_dim, last_dim = 0, 2
    elif image.ndim == 4:
        first_dim, last_dim = 1, 3
    elif image.ndim == 5:
        first_dim, last_dim = 2, 4
    else:
        raise ValueError(f"Unsupported number of image dimensions: {image.ndim}")

    if image.shape[first_dim] in num_channels and image.shape[last_dim] in num_channels:
        logger.warning(
            f"The channel dimension is ambiguous. Got image shape {image.shape}. Assuming channels are the first dimension. Use the [input_data_format](https://huggingface.co/docs/transformers/main/internal/image_processing_utils#transformers.image_transforms.rescale.input_data_format) parameter to assign the channel dimension."
        )
        return ChannelDimension.FIRST
    elif image.shape[first_dim] in num_channels:
        return ChannelDimension.FIRST
    elif image.shape[last_dim] in num_channels:
        return ChannelDimension.LAST
    raise ValueError("Unable to infer channel dimension format")


def get_channel_dimension_axis(image: np.ndarray, input_data_format: ChannelDimension | str | None = None) -> int:
    """
    Returns the channel dimension axis of the image.

    Args:
        image (`np.ndarray`):
            The image to get the channel dimension axis of.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format of the image. If `None`, will infer the channel dimension from the image.

    Returns:
        The channel dimension axis of the image.
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    if input_data_format == ChannelDimension.FIRST:
        return image.ndim - 3
    elif input_data_format == ChannelDimension.LAST:
        return image.ndim - 1
    raise ValueError(f"Unsupported data format: {input_data_format}")


def get_image_size(
    image: Union[np.ndarray, "PIL.Image.Image"], channel_dim: ChannelDimension | None = None
) -> tuple[int, int]:
    """
    Returns the (height, width) dimensions of the image.

    Args:
        image (`np.ndarray | PIL.Image.Image`):
            The image to get the dimensions of.
        channel_dim (`ChannelDimension`, *optional*):
            Which dimension the channel dimension is in. If `None`, will infer the channel dimension from the image.

    Returns:
        A tuple of the image's height and width.
    """
    if isinstance(image, PIL.Image.Image):
        return image.size

    if channel_dim is None:
        channel_dim = infer_channel_dimension_format(image)

    if channel_dim == ChannelDimension.FIRST:
        return image.shape[-2], image.shape[-1]
    elif channel_dim == ChannelDimension.LAST:
        return image.shape[-3], image.shape[-2]
    else:
        raise ValueError(f"Unsupported data format: {channel_dim}")


def get_image_size_for_max_height_width(
    image_size: tuple[int, int],
    max_height: int,
    max_width: int,
) -> tuple[int, int]:
    """
    Computes the output image size given the input image and the maximum allowed height and width. Keep aspect ratio.
    Important, even if image_height < max_height and image_width < max_width, the image will be resized
    to at least one of the edges be equal to max_height or max_width.

    For example:
        - input_size: (100, 200), max_height: 50, max_width: 50 -> output_size: (25, 50)
        - input_size: (100, 200), max_height: 200, max_width: 500 -> output_size: (200, 400)

    Args:
        image_size (`tuple[int, int]`):
            The image to resize.
        max_height (`int`):
            The maximum allowed height.
        max_width (`int`):
            The maximum allowed width.
    """
    height, width = image_size
    height_scale = max_height / height
    width_scale = max_width / width
    min_scale = min(height_scale, width_scale)
    new_height = int(height * min_scale)
    new_width = int(width * min_scale)
    return new_height, new_width


def max_across_indices(values: Iterable[Any]) -> list[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


def get_max_height_width(
    images: list[Union["torch.Tensor", np.ndarray]], input_data_format: str | ChannelDimension = ChannelDimension.FIRST
) -> list[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_data_format == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")
    return (max_height, max_width)


def is_valid_annotation_coco_detection(annotation: dict[str, list | tuple]) -> bool:
    return bool(
        isinstance(annotation, dict)
        and "image_id" in annotation
        and "annotations" in annotation
        and isinstance(annotation["annotations"], (list, tuple))
        and (
            # an image can have no annotations
            len(annotation["annotations"]) == 0 or isinstance(annotation["annotations"][0], dict)
        )
    )


def is_valid_annotation_coco_panoptic(annotation: dict[str, list | tuple]) -> bool:
    return bool(
        isinstance(annotation, dict)
        and "image_id" in annotation
        and "segments_info" in annotation
        and "file_name" in annotation
        and isinstance(annotation["segments_info"], (list, tuple))
        and (
            # an image can have no segments
            len(annotation["segments_info"]) == 0 or isinstance(annotation["segments_info"][0], dict)
        )
    )


def valid_coco_detection_annotations(annotations: Iterable[dict[str, list | tuple]]) -> bool:
    return all(is_valid_annotation_coco_detection(ann) for ann in annotations)


def valid_coco_panoptic_annotations(annotations: Iterable[dict[str, list | tuple]]) -> bool:
    return all(is_valid_annotation_coco_panoptic(ann) for ann in annotations)


def load_image(
    image: Union[str, "PIL.Image.Image"],
    timeout: float | None = None,
) -> "PIL.Image.Image":
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        timeout (`float`, *optional*):
            The timeout value in seconds for the URL request.

    Returns:
        `PIL.Image.Image`: A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith(("http://", "https://")):
            # We need to actually check for a real protocol, otherwise it's impossible to use a local file
            # like http_huggingface_co.png
            raise NotImplementedError()
            # image = PIL.Image.open(BytesIO(httpx.get(image, timeout=timeout, follow_redirects=True).content))
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            if image.startswith("data:image/"):
                image = image.split(",")[1]

            # Try to load as base64
            try:
                b64 = base64.decodebytes(image.encode())
                image = PIL.Image.open(BytesIO(b64))
            except Exception as e:  # noqa: BLE001
                raise ValueError(
                    f"Incorrect image source. Must be a valid URL starting with `http://` or `https://`, a valid path to an image file, or a base64 encoded string. Got {image}. Failed with {e}"
                )
    elif not isinstance(image, PIL.Image.Image):
        raise TypeError(
            "Incorrect format used for image. Should be an url linking to an image, a base64 string, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def load_image_as_tensor(
    image: Union[str, "PIL.Image.Image"],
    timeout: float | None = None,
) -> "torch.Tensor":
    """
    Loads `image` directly to a `torch.Tensor` using torchvision.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        timeout (`float`, *optional*):
            The timeout value in seconds for the URL request.

    Returns:
        `torch.Tensor`: A `[C, H, W]` uint8 tensor in RGB channel order.
    """
    import torch

    if isinstance(image, str):
        if image.startswith(("http://", "https://")):
            raise NotImplementedError()
            # raw = httpx.get(image, timeout=timeout, follow_redirects=True).content
            # buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            # return decode_image(buf, mode=ImageReadMode.RGB)
        elif os.path.isfile(image):
            return decode_image(image, mode=ImageReadMode.RGB)
        else:
            if image.startswith("data:image/"):
                image = image.split(",")[1]
            try:
                raw = base64.decodebytes(image.encode())
            except Exception as e:  # noqa: BLE001
                raise ValueError(
                    f"Incorrect image source. Must be a valid URL starting with `http://` or `https://`, a valid path to an image file, or a base64 encoded string. Got {image}. Failed with {e}"
                )
            buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            return decode_image(buf, mode=ImageReadMode.RGB)
    elif isinstance(image, PIL.Image.Image):
        image = PIL.ImageOps.exif_transpose(image)
        return pil_to_tensor(image.convert("RGB"))
    else:
        raise TypeError(
            "Incorrect format used for image. Should be a URL, a local path, a base64 string, or a PIL image."
        )


def load_images(
    images: Union[list, tuple, str, "PIL.Image.Image"], timeout: float | None = None
) -> Union["PIL.Image.Image", list["PIL.Image.Image"], list[list["PIL.Image.Image"]]]:
    """Loads images, handling different levels of nesting.

    Args:
      images: A single image, a list of images, or a list of lists of images to load.
      timeout: Timeout for loading images.

    Returns:
      A single image, a list of images, a list of lists of images.
    """
    if isinstance(images, (list, tuple)):
        if len(images) and isinstance(images[0], (list, tuple)):
            return [[load_image(image, timeout=timeout) for image in image_group] for image_group in images]
        else:
            return [load_image(image, timeout=timeout) for image in images]
    else:
        return load_image(images, timeout=timeout)


def validate_preprocess_arguments(
    do_rescale: bool | None = None,
    rescale_factor: float | None = None,
    do_normalize: bool | None = None,
    image_mean: float | list[float] | None = None,
    image_std: float | list[float] | None = None,
    do_pad: bool | None = None,
    pad_size: dict[str, int] | int | None = None,
    do_center_crop: bool | None = None,
    crop_size: dict[str, int] | None = None,
    do_resize: bool | None = None,
    size: dict[str, int] | None = None,
    resample: Union["PILImageResampling", "InterpolationMode", int] | None = None,
):
    """
    Checks validity of typically used arguments in an `ImageProcessor` `preprocess` method.
    Raises `ValueError` if arguments incompatibility is caught.
    Many incompatibilities are model-specific. `do_pad` sometimes needs `size_divisor`,
    sometimes `size_divisibility`, and sometimes `size`. New models and processors added should follow
    existing arguments when possible.

    """
    if do_rescale and rescale_factor is None:
        raise ValueError("`rescale_factor` must be specified if `do_rescale` is `True`.")

    if do_pad and pad_size is None:
        # Processors pad images using different args depending on the model, so the below check is pointless
        # but we keep it for BC for now. TODO: remove in v5
        # Usually padding can be called with:
        #   - "pad_size/size" if we're padding to specific values
        #   - "size_divisor" if we're padding to any value divisible by X
        #   - "None" if we're padding to the maximum size image in batch
        raise ValueError(
            "Depending on the model, `size_divisor` or `pad_size` or `size` must be specified if `do_pad` is `True`."
        )

    if do_normalize and (image_mean is None or image_std is None):
        raise ValueError("`image_mean` and `image_std` must both be specified if `do_normalize` is `True`.")

    if do_center_crop and crop_size is None:
        raise ValueError("`crop_size` must be specified if `do_center_crop` is `True`.")

    if do_resize and not (size is not None and resample is not None):
        raise ValueError("`size` and `resample` must be specified if `do_resize` is `True`.")


class ImageFeatureExtractionMixin:
    """
    Mixin that contain utilities for preparing image features.
    """

    def _ensure_format_supported(self, image):
        if not isinstance(image, (PIL.Image.Image, np.ndarray)) and not is_torch_tensor(image):
            raise ValueError(
                f"Got type {type(image)} which is not supported, only `PIL.Image.Image`, `np.ndarray` and "
                "`torch.Tensor` are."
            )

    def to_pil_image(self, image, rescale=None):
        """
        Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if
        needed.

        Args:
            image (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor`):
                The image to convert to the PIL Image format.
            rescale (`bool`, *optional*):
                Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will
                default to `True` if the image type is a floating type, `False` otherwise.
        """
        self._ensure_format_supported(image)

        if is_torch_tensor(image):
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if rescale is None:
                # rescale default to the array being of floating type.
                rescale = isinstance(image.flat[0], np.floating)
            # If the channel as been moved to first dim, we put it back at the end.
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
            if rescale:
                image = image * 255
            image = image.astype(np.uint8)
            return PIL.Image.fromarray(image)
        return image

    def convert_rgb(self, image):
        """
        Converts `PIL.Image.Image` to RGB format.

        Args:
            image (`PIL.Image.Image`):
                The image to convert.
        """
        self._ensure_format_supported(image)
        if not isinstance(image, PIL.Image.Image):
            return image

        return image.convert("RGB")

    def rescale(self, image: np.ndarray, scale: float) -> np.ndarray:
        """
        Rescale a numpy image by scale amount
        """
        self._ensure_format_supported(image)
        return image * scale

    def to_numpy_array(self, image, rescale=None, channel_first=True):
        """
        Converts `image` to a numpy array. Optionally rescales it and puts the channel dimension as the first
        dimension.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to convert to a NumPy array.
            rescale (`bool`, *optional*):
                Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.). Will
                default to `True` if the image is a PIL Image or an array/tensor of integers, `False` otherwise.
            channel_first (`bool`, *optional*, defaults to `True`):
                Whether or not to permute the dimensions of the image to put the channel dimension first.
        """
        self._ensure_format_supported(image)

        if isinstance(image, PIL.Image.Image):
            image = np.array(image)

        if is_torch_tensor(image):
            image = image.numpy()

        rescale = isinstance(image.flat[0], np.integer) if rescale is None else rescale

        if rescale:
            image = self.rescale(image.astype(np.float32), 1 / 255.0)

        if channel_first and image.ndim == 3:
            image = image.transpose(2, 0, 1)

        return image

    def expand_dims(self, image):
        """
        Expands 2-dimensional `image` to 3 dimensions.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to expand.
        """
        self._ensure_format_supported(image)

        # Do nothing if PIL image
        if isinstance(image, PIL.Image.Image):
            return image

        if is_torch_tensor(image):
            image = image.unsqueeze(0)
        else:
            image = np.expand_dims(image, axis=0)
        return image

    def normalize(self, image, mean, std, rescale=False):
        """
        Normalizes `image` with `mean` and `std`. Note that this will trigger a conversion of `image` to a NumPy array
        if it's a PIL Image.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to normalize.
            mean (`list[float]` or `np.ndarray` or `torch.Tensor`):
                The mean (per channel) to use for normalization.
            std (`list[float]` or `np.ndarray` or `torch.Tensor`):
                The standard deviation (per channel) to use for normalization.
            rescale (`bool`, *optional*, defaults to `False`):
                Whether or not to rescale the image to be between 0 and 1. If a PIL image is provided, scaling will
                happen automatically.
        """
        self._ensure_format_supported(image)

        if isinstance(image, PIL.Image.Image):
            image = self.to_numpy_array(image, rescale=True)
        # If the input image is a PIL image, it automatically gets rescaled. If it's another
        # type it may need rescaling.
        elif rescale:
            if isinstance(image, np.ndarray):
                image = self.rescale(image.astype(np.float32), 1 / 255.0)
            elif is_torch_tensor(image):
                image = self.rescale(image.float(), 1 / 255.0)

        if isinstance(image, np.ndarray):
            if not isinstance(mean, np.ndarray):
                mean = np.array(mean).astype(image.dtype)
            if not isinstance(std, np.ndarray):
                std = np.array(std).astype(image.dtype)
        elif is_torch_tensor(image):
            import torch

            if not isinstance(mean, torch.Tensor):
                if isinstance(mean, np.ndarray):
                    mean = torch.from_numpy(mean)
                else:
                    mean = torch.tensor(mean)
            if not isinstance(std, torch.Tensor):
                if isinstance(std, np.ndarray):
                    std = torch.from_numpy(std)
                else:
                    std = torch.tensor(std)

        if image.ndim == 3 and image.shape[0] in [1, 3]:
            return (image - mean[:, None, None]) / std[:, None, None]
        else:
            return (image - mean) / std

    def resize(self, image, size, resample=None, default_to_square=True, max_size=None):
        """
        Resizes `image`. Enforces conversion of input to PIL.Image.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to resize.
            size (`int` or `tuple[int, int]`):
                The size to use for resizing the image. If `size` is a sequence like (h, w), output size will be
                matched to this.

                If `size` is an int and `default_to_square` is `True`, then image will be resized to (size, size). If
                `size` is an int and `default_to_square` is `False`, then smaller edge of the image will be matched to
                this number. i.e, if height > width, then image will be rescaled to (size * height / width, size).
            resample (`int`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                The filter to user for resampling.
            default_to_square (`bool`, *optional*, defaults to `True`):
                How to convert `size` when it is a single int. If set to `True`, the `size` will be converted to a
                square (`size`,`size`). If set to `False`, will replicate
                [`torchvision.transforms.Resize`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Resize)
                with support for resizing only the smallest edge and providing an optional `max_size`.
            max_size (`int`, *optional*, defaults to `None`):
                The maximum allowed for the longer edge of the resized image: if the longer edge of the image is
                greater than `max_size` after being resized according to `size`, then the image is resized again so
                that the longer edge is equal to `max_size`. As a result, `size` might be overruled, i.e the smaller
                edge may be shorter than `size`. Only used if `default_to_square` is `False`.

        Returns:
            image: A resized `PIL.Image.Image`.
        """
        resample = resample if resample is not None else PILImageResampling.BILINEAR

        self._ensure_format_supported(image)

        if not isinstance(image, PIL.Image.Image):
            image = self.to_pil_image(image)

        if isinstance(size, list):
            size = tuple(size)

        if isinstance(size, int) or len(size) == 1:
            if default_to_square:
                size = (size, size) if isinstance(size, int) else (size[0], size[0])
            else:
                width, height = image.size
                # specified size only for the smallest edge
                short, long = (width, height) if width <= height else (height, width)
                requested_new_short = size if isinstance(size, int) else size[0]

                if short == requested_new_short:
                    return image

                new_short, new_long = requested_new_short, int(requested_new_short * long / short)

                if max_size is not None:
                    if max_size <= requested_new_short:
                        raise ValueError(
                            f"max_size = {max_size} must be strictly greater than the requested "
                            f"size for the smaller edge size = {size}"
                        )
                    if new_long > max_size:
                        new_short, new_long = int(max_size * new_short / new_long), max_size

                size = (new_short, new_long) if width <= height else (new_long, new_short)

        return image.resize(size, resample=resample)

    def center_crop(self, image, size):
        """
        Crops `image` to the given size using a center crop. Note that if the image is too small to be cropped to the
        size given, it will be padded (so the returned result has the size asked).

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor` of shape (n_channels, height, width) or (height, width, n_channels)):
                The image to resize.
            size (`int` or `tuple[int, int]`):
                The size to which crop the image.

        Returns:
            new_image: A center cropped `PIL.Image.Image` or `np.ndarray` or `torch.Tensor` of shape: (n_channels,
            height, width).
        """
        self._ensure_format_supported(image)

        if not isinstance(size, tuple):
            size = (size, size)

        # PIL Image.size is (width, height) but NumPy array and torch Tensors have (height, width)
        if is_torch_tensor(image) or isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = self.expand_dims(image)
            image_shape = image.shape[1:] if image.shape[0] in [1, 3] else image.shape[:2]
        else:
            image_shape = (image.size[1], image.size[0])

        top = (image_shape[0] - size[0]) // 2
        bottom = top + size[0]  # In case size is odd, (image_shape[0] + size[0]) // 2 won't give the proper result.
        left = (image_shape[1] - size[1]) // 2
        right = left + size[1]  # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.

        # For PIL Images we have a method to crop directly.
        if isinstance(image, PIL.Image.Image):
            return image.crop((left, top, right, bottom))

        # Check if image is in (n_channels, height, width) or (height, width, n_channels) format
        channel_first = image.shape[0] in [1, 3]

        # Transpose (height, width, n_channels) format images
        if not channel_first:
            if isinstance(image, np.ndarray):
                image = image.transpose(2, 0, 1)
            if is_torch_tensor(image):
                image = image.permute(2, 0, 1)

        # Check if cropped area is within image boundaries
        if top >= 0 and bottom <= image_shape[0] and left >= 0 and right <= image_shape[1]:
            return image[..., top:bottom, left:right]

        # Otherwise, we may need to pad if the image is too small. Oh joy...
        new_shape = image.shape[:-2] + (max(size[0], image_shape[0]), max(size[1], image_shape[1]))
        if isinstance(image, np.ndarray):
            new_image = np.zeros_like(image, shape=new_shape)
        elif is_torch_tensor(image):
            new_image = image.new_zeros(new_shape)

        top_pad = (new_shape[-2] - image_shape[0]) // 2
        bottom_pad = top_pad + image_shape[0]
        left_pad = (new_shape[-1] - image_shape[1]) // 2
        right_pad = left_pad + image_shape[1]
        new_image[..., top_pad:bottom_pad, left_pad:right_pad] = image

        top += top_pad
        bottom += top_pad
        left += left_pad
        right += left_pad

        new_image = new_image[
            ..., max(0, top) : min(new_image.shape[-2], bottom), max(0, left) : min(new_image.shape[-1], right)
        ]

        return new_image

    def flip_channel_order(self, image):
        """
        Flips the channel order of `image` from RGB to BGR, or vice versa. Note that this will trigger a conversion of
        `image` to a NumPy array if it's a PIL Image.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image whose color channels to flip. If `np.ndarray` or `torch.Tensor`, the channel dimension should
                be first.
        """
        self._ensure_format_supported(image)

        if isinstance(image, PIL.Image.Image):
            image = self.to_numpy_array(image)

        return image[::-1, :, :]

    def rotate(self, image, angle, resample=None, expand=0, center=None, translate=None, fillcolor=None):
        """
        Returns a rotated copy of `image`. This method returns a copy of `image`, rotated the given number of degrees
        counter clockwise around its centre.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to rotate. If `np.ndarray` or `torch.Tensor`, will be converted to `PIL.Image.Image` before
                rotating.

        Returns:
            image: A rotated `PIL.Image.Image`.
        """
        resample = resample if resample is not None else PIL.Image.NEAREST

        self._ensure_format_supported(image)

        if not isinstance(image, PIL.Image.Image):
            image = self.to_pil_image(image)

        return image.rotate(
            angle, resample=resample, expand=expand, center=center, translate=translate, fillcolor=fillcolor
        )


def validate_annotations(
    annotation_format: AnnotationFormat,
    supported_annotation_formats: tuple[AnnotationFormat, ...],
    annotations: list[dict],
) -> None:
    if annotation_format not in supported_annotation_formats:
        raise ValueError(f"Unsupported annotation format: {format} must be one of {supported_annotation_formats}")

    if (
        annotation_format is AnnotationFormat.COCO_DETECTION
        and not valid_coco_detection_annotations(annotations)
    ):
        raise ValueError(
            "Invalid COCO detection annotations. Annotations must a dict (single image) or list of dicts "
            "(batch of images) with the following keys: `image_id` and `annotations`, with the latter "
            "being a list of annotations in the COCO format."
        )

    if (
        annotation_format is AnnotationFormat.COCO_PANOPTIC
        and not valid_coco_panoptic_annotations(annotations)
    ):
        raise ValueError(
            "Invalid COCO panoptic annotations. Annotations must a dict (single image) or list of dicts "
            "(batch of images) with the following keys: `image_id`, `file_name` and `segments_info`, with "
            "the latter being a list of annotations in the COCO format."
        )


def validate_kwargs(valid_processor_keys: list[str], captured_kwargs: list[str]):
    unused_keys = set(captured_kwargs).difference(set(valid_processor_keys))
    if unused_keys:
        unused_key_str = ", ".join(unused_keys)
        # TODO raise a warning here instead of simply logging?
        logger.warning(f"Unused or unrecognized kwargs: {unused_key_str}.")


@dataclass()
class SizeDict:
    """
    Hashable dictionary to store image size information.
    """

    height: int | None = None
    width: int | None = None
    longest_edge: int | None = None
    shortest_edge: int | None = None
    max_height: int | None = None
    max_width: int | None = None
    min_pixels: int | None = None
    max_pixels: int | None = None

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key {key} not found in SizeDict.")

    def get(self, key, default=None):
        if hasattr(self, key) and getattr(self, key) is not None:
            return getattr(self, key)
        return default

    def __iter__(self):
        # Yield only non-None (key, value) pairs so dict(self) excludes missing values.
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                yield f.name, val

    def __hash__(self):
        return hash(astuple(self))

    def __contains__(self, key):
        return hasattr(self, key) and getattr(self, key) is not None

    def __setitem__(self, key, value):
        if not hasattr(self, key):
            raise KeyError(f"Key {key} is not a valid field of SizeDict.")
        object.__setattr__(self, key, value)

    def __eq__(self, other):
        if isinstance(other, dict):
            return dict(self) == other
        if isinstance(other, SizeDict):
            return tuple(getattr(self, f.name) for f in fields(self)) == tuple(
                getattr(other, f.name) for f in fields(self)
            )
        return NotImplemented

    def __or__(self, other) -> "SizeDict":
        if isinstance(other, dict | SizeDict):
            merged = dict(self)
            merged.update(dict(other))
            return SizeDict(**merged)
        return NotImplemented

    def __ror__(self, other) -> dict:
        if isinstance(other, dict):
            merged = dict(other)
            merged.update(dict(self))
            return merged
        return NotImplemented


#### Image Transform Utility

def to_channel_dimension_format(
    image: np.ndarray,
    channel_dim: ChannelDimension | str,
    input_channel_dim: ChannelDimension | str | None = None,
) -> np.ndarray:
    """
    Converts `image` to the channel dimension format specified by `channel_dim`. The input
    can have arbitrary number of leading dimensions. Only last three dimension will be permuted
    to format the `image`.

    Args:
        image (`numpy.ndarray`):
            The image to have its channel dimension set.
        channel_dim (`ChannelDimension`):
            The channel dimension format to use.
        input_channel_dim (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.

    Returns:
        `np.ndarray`: The image with the channel dimension set to `channel_dim`.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input image must be of type np.ndarray, got {type(image)}")

    if input_channel_dim is None:
        input_channel_dim = infer_channel_dimension_format(image)

    target_channel_dim = ChannelDimension(channel_dim)
    if input_channel_dim == target_channel_dim:
        return image

    if target_channel_dim == ChannelDimension.FIRST:
        axes = list(range(image.ndim - 3)) + [image.ndim - 1, image.ndim - 3, image.ndim - 2]
        image = image.transpose(axes)
    elif target_channel_dim == ChannelDimension.LAST:
        axes = list(range(image.ndim - 3)) + [image.ndim - 2, image.ndim - 1, image.ndim - 3]
        image = image.transpose(axes)
    else:
        raise ValueError(f"Unsupported channel dimension format: {channel_dim}")

    return image


def rescale(
    image: np.ndarray,
    scale: float,
    data_format: ChannelDimension | None = None,
    dtype: np.dtype = np.float32,
    input_data_format: str | ChannelDimension | None = None,
) -> np.ndarray:
    """
    Rescales `image` by `scale`.

    Args:
        image (`np.ndarray`):
            The image to rescale.
        scale (`float`):
            The scale to use for rescaling the image.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the image. If not provided, it will be the same as the input image.
        dtype (`np.dtype`, *optional*, defaults to `np.float32`):
            The dtype of the output image. Defaults to `np.float32`. Used for backwards compatibility with feature
            extractors.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.

    Returns:
        `np.ndarray`: The rescaled image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input image must be of type np.ndarray, got {type(image)}")

    rescaled_image = image.astype(np.float64) * scale  # Numpy type promotion has changed, so always upcast first
    if data_format is not None:
        rescaled_image = to_channel_dimension_format(rescaled_image, data_format, input_data_format)

    rescaled_image = rescaled_image.astype(dtype)  # Finally downcast to the desired dtype at the end

    return rescaled_image


def _rescale_for_pil_conversion(image):
    """
    Detects whether or not the image needs to be rescaled before being converted to a PIL image.

    The assumption is that if the image is of type `np.float` and all values are between 0 and 1, it needs to be
    rescaled.
    """
    if image.dtype == np.uint8:
        do_rescale = False
    elif np.allclose(image, image.astype(int)):
        if np.all(image >= 0) and np.all(image <= 255):
            do_rescale = False
        else:
            raise ValueError(
                "The image to be converted to a PIL image contains values outside the range [0, 255], "
                f"got [{image.min()}, {image.max()}] which cannot be converted to uint8."
            )
    elif np.all(image >= 0) and np.all(image <= 1):
        do_rescale = True
    else:
        raise ValueError(
            "The image to be converted to a PIL image contains values outside the range [0, 1], "
            f"got [{image.min()}, {image.max()}] which cannot be converted to uint8."
        )
    return do_rescale


def to_pil_image(
    image: Union[np.ndarray, "PIL.Image.Image", "torch.Tensor"],
    do_rescale: bool | None = None,
    image_mode: str | None = None,
    input_data_format: str | ChannelDimension | None = None,
) -> "PIL.Image.Image":
    """
    Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if
    needed.

    Args:
        image (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor`):
            The image to convert to the `PIL.Image` format.
        do_rescale (`bool`, *optional*):
            Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will default
            to `True` if the image type is a floating type and casting to `int` would result in a loss of precision,
            and `False` otherwise.
        image_mode (`str`, *optional*):
            The mode to use for the PIL image. If unset, will use the default mode for the input image type.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        `PIL.Image.Image`: The converted image.
    """

    if isinstance(image, PIL.Image.Image):
        return image

    # Convert all tensors to numpy arrays before converting to PIL image
    if is_torch_tensor(image):
        image = image.numpy()
    elif not isinstance(image, np.ndarray):
        raise ValueError(f"Input image type not supported: {type(image)}")

    # If the channel has been moved to first dim, we put it back at the end.
    image = to_channel_dimension_format(image, ChannelDimension.LAST, input_data_format)

    # If there is a single channel, we squeeze it, as otherwise PIL can't handle it.
    image = np.squeeze(image, axis=-1) if image.shape[-1] == 1 else image

    # PIL.Image can only store uint8 values so we rescale the image to be between 0 and 255 if needed.
    do_rescale = _rescale_for_pil_conversion(image) if do_rescale is None else do_rescale

    if do_rescale:
        image = rescale(image, 255)

    image = image.astype(np.uint8)
    return PIL.Image.fromarray(image, mode=image_mode)


def get_size_with_aspect_ratio(image_size, size, max_size=None) -> tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size.

    Args:
        image_size (`tuple[int, int]`):
            The input image size.
        size (`int`):
            The desired output size.
        max_size (`int`, *optional*):
            The maximum allowed output size.
    """
    height, width = image_size
    raw_size = None
    if max_size is not None:
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        if max_original_size / min_original_size * size > max_size:
            raw_size = max_size * min_original_size / max_original_size
            size = round(raw_size)

    if (height <= width and height == size) or (width <= height and width == size):
        oh, ow = height, width
    elif width < height:
        ow = size
        if max_size is not None and raw_size is not None:
            oh = int(raw_size * height / width)
        else:
            oh = int(size * height / width)
    else:
        oh = size
        if max_size is not None and raw_size is not None:
            ow = int(raw_size * width / height)
        else:
            ow = int(size * width / height)

    return (oh, ow)


# Logic adapted from torchvision resizing logic: https://github.com/pytorch/vision/blob/511924c1ced4ce0461197e5caa64ce5b9e558aab/torchvision/transforms/functional.py#L366
def get_resize_output_image_size(
    input_image: np.ndarray,
    size: int | tuple[int, int] | list[int] | tuple[int, ...],
    default_to_square: bool = True,
    max_size: int | None = None,
    input_data_format: str | ChannelDimension | None = None,
) -> tuple:
    """
    Find the target (height, width) dimension of the output image after resizing given the input image and the desired
    size.

    Args:
        input_image (`np.ndarray`):
            The image to resize.
        size (`int` or `tuple[int, int]` or list[int] or `tuple[int]`):
            The size to use for resizing the image. If `size` is a sequence like (h, w), output size will be matched to
            this.

            If `size` is an int and `default_to_square` is `True`, then image will be resized to (size, size). If
            `size` is an int and `default_to_square` is `False`, then smaller edge of the image will be matched to this
            number. i.e, if height > width, then image will be rescaled to (size * height / width, size).
        default_to_square (`bool`, *optional*, defaults to `True`):
            How to convert `size` when it is a single int. If set to `True`, the `size` will be converted to a square
            (`size`,`size`). If set to `False`, will replicate
            [`torchvision.transforms.Resize`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Resize)
            with support for resizing only the smallest edge and providing an optional `max_size`.
        max_size (`int`, *optional*):
            The maximum allowed for the longer edge of the resized image: if the longer edge of the image is greater
            than `max_size` after being resized according to `size`, then the image is resized again so that the longer
            edge is equal to `max_size`. As a result, `size` might be overruled, i.e the smaller edge may be shorter
            than `size`. Only used if `default_to_square` is `False`.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        `tuple`: The target (height, width) dimension of the output image after resizing.
    """
    if isinstance(size, (tuple, list)):
        if len(size) == 2:
            return tuple(size)
        elif len(size) == 1:
            # Perform same logic as if size was an int
            size = size[0]
        else:
            raise ValueError("size must have 1 or 2 elements if it is a list or tuple")

    if default_to_square:
        return (size, size)

    height, width = get_image_size(input_image, input_data_format)
    short, long = (width, height) if width <= height else (height, width)
    requested_new_short = size

    new_short, new_long = requested_new_short, int(requested_new_short * long / short)

    if max_size is not None:
        if max_size <= requested_new_short:
            raise ValueError(
                f"max_size = {max_size} must be strictly greater than the requested "
                f"size for the smaller edge size = {size}"
            )
        if new_long > max_size:
            new_short, new_long = int(max_size * new_short / new_long), max_size

    return (new_long, new_short) if width <= height else (new_short, new_long)


def resize(
    image: np.ndarray,
    size: tuple[int, int],
    resample: Optional["PILImageResampling"] = None,
    reducing_gap: int | None = None,
    data_format: ChannelDimension | None = None,
    return_numpy: bool = True,
    input_data_format: str | ChannelDimension | None = None,
) -> np.ndarray:
    """
    Resizes `image` to `(height, width)` specified by `size` using the PIL library.

    Args:
        image (`np.ndarray`):
            The image to resize.
        size (`tuple[int, int]`):
            The size to use for resizing the image.
        resample (`int`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            The filter to user for resampling.
        reducing_gap (`int`, *optional*):
            Apply optimization by resizing the image in two steps. The bigger `reducing_gap`, the closer the result to
            the fair resampling. See corresponding Pillow documentation for more details.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the output image. If unset, will use the inferred format from the input.
        return_numpy (`bool`, *optional*, defaults to `True`):
            Whether or not to return the resized image as a numpy array. If False a `PIL.Image.Image` object is
            returned.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        `np.ndarray`: The resized image.
    """

    resample = resample if resample is not None else PILImageResampling.BILINEAR

    if not len(size) == 2:
        raise ValueError("size must have 2 elements")

    # For all transformations, we want to keep the same data format as the input image unless otherwise specified.
    # The resized image from PIL will always have channels last, so find the input format first.
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    data_format = input_data_format if data_format is None else data_format

    # To maintain backwards compatibility with the resizing done in previous image feature extractors, we use
    # the pillow library to resize the image and then convert back to numpy
    do_rescale = False
    if not isinstance(image, PIL.Image.Image):
        do_rescale = _rescale_for_pil_conversion(image)
        image = to_pil_image(image, do_rescale=do_rescale, input_data_format=input_data_format)
    height, width = size
    # PIL images are in the format (width, height)
    resized_image = image.resize((width, height), resample=resample, reducing_gap=reducing_gap)

    if return_numpy:
        resized_image = np.array(resized_image)
        # If the input image channel dimension was of size 1, then it is dropped when converting to a PIL image
        # so we need to add it back if necessary.
        resized_image = np.expand_dims(resized_image, axis=-1) if resized_image.ndim == 2 else resized_image
        # The image is always in channels last format after converting from a PIL image
        resized_image = to_channel_dimension_format(
            resized_image, data_format, input_channel_dim=ChannelDimension.LAST
        )
        # If an image was rescaled to be in the range [0, 255] before converting to a PIL image, then we need to
        # rescale it back to the original range.
        resized_image = rescale(resized_image, 1 / 255) if do_rescale else resized_image
    return resized_image


def normalize(
    image: np.ndarray,
    mean: float | Collection[float],
    std: float | Collection[float],
    data_format: ChannelDimension | None = None,
    input_data_format: str | ChannelDimension | None = None,
) -> np.ndarray:
    """
    Normalizes `image` using the mean and standard deviation specified by `mean` and `std`.

    image = (image - mean) / std

    Args:
        image (`np.ndarray`):
            The image to normalize.
        mean (`float` or `Collection[float]`):
            The mean to use for normalization.
        std (`float` or `Collection[float]`):
            The standard deviation to use for normalization.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the output image. If unset, will use the inferred format from the input.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")

    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)

    channel_axis = get_channel_dimension_axis(image, input_data_format=input_data_format)
    num_channels = image.shape[channel_axis]

    # We cast to float32 to avoid errors that can occur when subtracting uint8 values.
    # We preserve the original dtype if it is a float type to prevent upcasting float16.
    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)

    if isinstance(mean, Collection):
        if len(mean) != num_channels:
            raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(mean)}")
    else:
        mean = [mean] * num_channels
    mean = np.array(mean, dtype=image.dtype)

    if isinstance(std, Collection):
        if len(std) != num_channels:
            raise ValueError(f"std must have {num_channels} elements if it is an iterable, got {len(std)}")
    else:
        std = [std] * num_channels
    std = np.array(std, dtype=image.dtype)

    if input_data_format == ChannelDimension.LAST:
        image = (image - mean) / std
    else:
        image = ((image.T - mean) / std).T

    image = to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
    return image


def center_crop(
    image: np.ndarray,
    size: tuple[int, int],
    data_format: str | ChannelDimension | None = None,
    input_data_format: str | ChannelDimension | None = None,
) -> np.ndarray:
    """
    Crops the `image` to the specified `size` using a center crop. Note that if the image is too small to be cropped to
    the size given, it will be padded (so the returned result will always be of size `size`).

    Args:
        image (`np.ndarray`):
            The image to crop.
        size (`tuple[int, int]`):
            The target size for the cropped image.
        data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use the inferred format of the input image.
        input_data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use the inferred format of the input image.
    Returns:
        `np.ndarray`: The cropped image.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input image must be of type np.ndarray, got {type(image)}")

    if not isinstance(size, Iterable) or len(size) != 2:
        raise ValueError("size must have 2 elements representing the height and width of the output image")

    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    output_data_format = data_format if data_format is not None else input_data_format

    # We perform the crop in (C, H, W) format and then convert to the output format
    image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)

    orig_height, orig_width = get_image_size(image, ChannelDimension.FIRST)
    crop_height, crop_width = size
    crop_height, crop_width = int(crop_height), int(crop_width)

    # In case size is odd, (image_shape[0] + size[0]) // 2 won't give the proper result.
    top = (orig_height - crop_height) // 2
    bottom = top + crop_height
    # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.
    left = (orig_width - crop_width) // 2
    right = left + crop_width

    # Check if cropped area is within image boundaries
    if top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width:
        image = image[..., top:bottom, left:right]
        image = to_channel_dimension_format(image, output_data_format, ChannelDimension.FIRST)
        return image

    # Otherwise, we may need to pad if the image is too small. Oh joy...
    new_height = max(crop_height, orig_height)
    new_width = max(crop_width, orig_width)
    new_shape = image.shape[:-2] + (new_height, new_width)
    new_image = np.zeros_like(image, shape=new_shape)

    # If the image is too small, pad it with zeros
    top_pad = ceil((new_height - orig_height) / 2)
    bottom_pad = top_pad + orig_height
    left_pad = ceil((new_width - orig_width) / 2)
    right_pad = left_pad + orig_width
    new_image[..., top_pad:bottom_pad, left_pad:right_pad] = image

    top += top_pad
    bottom += top_pad
    left += left_pad
    right += left_pad

    new_image = new_image[..., max(0, top) : min(new_height, bottom), max(0, left) : min(new_width, right)]
    new_image = to_channel_dimension_format(new_image, output_data_format, ChannelDimension.FIRST)

    return new_image


def _center_to_corners_format_torch(bboxes_center: "torch.Tensor") -> "torch.Tensor":
    center_x, center_y, width, height = bboxes_center.unbind(-1)
    bbox_corners = torch.stack(
        # top left x, top left y, bottom right x, bottom right y
        [(center_x - 0.5 * width), (center_y - 0.5 * height), (center_x + 0.5 * width), (center_y + 0.5 * height)],
        dim=-1,
    )
    return bbox_corners


def _center_to_corners_format_numpy(bboxes_center: np.ndarray) -> np.ndarray:
    center_x, center_y, width, height = bboxes_center.T
    bboxes_corners = np.stack(
        # top left x, top left y, bottom right x, bottom right y
        [center_x - 0.5 * width, center_y - 0.5 * height, center_x + 0.5 * width, center_y + 0.5 * height],
        axis=-1,
    )
    return bboxes_corners


# 2 functions below inspired by https://github.com/facebookresearch/detr/blob/master/util/box_ops.py
def center_to_corners_format(bboxes_center: TensorType) -> TensorType:
    """
    Converts bounding boxes from center format to corners format.

    center format: contains the coordinate for the center of the box and its width, height dimensions
        (center_x, center_y, width, height)
    corners format: contains the coordinates for the top-left and bottom-right corners of the box
        (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    """
    # Function is used during model forward pass, so we use torch if relevant, without converting to numpy
    if is_torch_tensor(bboxes_center):
        return _center_to_corners_format_torch(bboxes_center)
    elif isinstance(bboxes_center, np.ndarray):
        return _center_to_corners_format_numpy(bboxes_center)

    raise ValueError(f"Unsupported input type {type(bboxes_center)}")


def _corners_to_center_format_torch(bboxes_corners: "torch.Tensor") -> "torch.Tensor":
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = bboxes_corners.unbind(-1)
    b = [
        (top_left_x + bottom_right_x) / 2,  # center x
        (top_left_y + bottom_right_y) / 2,  # center y
        (bottom_right_x - top_left_x),  # width
        (bottom_right_y - top_left_y),  # height
    ]
    return torch.stack(b, dim=-1)


def _corners_to_center_format_numpy(bboxes_corners: np.ndarray) -> np.ndarray:
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = bboxes_corners.T
    bboxes_center = np.stack(
        [
            (top_left_x + bottom_right_x) / 2,  # center x
            (top_left_y + bottom_right_y) / 2,  # center y
            (bottom_right_x - top_left_x),  # width
            (bottom_right_y - top_left_y),  # height
        ],
        axis=-1,
    )
    return bboxes_center


def corners_to_center_format(bboxes_corners: TensorType) -> TensorType:
    """
    Converts bounding boxes from corners format to center format.

    corners format: contains the coordinates for the top-left and bottom-right corners of the box
        (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    center format: contains the coordinate for the center of the box and its the width, height dimensions
        (center_x, center_y, width, height)
    """
    # Inverse function accepts different input types so implemented here too
    if is_torch_tensor(bboxes_corners):
        return _corners_to_center_format_torch(bboxes_corners)
    elif isinstance(bboxes_corners, np.ndarray):
        return _corners_to_center_format_numpy(bboxes_corners)

    raise ValueError(f"Unsupported input type {type(bboxes_corners)}")


def safe_squeeze(
    tensor: Union[np.ndarray, "torch.Tensor"], axis: int | None = None
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Squeezes a tensor, but only if the axis specified has dim 1.
    """
    if axis is None:
        return tensor.squeeze()

    try:
        return tensor.squeeze(axis=axis)
    except ValueError:
        return tensor


# 2 functions below copied from https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
# Copyright (c) 2018, Alexander Kirillov
# All rights reserved.
def rgb_to_id(color):
    """
    Converts RGB color to unique ID.
    """
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id_to_rgb(id_map):
    """
    Converts unique ID to RGB color.
    """
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


class PaddingMode(ExplicitEnum):
    """
    Enum class for the different padding modes to use when padding images.
    """

    CONSTANT = "constant"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    SYMMETRIC = "symmetric"


def pad(
    image: np.ndarray,
    padding: int | tuple[int, int] | Iterable[tuple[int, int]],
    mode: PaddingMode = PaddingMode.CONSTANT,
    constant_values: float | Iterable[float] = 0.0,
    data_format: str | ChannelDimension | None = None,
    input_data_format: str | ChannelDimension | None = None,
) -> np.ndarray:
    """
    Pads the `image` with the specified (height, width) `padding` and `mode`.

    Args:
        image (`np.ndarray`):
            The image to pad.
        padding (`int` or `tuple[int, int]` or `Iterable[tuple[int, int]]`):
            Padding to apply to the edges of the height, width axes. Can be one of three formats:
            - `((before_height, after_height), (before_width, after_width))` unique pad widths for each axis.
            - `((before, after),)` yields same before and after pad for height and width.
            - `(pad,)` or int is a shortcut for before = after = pad width for all axes.
        mode (`PaddingMode`):
            The padding mode to use. Can be one of:
                - `"constant"`: pads with a constant value.
                - `"reflect"`: pads with the reflection of the vector mirrored on the first and last values of the
                  vector along each axis.
                - `"replicate"`: pads with the replication of the last value on the edge of the array along each axis.
                - `"symmetric"`: pads with the reflection of the vector mirrored along the edge of the array.
        constant_values (`float` or `Iterable[float]`, *optional*):
            The value to use for the padding if `mode` is `"constant"`.
        data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use same as the input image.
        input_data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use the inferred format of the input image.

    Returns:
        `np.ndarray`: The padded image.

    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)

    def _expand_for_data_format(values):
        """
        Convert values to be in the format expected by np.pad based on the data format.
        """
        if isinstance(values, (int, float)):
            values = ((values, values), (values, values))
        elif isinstance(values, tuple) and len(values) == 1:
            values = ((values[0], values[0]), (values[0], values[0]))
        elif isinstance(values, tuple) and len(values) == 2 and isinstance(values[0], int):
            values = (values, values)
        elif isinstance(values, tuple) and len(values) == 2 and isinstance(values[0], tuple):
            pass
        else:
            raise ValueError(f"Unsupported format: {values}")

        # add 0 for channel dimension
        values = ((0, 0), *values) if input_data_format == ChannelDimension.FIRST else (*values, (0, 0))

        # Add additional padding if there's a batch dimension
        values = ((0, 0), *values) if image.ndim == 4 else values
        return values

    padding = _expand_for_data_format(padding)

    if mode == PaddingMode.CONSTANT:
        constant_values = _expand_for_data_format(constant_values)
        image = np.pad(image, padding, mode="constant", constant_values=constant_values)
    elif mode == PaddingMode.REFLECT:
        image = np.pad(image, padding, mode="reflect")
    elif mode == PaddingMode.REPLICATE:
        image = np.pad(image, padding, mode="edge")
    elif mode == PaddingMode.SYMMETRIC:
        image = np.pad(image, padding, mode="symmetric")
    else:
        raise ValueError(f"Invalid padding mode: {mode}")

    image = to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
    return image


# TODO (Amy): Accept 1/3/4 channel numpy array as input and return np.array as default
def convert_to_rgb(image: ImageInput) -> ImageInput:
    """
    Converts an image to RGB format. Only converts if the image is of type PIL.Image.Image, otherwise returns the image
    as is.
    Args:
        image (Image):
            The image to convert.
    """

    if not isinstance(image, PIL.Image.Image):
        return image

    if image.mode == "RGB":
        return image

    image = image.convert("RGB")
    return image


def flip_channel_order(
    image: np.ndarray,
    data_format: ChannelDimension | None = None,
    input_data_format: str | ChannelDimension | None = None,
) -> np.ndarray:
    """
    Flips the channel order of the image.

    If the image is in RGB format, it will be converted to BGR and vice versa.

    Args:
        image (`np.ndarray`):
            The image to flip.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format for the output image. Can be one of:
                - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use same as the input image.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format for the input image. Can be one of:
                - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use the inferred format of the input image.
    """
    input_data_format = infer_channel_dimension_format(image) if input_data_format is None else input_data_format

    if input_data_format == ChannelDimension.LAST:
        image = image[..., ::-1]
    elif input_data_format == ChannelDimension.FIRST:
        image = image[::-1, ...]
    else:
        raise ValueError(f"Unsupported channel dimension: {input_data_format}")

    if data_format is not None:
        image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
    return image


def split_to_tiles(images: "torch.Tensor", num_tiles_height: int, num_tiles_width: int) -> "torch.Tensor":
    # Split image into number of required tiles (width x height)
    batch_size, num_channels, height, width = images.size()
    images = images.view(
        batch_size,
        num_channels,
        num_tiles_height,
        height // num_tiles_height,
        num_tiles_width,
        width // num_tiles_width,
    )
    # Permute dimensions to reorder the axes
    image = images.permute(0, 2, 4, 1, 3, 5).contiguous()
    # Reshape into the desired output shape (batch_size * 4, num_channels, width/2, height/2)
    image = image.view(
        batch_size,
        num_tiles_width * num_tiles_height,
        num_channels,
        height // num_tiles_height,
        width // num_tiles_width,
    )
    return image


def divide_to_patches(
    image: Union[np.ndarray, "torch.Tensor"], patch_size: int | tuple[int, int]
) -> list[Union[np.ndarray, "torch.Tensor"]]:
    """
    Divides an image into patches of a specified size.

    Args:
        image (`np.array | "torch.Tensor"`):
            The input image.
        patch_size (`int` or `tuple[int, int]`):
            The size of each patch. If an int, patches are square. If a tuple,
            it is interpreted as `(patch_height, patch_width)`.
    Returns:
        list: A list of `np.array | "torch.Tensor"` representing the patches.
    """
    patch_h, patch_w = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    patches = []
    height, width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
    for i in range(0, height, patch_h):
        for j in range(0, width, patch_w):
            patch = image[..., i : i + patch_h, j : j + patch_w]
            patches.append(patch)

    return patches


def _group_images_by_shape(nested_images, *paired_inputs, is_nested: bool = False):
    """
    Helper function to flatten a single level of nested image and batch structures and group by shape.
    Args:
        nested_images (list):
            A list of images or a single tensor
        paired_inputs (Any, *optional*):
            Zero or more lists that mirror the structure of `nested_images` (flat list, or list of lists when
            `is_nested=True`). Each element is paired 1:1 with the corresponding image so it can be grouped by the
            same shape key. These paired values are grouped alongside `nested_images` but are not stacked in the output, so
            they do not need to be tensors. A paired input can also be `None`, in which case it is grouped as `None`
            for every shape so callers can index it like the others.
        is_nested (bool, *optional*, defaults to False):
            Whether the images are nested.
    Returns:
        tuple[dict, ...]:
            - A dictionary with shape as key and list of images with that shape as value
            - A dictionary with shape as key and list of paired values with that shape as value, or `None` values if
              the corresponding paired input is `None`
            - A dictionary mapping original indices to (shape, index) tuples
            - A dictionary mapping original indices to (shape, index) tuples for each paired input
    """
    grouped_images = defaultdict(list)
    grouped_images_index = {}
    paired_grouped_values = [defaultdict(list) for _ in paired_inputs]

    # Normalize inputs to consistent nested structure, mirroring the images with `None` for missing paired inputs
    normalized_images = [nested_images] if not is_nested else nested_images
    normalized_paired = []
    for paired_input in paired_inputs:
        if paired_input is None:
            normalized_paired.append([[None] * len(sublist) for sublist in normalized_images])
        else:
            normalized_paired.append([paired_input] if not is_nested else paired_input)

    # Process each image and group by shape
    for i, (sublist, *paired_sublists) in enumerate(zip(normalized_images, *normalized_paired)):
        for j, (image, *paired_values) in enumerate(zip(sublist, *paired_sublists)):
            key = (i, j) if is_nested else j
            shape = image.shape[1:]

            # Add to grouped structures
            grouped_images[shape].append(image)
            for paired_index, paired_value in enumerate(paired_values):
                paired_grouped_values[paired_index][shape].append(paired_value)
            grouped_images_index[key] = (shape, len(grouped_images[shape]) - 1)

    # Store structure size for nested inputs to handle empty sublists during reconstruction
    if is_nested:
        grouped_images_index["_num_sublists"] = len(normalized_images)

    # Missing paired inputs are grouped as a single `None` per shape instead of a list of `None` values
    paired_grouped_values = [
        dict.fromkeys(grouped_images, None) if paired_input is None else grouped_values
        for paired_input, grouped_values in zip(paired_inputs, paired_grouped_values)
    ]

    return grouped_images, *paired_grouped_values, grouped_images_index


def _reconstruct_nested_structure(indices, processed_images):
    """Helper function to reconstruct a single level nested structure."""
    # Get the number of sublists (handles empty sublists like in [[], [image]])
    num_sublists = indices.pop("_num_sublists", None)

    # Group indices by outer index
    nested_indices = defaultdict(list)
    for i, j in indices:
        nested_indices[i].append(j)

    # Determine the number of outer sublists
    if num_sublists is not None:
        max_outer_idx = num_sublists - 1
    elif nested_indices:
        max_outer_idx = max(nested_indices.keys())
    else:
        return []

    # Create the result structure
    result = []
    for i in range(max_outer_idx + 1):
        if i not in nested_indices:
            result.append([])
        else:
            inner_max_idx = max(nested_indices[i])
            inner_list = [None] * (inner_max_idx + 1)
            for j in nested_indices[i]:
                shape, idx = indices[(i, j)]
                inner_list[j] = processed_images[shape][idx]
            result.append(inner_list)

    return result


def _iterate_items(items, is_nested: bool):
    """
    Helper function to iterate over items yielding (key, item) pairs.

    For nested structures, yields ((row_index, col_index), item).
    For flat structures, yields (index, item).
    """
    if is_nested:
        for i, row in enumerate(items):
            for j, item in enumerate(row):
                yield (i, j), item
    else:
        for i, item in enumerate(items):
            yield i, item


def _get_device_from_images(images, is_nested: bool) -> "torch.device":
    """
    Get the device from the first non-empty element in a (potentially nested) list of images.

    Handles cases like `images = [[], [image]]` where the first sublist may be empty.
    """
    if is_nested:
        for row in images:
            if isinstance(row, torch.Tensor):
                return row.device
            if isinstance(row, list) and len(row) > 0:
                return row[0].device
    return images[0].device


def group_images_by_shape(
    images: Union[list["torch.Tensor"], "torch.Tensor"],
    *paired_inputs,
    disable_grouping: bool | None,
    is_nested: bool = False,
) -> tuple[dict, ...]:
    """
    Groups images by shape.
    Returns a dictionary with the shape or index as key and a list of images with that shape as value,
    and a dictionary with the index of the image in the original list as key and the shape and index in the grouped list as value.

    The function supports both flat lists of tensors and nested structures.
    The input must be either all flat or all nested, not a mix of both.

    Args:
        images (Union[list["torch.Tensor"], "torch.Tensor"]):
            A list of images or a single tensor
        paired_inputs (Any, *optional*):
            Zero or more lists that mirror the structure of `images` (flat list, or list of lists when
            `is_nested=True`). Each element is paired 1:1 with the corresponding image so it can be grouped by the
            same shape key. These paired values are grouped alongside `images` but are not stacked in the output, so
            they do not need to be tensors. A paired input can also be `None`, in which case it is grouped as `None`
            for every shape so callers can index it like the others.
        disable_grouping (bool):
            Whether to disable grouping. If None, will be set to True if the images are on CPU, and False otherwise.
            This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157
        is_nested (bool, *optional*, defaults to False):
            Whether the images are nested.

    Returns:
        tuple[dict, ...]:
            - A dictionary with shape or index as key and list/batch of images with that shape as value.
              The key is the shape of the images if `disable_grouping` evaluates to True, and the index
              of the image in the original list otherwise.
            - Zero or more dictionaries (one per argument in `*paired_inputs`) grouped consistently with `images`; these carry
              the corresponding per-item values and are not stacked, or `None` values if the corresponding paired
              input is `None`
            - A dictionary mapping original indices to (shape, index) tuples
    """
    # If disable grouping is not explicitly provided, we favor disabling it if the images are on CPU, and enabling it otherwise.
    if disable_grouping is None:
        device = _get_device_from_images(images, is_nested)
        disable_grouping = device.type == "cpu"

    if disable_grouping:
        grouped_images_index = {key: (key, 0) for key, _ in _iterate_items(images, is_nested)}
        if is_nested:
            grouped_images_index["_num_sublists"] = len(images)

        grouped_images = {key: img.unsqueeze(0) for key, img in _iterate_items(images, is_nested)}
        paired_grouped_values = [
            dict.fromkeys(grouped_images, None)
            if paired_list is None
            else {key: [item] for key, item in _iterate_items(paired_list, is_nested)}
            for paired_list in paired_inputs
        ]

        return grouped_images, *paired_grouped_values, grouped_images_index

    # A batched tensor is already a single group of images sharing the same shape, no need to unbind and re-stack it.
    if isinstance(images, torch.Tensor) and not is_nested and len(images) > 0:
        shape = images.shape[2:]
        return (
            {shape: images},
            *[{shape: list(paired_list)} for paired_list in paired_inputs],
            {index: (shape, index) for index in range(len(images))},
        )

    # Handle single level nested structure
    grouped_images, *paired_grouped_values, grouped_images_index = _group_images_by_shape(
        images, *paired_inputs, is_nested=is_nested
    )

    # Stack images with the same shape. Groups holding a single image are unsqueezed instead, as stacking would copy
    # the image for no reason.
    grouped_images = {
        shape: images_list[0].unsqueeze(0) if len(images_list) == 1 else torch.stack(images_list, dim=0)
        for shape, images_list in grouped_images.items()
    }

    return grouped_images, *paired_grouped_values, grouped_images_index


def reorder_images(
    processed_images: dict[tuple[int, int], "torch.Tensor"],
    grouped_images_index: dict[int | tuple[int, int], tuple[tuple[int, int], int]],
    is_nested: bool = False,
) -> Union[list["torch.Tensor"], "torch.Tensor"]:
    """
    Reconstructs images in the original order, preserving the original structure (nested or not).
    The input structure is either all flat or all nested.

    Args:
        processed_images (dict[tuple[int, int], "torch.Tensor"]):
            Dictionary mapping shapes to batched processed images.
        grouped_images_index (dict[Union[int, tuple[int, int]], tuple[tuple[int, int], int]]):
            Dictionary mapping original indices to (shape, index) tuples.
        is_nested (bool, *optional*, defaults to False):
            Whether the images are nested. Cannot be inferred from the input, as some processing functions outputs nested images.
            even with non nested images, e.g. functions splitting images into patches. We thus can't deduce is_nested from the input.


    Returns:
        Union[list["torch.Tensor"], "torch.Tensor"]:
            Images in the original structure.
    """
    if not is_nested:
        return [
            processed_images[grouped_images_index[i][0]][grouped_images_index[i][1]]
            for i in range(len(grouped_images_index))
        ]

    return _reconstruct_nested_structure(grouped_images_index, processed_images)
