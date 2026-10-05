
import base64
import importlib
import importlib.metadata
import io
import os
import warnings
from collections.abc import Sequence
from io import BytesIO
from typing import TYPE_CHECKING, Any, Union
from urllib.parse import urlparse

import numpy as np
from packaging import version

from iantirta.models.common.tokenization_utils import AudioInput
from iantirta.models.tools._torch import (
    is_torchaudio_available,
    is_torchcodec_available,
)
from iantirta.models.tools.tensor import is_numpy_array, is_torch_tensor


def is_valid_audio(audio):
    return (
        is_numpy_array(audio)
        or is_torch_tensor(audio)
        or (isinstance(audio, (list, tuple)) and isinstance(audio[0], float))
    )


def is_valid_list_of_audio(audio):
    return audio and all(is_valid_audio(audio_i) for audio_i in audio)


def make_list_of_audio(
    audio: list[AudioInput] | AudioInput,
) -> AudioInput:
    """
    Ensure that the output is a list of audio.
    Args:
        audio (`Union[list[AudioInput], AudioInput]`):
            The input audio.
    Returns:
        list: A list of audio.
    """
    # If it's a list of audios, it's already in the right format
    if isinstance(audio, (list, tuple)) and is_valid_list_of_audio(audio):
        return audio

    # If it's a single audio, convert it to a list of
    if is_valid_audio(audio):
        return [audio]

    raise ValueError("Invalid input type. Must be a single audio or a list of audio")


def make_list_of_audio_chat_template(
    audio: list[AudioInput] | AudioInput | str | list[str],
) -> AudioInput:
    """
    Ensure that the output is a list of audio. Unlike `make_list_of_audio`, this function also accepts a URL string or
    local path, as accepted by chat templates.

    Args:
        audio (`Union[list[AudioInput], AudioInput]`):
            The input audio. Can be a URL string, local path, numpy/torch array,  or a list of these.
    Returns:
        list: A list of audio.
    """

    # Handle string inputs
    if isinstance(audio, str):
        return [audio]
    if isinstance(audio, (list, tuple)) and audio and all(isinstance(a, str) for a in audio):
        return list(audio)

    # Handle numpy/torch array inputs
    return make_list_of_audio(audio)


def make_audio_chat_template_content(audio_item) -> dict:
    """
    Build a chat-template content dict for a single audio item.

    Args:
        audio_item (`str` or array-like):
            A single audio item. Strings are treated as local paths or URLs; other values (numpy/torch arrays) are
            forwarded directly.

    Returns:
        `dict`: A chat-template content dict, e.g. `{"type": "audio", "path": ...}` for strings or
        `{"type": "audio", "audio": ...}` otherwise.
    """
    if isinstance(audio_item, str):
        return {"type": "audio", "path": audio_item}
    return {"type": "audio", "audio": audio_item}


def resolve_language(language: str | None, code_to_name: dict[str, str], return_code: bool = True) -> str | None:
    """
    Map a language code or name to its canonical form, with validation.

    Accepts either a language code (e.g. ``"zh"``, ``"en"``) or a full name (e.g. ``"Chinese"``, ``"English"``) and
    returns the canonical code or name depending on ``return_code``. ``None`` passes through unchanged (auto-detect).

    Args:
        language (`str` or `None`):
            The language code or full name to resolve. ``None`` is returned unchanged.
        code_to_name (`dict[str, str]`):
            Mapping from language code to full language name for the model's supported languages.
        return_code (`bool`, *optional*, defaults to `True`):
            Whether to return the canonical language ``code``. If ``False``, returns the full language ``name``.

    Returns:
        `str` or `None`: The canonical language code or name, or ``None`` if ``language`` is ``None``.

    Raises:
        `ValueError`: If the language is not recognized.
    """
    if language is None:
        return None

    language_lower = language.lower()
    # Try code lookup first, then full-name lookup (both case-insensitive)
    for code, name in code_to_name.items():
        if language_lower == code.lower() or language_lower == name.lower():
            return code if return_code else name

    raise ValueError(
        f"Unsupported language: {language!r}. Use a language code "
        f"(e.g. 'en', 'zh') or full name (e.g. 'English', 'Chinese'). "
        f"Supported codes: {sorted(code_to_name.keys())}. "
        f"Supported names: {sorted(set(code_to_name.values()))}."
    )


def prepare_language_inputs(
    language: str | list[str] | None,
    batch_size: int,
    code_to_name: dict[str, str],
    allow_broadcast: bool = False,
    return_code: bool = True,
) -> list[str | None]:
    """
    Broadcast and validate a language argument to match ``batch_size``.

    Accepts language codes (e.g. ``"zh"``, ``"en"``) or full names (e.g. ``"Chinese"``, ``"English"``). Each value is
    resolved to its canonical form via [`resolve_language`].

    Args:
        language (`str`, `list[str]`, or `None`):
            The language hint(s). A single value is broadcast to the whole batch; a list must match ``batch_size``
            (unless ``allow_broadcast`` is set). ``None`` disables language hints for the whole batch.
        batch_size (`int`):
            The number of samples in the batch.
        code_to_name (`dict[str, str]`):
            Mapping from language code to full language name for the model's supported languages.
        allow_broadcast (`bool`, *optional*, defaults to `False`):
            Whether a single-element list may be broadcast to the whole batch.
        return_code (`bool`, *optional*, defaults to `True`):
            Whether to return canonical language ``code``s. If ``False``, returns full language ``name``s.

    Returns:
        `list[str | None]`: The resolved language for each sample.
    """
    if language is None:
        return [None] * batch_size
    if isinstance(language, str):
        return [resolve_language(language, code_to_name, return_code)] * batch_size
    if isinstance(language, (list, tuple)):
        if allow_broadcast and len(language) == 1 and batch_size > 1:
            return [resolve_language(language[0], code_to_name, return_code)] * batch_size
        if len(language) != batch_size:
            raise ValueError(f"Got {len(language)} language(s) for {batch_size} sample(s); counts must match.")
        return [resolve_language(lang, code_to_name, return_code) for lang in language]
    raise TypeError("`language` must be a string, a list of strings, or `None`.")


# Common Audio Utility

if is_torchcodec_available():
    TORCHCODEC_VERSION = version.parse(importlib.metadata.version("torchcodec"))


_NEEDS_TORCHCODEC = "Install torchcodec>=0.3.0 (`pip install torchcodec`) to load audio from this source."


TORCHCODEC_ONLY_FILETYPES = frozenset(
    {
        "3gp",
        "aac",
        "ac3",
        "amr",
        "avi",
        "flv",
        "m4a",
        "m4v",
        "mkv",
        "mov",
        "mp4",
        "mpg",
        "ogv",
        "sox",
        "ts",
        "webm",
        "wma",
        "wmv",
        "wv",
    }
)


def _fetch_audio_bytes(url: str, timeout: float | None = 10.0) -> bytes:
    raise NotImplementedError()


def _format_from_source(audio: str) -> "str | None":
    """Best-effort format token from the source *string* — the file extension (paths and URLs) or
    the media subtype (`data:` URIs) — without resolving or decoding it. Returns None when the
    string carries no hint, e.g. a raw base64 payload."""
    if audio.startswith("data:"):
        media_type = audio[len("data:") :].split(",", 1)[0].split(";", 1)[0]
        return media_type.rpartition("/")[2].removeprefix("x-") or None
    path = urlparse(audio).path if audio.startswith(("http://", "https://")) else audio
    return os.path.splitext(path)[1].lstrip(".").lower() or None


def get_audio_filetype(data: bytes) -> str:
    """Identify a file's container/codec from its magic bytes.

    A few extensions are byte-identical in their headers and collapse to a canonical type:
    ``wavex`` -> ``wav`` and ``m4v``/``hevc.mp4`` -> ``mp4`` (all carry the ``isom`` ftyp brand).

    Raises ValueError if the bytes match no supported filetype.
    """
    head = data[:64]

    # Containers that host several filetypes -> sniff a bit deeper.
    if head[4:8] == b"ftyp":  # ISO-BMFF: m4v & hevc share the 'isom' brand -> mp4
        brand = head[8:12]
        return (
            "3gp" if brand[:3] == b"3gp" else "m4a" if brand[:3] == b"M4A" else "mov" if brand[:2] == b"qt" else "mp4"
        )
    if head[:4] == b"RIFF" and head[8:12] in (b"WAVE", b"AVI "):
        return "wav" if head[8:12] == b"WAVE" else "avi"
    if head[:4] == b"riff" and head[4:8] == bytes.fromhex("2e91cf11"):  # Wave64
        return "w64"
    if head[:4] == bytes.fromhex("1a45dfa3"):  # EBML: Matroska vs WebM
        return "webm" if b"webm" in head else "mkv"
    if head[:4] == b"OggS":  # OGG: Opus / Theora (ogv) / Vorbis (ogg)
        page = data[:128]
        return "opus" if b"OpusHead" in page else "ogv" if b"theora" in page else "ogg"
    if head[:16] == bytes.fromhex("3026b2758e66cf11a6d900aa0062ce6c"):  # ASF
        return "wmv" if bytes.fromhex("c0ef19bc4d5bcf11a8fd00805f5c442b") in data else "wma"
    if head[:1] == b"\xff" and len(head) > 1 and head[1] & 0xE0 == 0xE0:  # MPEG/AAC sync
        if head[1] & 0xF6 == 0xF0:  # ADTS layer bits 00 -> AAC
            return "aac"
        layer = head[1] >> 1 & 0x3  # MPEG audio layer field (II -> mp2, III -> mp3)
        if layer in (0b10, 0b01):
            return "mp2" if layer == 0b10 else "mp3"
    if head[:1] == b"\x47" and len(data) > 188 and data[188] == 0x47:
        return "ts"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"

    # Single fixed-signature formats, keyed by their leading bytes.
    signatures = {
        b"fLaC": "flac",
        b"RF64": "rf64",
        b"caff": "caf",
        b".snd": "au",
        b"#!AMR": "amr",
        b"wvpk": "wv",
        b".SoX": "sox",
        b"XoS.": "sox",
        b"Creative Voice File": "voc",
        b"\x64\xa3\x01\x00": "sf",
        b"\x00\x01\xa3\x64": "sf",
        b"\x0b\x77": "ac3",
        b"\x00\x00\x01\xba": "mpg",
        b"FLV": "flv",
        b"ID3": "mp3",
    }
    for sig, filetype in signatures.items():
        if head.startswith(sig):
            return filetype

    raise ValueError("not supported filetype")


def _resolve_audio_source(audio: str, timeout: float | None = None) -> "str | bytes":
    """Resolve an audio source string to a local file path or raw bytes for a decoder.

    Accepts `http(s)://` URLs (fetched with retry), local file paths (returned unchanged),
    and base64 strings (optionally wrapped as a `data:...` URI).
    """
    if audio.startswith(("http://", "https://")):
        return _fetch_audio_bytes(audio, timeout=timeout)
    if os.path.isfile(audio):
        return audio
    # Not a URL or a local path — assume base64, optionally wrapped as a `data:<media-type>;base64,` URI
    if audio.startswith("data:"):
        audio = audio.split(",", 1)[1]
    try:
        return base64.b64decode(audio)
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            "Incorrect audio source. Must be a valid URL starting with `http://` or `https://`, "
            f"a valid path to an audio file, or a base64 encoded string. Got {audio}. Failed with {e}"
        )


def load_audio(audio: str | np.ndarray, sampling_rate=16000, timeout=None, backend: str = "auto") -> np.ndarray:
    """
    Loads `audio` to an np.ndarray object.

    Args:
        audio (`str` or `np.ndarray`):
            The audio to be loaded to the numpy array format. If a `str`, it can be an `http(s)://`
            URL, a local file path, or a base64-encoded string (optionally wrapped as a
            `data:<media-type>;base64,` URI).
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate to be used when loading the audio. It should be same as the
            sampling rate the model you will be using further was trained with.
        timeout (`float`, *optional*):
            The timeout value in seconds for the URL request.
        backend (`str`, *optional*, defaults to `"auto"`):
            Decoding backend: `"auto"` uses torchcodec when available (>=0.3.0) and falls back to
            librosa; `"torchcodec"`, `"librosa"` or `"torchaudio"` force that backend (and error if it
            is missing). `"torchaudio"` decodes with `torchaudio.load` and resamples with
            `torchaudio.functional.resample` (matches serving stacks such as sglang bit-for-bit).

    Returns:
        `np.ndarray`: A numpy array representing the audio.
    """
    if isinstance(audio, np.ndarray):
        return audio
    if not isinstance(audio, str):
        raise TypeError(
            "Incorrect format used for `audio`. Should be a numpy array or a `str`: an `http(s)://` URL, "
            "a local file path, or a base64-encoded string (optionally wrapped as a `data:...` URI)."
        )

    # torchcodec handles audio/video; librosa only plain audio. `backend` lets callers pin one.
    if backend == "auto":
        resolved_backend = (
            "torchcodec" if is_torchcodec_available() and version.parse("0.3.0") <= TORCHCODEC_VERSION else "librosa"
        )
    elif backend in ("torchcodec", "librosa", "torchaudio"):
        resolved_backend = backend
    else:
        raise ValueError(f"Unknown backend {backend!r}; expected 'auto', 'torchcodec', 'librosa', or 'torchaudio'.")
    # soundfile-based backends (librosa / torchaudio) cannot decode the video-ish formats below.
    use_torchcodec = resolved_backend == "torchcodec"

    # 1. Identify the format from the source string (extension / `data:` media type), without fetching.
    filetype = _format_from_source(audio)
    # 2. With librosa as the only backend, fail fast and clearly on a format it cannot decode.
    if not use_torchcodec and filetype in TORCHCODEC_ONLY_FILETYPES:
        raise RuntimeError(
            f"The audio source is a '{filetype}' file, which librosa cannot decode. {_NEEDS_TORCHCODEC}"
        )

    # 3. Resolve to local path or bytes; sniff format for raw base64 payloads before passing to librosa.
    source = _resolve_audio_source(audio, timeout=timeout)
    if not use_torchcodec and filetype is None and isinstance(source, bytes):
        try:
            filetype = get_audio_filetype(source)
        except ValueError:
            filetype = None
        if filetype in TORCHCODEC_ONLY_FILETYPES:
            raise RuntimeError(
                f"The audio source is a '{filetype}' file, which librosa cannot decode. {_NEEDS_TORCHCODEC}"
            )

    # 4. Decode with the selected backend (`requires_backends` raises a clear error if it is missing).
    if use_torchcodec:
        from torchcodec.decoders import AudioDecoder

        # `num_channels=1` matches what most models expect and librosa's default.
        return AudioDecoder(source, sample_rate=sampling_rate, num_channels=1).get_all_samples().data[0].numpy()

    if resolved_backend == "torchaudio" and is_torchaudio_available():
        import torchaudio

        waveform, src_sampling_rate = torchaudio.load(BytesIO(source) if isinstance(source, bytes) else source)
        waveform = waveform.mean(dim=0)  # to mono

        if src_sampling_rate != sampling_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=src_sampling_rate, new_freq=sampling_rate)
        return waveform.numpy().astype(np.float32)

    import librosa
    return librosa.load(BytesIO(source) if isinstance(source, bytes) else source, sr=sampling_rate)[0]
