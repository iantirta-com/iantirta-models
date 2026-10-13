import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import numpy as np

if TYPE_CHECKING:
    pass

__all__ = [
    "load_audio",
]


logger = logging.getLogger(__name__)


TORCHCODEC_ONLY_FILETYPES = frozenset({
    "3gp", "aac", "ac3", "amr", "avi",
    "flv", "m4a", "m4v", "mkv", "mov", "mp4",
    "mpg", "ogv", "sox", "ts", "webm", "wma",
    "wmv", "wv",
})


@lru_cache
def _is_torchcodec_available() -> bool:
    try:
        import importlib
        import importlib.metadata

        from packaging import version

        torch_codec_min_version = version.parse("0.3.0")

        spec = importlib.util.find_spec("torchcodec")
        if not spec:
            return False

        torchcodec_version = version.parse(importlib.metadata.version("torchcodec"))

        if torch_codec_min_version > torchcodec_version:
            return False

        import torchcodec
        return True

    except (ModuleNotFoundError, ImportError, OSError) as e:
        logger.debug(f"torchcodec unavailable or failed to load: {e}")
        return False
    else:
        return False


def _fetch_audio_bytes(url: str, timeout: float | None = 10.0) -> bytes:
    """Fetch audio bytes from a URL."""
    import requests
    response = requests.get(url, follow_redirects=True, timeout=timeout)
    response.raise_for_status()
    return response.content


def resolve_audio_from_source(source: str):
    """Resolve an audio source string to a local file path or raw bytes for a decoder.

    Accepts `http(s)://` URLs (fetched with retry), local file paths (returned unchanged),
    and base64 strings (optionally wrapped as a `data:...` URI).
    """
    if source.startswith(("http://", "https://")):
        return _fetch_audio_bytes(source,)
    if os.path.isfile(source):
        return source
    # Not a URL or a local path — assume base64, optionally wrapped as a `data:<media-type>;base64,` URI
    if source.startswith("data:"):
        source = source.split(",", 1)[1]
    try:
        import base64
        return base64.b64decode(source)
    except Exception as e:
        raise ValueError(
            "Incorrect audio source. Must be a valid URL starting with `http://` or `https://`, "
            f"a valid path to an audio file, or a base64 encoded string. Got {source}. Failed with {e}"
        )


def get_file_type_from_source(source: str) -> str:
    """Best-effort format token from the source *string* — the file extension (paths and URLs) or
        the media subtype (`data:` URIs) — without resolving or decoding it. Returns None when the
        string carries no hint, e.g. a raw base64 payload."""
    if source.startswith("data:"):
        media_type = source[len("data:") :].split(",", 1)[0].split(";", 1)[0]
        return media_type.rpartition("/")[2].removeprefix("x-") or None

    path = urlparse(source).path if source.startswith(("http://", "https://")) else source
    return os.path.splitext(path)[1].lstrip(".").lower() or None


def get_file_type_from_bytes(data: bytes) -> str:
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
            "3gp" if brand[:3] == b"3gp"
            else "m4a" if brand[:3] == b"M4A"
            else "mov" if brand[:2] == b"qt"
            else "mp4"
        )

    if head[:4] == b"RIFF" and head[8:12] in (b"WAVE", b"AVI "):
        return "wav" if head[8:12] == b"WAVE" else "avi"

    if head[:4] == b"riff" and head[4:8] == bytes.fromhex("2e91cf11"):  # Wave64
        return "w64"

    if head[:4] == bytes.fromhex("1a45dfa3"):  # EBML: Matroska vs WebM
        return "webm" if b"webm" in head else "mkv"

    if head[:4] == b"OggS":  # OGG: Opus / Theora (ogv) / Vorbis (ogg)
        page = data[:128]
        return (
            "opus" if b"OpusHead" in page
            else "ogv" if b"theora" in page
            else "ogg"
        )

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


def load_audio(
    audio: str | Path | np.ndarray,
    sampling_rate: int | None = None,
    num_channels: int | None = None,
    backend: str = "auto",
) -> np.ndarray:

    if isinstance(audio, np.ndarray):
        return audio

    if not isinstance(audio, (str, Path)):
        raise TypeError(
            "Incorrect format used for `audio`. Should be a numpy array or a `str`: an `http(s)://` URL, "
            "a local file path, or a base64-encoded string (optionally wrapped as a `data:...` URI)."
        )

    audio = str(audio)
    filetype = get_file_type_from_source(audio)

    audio_source = resolve_audio_from_source(audio)

    use_torchcodec = False
    if backend == "torchcodec" or (backend == "auto" and filetype in TORCHCODEC_ONLY_FILETYPES):
        use_torchcodec = _is_torchcodec_available()

    if isinstance(audio_source, bytes) and filetype is None and not use_torchcodec:
        try:
            filetype = get_file_type_from_bytes(audio_source)

        except ValueError:
            filetype = None

    if use_torchcodec:
        from torchcodec.decoders import AudioDecoder

        audio_samples = AudioDecoder(
            audio_source,
            sample_rate=sampling_rate,
            num_channels=num_channels
        ).get_all_samples()
        return audio_samples.data[0].numpy(), sampling_rate if sampling_rate is not None else audio_samples.sample_rate
    
    elif not use_torchcodec and filetype in TORCHCODEC_ONLY_FILETYPES:
        from ._audio import AudioLoader

        audio_samples = AudioLoader(audio_source)
        return audio_samples.read(
            streams=0, sampling_rate=sampling_rate, num_channels=num_channels
        ).numpy(), sampling_rate if sampling_rate is not None else audio_samples.samplerate()

    else:
        if backend == "torchaudio": # Specific only, i think its already deprecated
            from io import BytesIO

            import torchaudio
            wav, src_sr = torchaudio.load(BytesIO(audio_source) if isinstance(audio_source, bytes) else audio_source)

            if sampling_rate and sampling_rate != src_sr:
                wav = torchaudio.functional.resample(wav, orig_freq=src_sr, new_freq=sampling_rate)

            if num_channels == 1:
                wav = wav.mean(dim=0) # To mono

            return wav.numpy().astype(np.float32), sampling_rate if sampling_rate is not None else src_sr

        try:
            import librosa

            return librosa.load(
                BytesIO(audio_source) if isinstance(audio_source, bytes)
                else audio_source,
                sr=sampling_rate,
                mono=num_channels==1
            )
        except Exception:
            from ._audio import AudioLoader
            
            audio_samples = AudioLoader(audio_source)
            return audio_samples.read(
                streams=0, sampling_rate=sampling_rate, num_channels=num_channels
            ).numpy(), sampling_rate if sampling_rate is not None else audio_samples.samplerate()

    raise ValueError()


if __name__ == "__main__":
    video_path = Path("../dataset/mahalini.mp4")
    audio_path = Path("../dataset/mahalini.wav")
    assert audio_path.is_file()

    load_audio(audio_path)