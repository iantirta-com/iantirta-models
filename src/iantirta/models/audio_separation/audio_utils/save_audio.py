
import struct
import subprocess
from pathlib import Path
from typing import Literal, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


__all__ = [
    "save_audio"
]


def prevent_clip(wav: "torch.Tensor", mode='rescale'):
    """
    different strategies for avoiding raw clipping.
    """
    if mode is None or mode == 'none':
        return wav

    assert wav.dtype.is_floating_point, "too late for clipping"

    if mode == 'rescale':
        wav = wav / max(1.01 * wav.abs().max(), 1)
    
    elif mode == 'clamp':
        wav = wav.clamp(-0.99, 0.99)

    elif mode == 'tanh':
        import torch
        
        wav = torch.tanh(wav)

    else:
        raise ValueError(f"Invalid mode {mode}")

    return wav


def i16_pcm(wav: "torch.Tensor"):
    """Convert audio to 16 bits integer PCM format."""
    if wav.dtype.is_floating_point:
        return (wav.clamp_(-1, 1) * (2**15 - 1)).short()
    else:
        return wav


def encode_mp3(
    wav: "torch.Tensor",
    outpath: str | Path,
    samplerate: int = 44100,
    bitrate: int = 320,
    quality: Literal[2, 3, 4, 5, 6, 7] = 2,
    verbose: bool = False
):
    """Save given audio as mp3. This should work on all OSes."""
    import lameenc

    C, T = wav.shape
    wav = i16_pcm(wav)

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(samplerate)
    encoder.set_channels(C)
    encoder.set_quality(quality)  # 2-highest, 7-fastest

    if not verbose:
        encoder.silence()

    wav = wav.data.cpu()
    wav = wav.transpose(0, 1).numpy()

    mp3_data = encoder.encode(wav.tobytes())
    mp3_data += encoder.flush()

    with open(outpath, "wb") as f:
        f.write(mp3_data)


def _pcm_bytes(
    wav: "torch.Tensor",
    bits_per_sample: int
) -> bytes:
    """Convert a `[C, T]` float tensor to raw interleaved little-endian PCM bytes.
    24 bits samples are stored over the top bits of an int32, so that the low byte
    can either be stripped (wav) or truncated by the encoder (ffmpeg flac)."""
    frames = wav.clamp(-1, 1).t().contiguous().cpu().numpy()

    if bits_per_sample == 16:
        return (frames * (2**15 - 1)).astype('<i2').tobytes()

    elif bits_per_sample == 24:
        ints = ((frames * (2**23 - 1)).astype('<i4') << 8).view(np.uint8)
        return ints.tobytes()

    else:
        raise ValueError(f"Invalid bits_per_sample: {bits_per_sample}")


def encode_wav(
    wav: "torch.Tensor",
    outpath: str | Path,
    samplerate: int,
    bits_per_sample: int = 16,
    as_float: bool = False
):
    """Save the given `[C, T]` float tensor as a wav file, either as 16 / 24 bits PCM,
    or as float32 (`as_float=True`)."""
    channels, length = wav.shape
    if as_float:
        audio_format = 3  # IEEE float
        bits_per_sample = 32
        data = wav.clamp(-1, 1).t().cpu().numpy().astype('<f4').tobytes()

    else:
        audio_format = 1  # integer PCM
        data = _pcm_bytes(wav, bits_per_sample)
        if bits_per_sample == 24:
            # Keep only the top 3 bytes of each little-endian int32 sample.
            data = np.frombuffer(data, dtype=np.uint8).reshape(-1, 4)[:, 1:].tobytes()

    byte_rate = samplerate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    fmt = struct.pack('<HHIIHH', audio_format, channels, samplerate,
                      byte_rate, block_align, bits_per_sample)
    chunks = [(b'fmt ', fmt)]

    if audio_format == 3:
        chunks.append((b'fact', struct.pack('<I', length)))

    chunks.append((b'data', data))
    riff_size = 4 + sum(8 + len(content) for _, content in chunks)

    with open(outpath, 'wb') as file:
        file.write(b'RIFF' + struct.pack('<I', riff_size) + b'WAVE')
        for name, content in chunks:
            file.write(name + struct.pack('<I', len(content)))
            file.write(content)


def encode_ffmpeg(
    wav: "torch.Tensor",
    outpath: str | Path,
    samplerate: int,
    bits_per_sample: int = 16
):
    """Save the given `[C, T]` float tensor with ffmpeg, in the format given
    by the extension of `path` (e.g. flac, which sphn can only decode)."""
    channels, _ = wav.shape
    input_format = 's16le' if bits_per_sample == 16 else 's32le'
    command = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', input_format, '-ar', str(samplerate), '-ac', str(channels), '-i', '-',
        str(outpath)
    ]
    try:
        subprocess.run(command, input=_pcm_bytes(wav, bits_per_sample), check=True)

    except FileNotFoundError:
        raise RuntimeError(f"Saving as {Path(outpath).suffix} requires ffmpeg to be installed.")


def save_audio(
    audio: "torch.Tensor | np.ndarray",
    outpath: str | Path,
    *,
    samplerate: int,
    bitrate: int = 320,
    bits_per_sample: Literal[16, 24, 32] = 16,
    clip: Literal["rescale", "clamp", "tanh", "none"] = 'rescale',
    # .wav specific
    as_float: bool = False,
    # .mp3 specific
    preset: Literal[2, 3, 4, 5, 6, 7] = 2
) -> Path:
    """Save audio file, automatically preventing clipping
    if necessary based on the given `clip` strategy.
    
    If the path ends in `.mp3`, this will save as mp3 with the given `bitrate`.
    Use `preset` to set mp3 quality:
        2 for highest quality,
        7 for fastest speed
    """
    import torch

    if not isinstance(audio, (torch.Tensor, np.ndarray)):
        raise TypeError(f"Invalud audio type: {type(audio)}")

    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    
    audio = prevent_clip(audio, mode=clip)

    outpath = Path(outpath)
    suffix = outpath.suffix.lower()

    if suffix == ".mp3":
        encode_mp3(audio, outpath, samplerate, bitrate, preset, verbose=True)

    elif suffix == ".wav":
        encode_wav(audio, outpath, samplerate, bits_per_sample=bits_per_sample, as_float=as_float)

    elif suffix == ".flac":
        encode_ffmpeg(audio, outpath, samplerate, bits_per_sample=bits_per_sample)

    else:
        raise ValueError(f"Invalid suffix for path: {suffix}")


if __name__ == "__main__":
    audio_path = Path("../dataset/mahalini.wav")
    with audio_path.open("rb") as audio_file:
        audio_bytes = audio_file.read()
    print(save_audio(
        audio_bytes, "test.tmp"
    ))
