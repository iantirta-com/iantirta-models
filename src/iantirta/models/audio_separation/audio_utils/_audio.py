import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch


@contextmanager
def temp_filenames(count: int, delete=True):
    names = []
    try:
        for _ in range(count):
            names.append(tempfile.NamedTemporaryFile(delete=False).name)
        yield names
    finally:
        if delete:
            for name in names:
                os.unlink(name)


@lru_cache
def _read_metadata(path):
    stdout_data = subprocess.check_output([
        'ffprobe', "-loglevel", "panic",
        str(path), '-print_format', 'json', '-show_format', '-show_streams'
    ])
    return json.loads(stdout_data.decode('utf-8'))


@dataclass(slots=True)
class AudioLoader:
    """
    Allows to read audio from any format supported by ffmpeg, as well as resampling or
    converting to mono on the fly. See :method:`read` for more details.
    """
    audio_source: str | bytes | Path

    _metadata: dict[str, Any] | None = None

    _temp_file: str | None = field(default=None, init=False, repr=False)
    
    def __post_init__(self, *args, **kwargs):
        # If input is raw bytes (e.g. from URL or base64), dump to a temp file 
        # so ffprobe and ffmpeg can read it as a valid media stream.
        if isinstance(self.audio_source, bytes):
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".audio")
            tf.write(self.audio_source)
            tf.flush()
            tf.close()
            self._temp_file = tf.name
            self.audio_source = self._temp_file
        else:
            self.audio_source = str(self.audio_source)

    def __del__(self):
        # Clean up temporary file if one was created for raw bytes input
        if self._temp_file and os.path.exists(self._temp_file):
            try:
                os.unlink(self._temp_file)
            except Exception:
                pass
    
    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = _read_metadata(self.audio_source)

        return self._metadata

    @property
    def _audio_streams(self):
        return [
            index for index, stream in enumerate(self.metadata["streams"])
            if stream["codec_type"] == "audio"
        ]

    def __len__(self):
        return len(self._audio_streams)

    def channels(self, stream=0):
        return int(self.metadata['streams'][self._audio_streams[stream]]['channels'])

    def samplerate(self, stream=0):
        return int(self.metadata['streams'][self._audio_streams[stream]]['sample_rate'])

    def read(
        self,
        start_seconds: float | None = None,  # seek time
        stop_seconds: float | None = None,  # Duration
        streams=slice(None),
        sampling_rate: int | None = None,
        num_channels: int | None = None,
    ) -> torch.Tensor:
        """
        Slightly more efficient implementation than stempeg,
        in particular, this will extract all stems at once
        rather than having to loop over one file multiple times
        for each stream.

        Args:
            seek_time (float):  seek time in seconds or None if no seeking is needed.
            duration (float): duration in seconds to extract or None to extract until the end.
            streams (slice, int or list): streams to extract, can be a single int, a list or
                a slice. If it is a slice or list, the output will be of size [S, C, T]
                with S the number of streams, C the number of channels and T the number of samples.
                If it is an int, the output will be [C, T].
            samplerate (int): if provided, will resample on the fly. If None, no resampling will
                be done. Original sampling rate can be obtained with :method:`samplerate`.
            channels (int): if 1, will convert to mono. We do not rely on ffmpeg for that
                as ffmpeg automatically scale by +3dB to conserve volume when playing on speakers.
                See https://sound.stackexchange.com/a/42710.
                Our definition of mono is simply the average of the two channels. Any other
                value will be ignored.
        """
        streams = np.array(range(len(self)))[streams]
        single = not isinstance(streams, np.ndarray)
        if single:
            streams = [streams]

        if stop_seconds is None:
            target_size = None
            query_duration = None
        else:
            target_size = int((sampling_rate or self.samplerate()) * stop_seconds)
            query_duration = float((target_size + 1) / (sampling_rate or self.samplerate()))

        with temp_filenames(len(streams)) as filenames:
            command = ['ffmpeg', '-y']
            command += ['-loglevel', 'panic']
            if start_seconds:
                command += ['-ss', str(start_seconds)]
            command += ['-i', str(self.audio_source)]
            for stream, filename in zip(streams, filenames):
                command += ['-map', f'0:{self._audio_streams[stream]}']
                if query_duration is not None:
                    command += ['-t', str(query_duration)]
                command += ['-threads', '1']
                command += ['-f', 'f32le']
                if sampling_rate is not None:
                    command += ['-ar', str(sampling_rate)]
                command += [filename]

            subprocess.run(command, check=True)

            wavs = []
            for stream, filename in zip(streams, filenames):
                wav = np.fromfile(filename, dtype=np.float32)
                wav = torch.from_numpy(wav)
                wav = wav.view(-1, self.channels(stream)).t()
                if num_channels is not None:
                    wav = convert_audio_channels(wav, num_channels)
                if target_size is not None:
                    wav = wav[..., :target_size]
                wavs.append(wav)

        wav = torch.stack(wavs, dim=0)
        if single:
            wav = wav[0]

        return wav


def convert_audio_channels(wav: torch.Tensor, channels=2):
    """Convert audio to the given number of channels."""
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, but the stream have multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file have
        # one single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file have
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav


if __name__ == "__main__":
    audio_path = Path("../dataset/mahalini.mp4")
    audio_bytes = audio_path.read_bytes()
    audio_1 = AudioLoader(
        audio_path
    ).read(streams=0)

    audio_2 = AudioLoader(
        audio_bytes
    ).read(streams=0)

    assert audio_1.shape == audio_2.shape