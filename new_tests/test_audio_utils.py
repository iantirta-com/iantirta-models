import base64
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

# Import your audio utilities (adjust package path as needed)
from iantirta.models.audio_separation.audio_utils import (
    get_file_type_from_source,
    get_file_type_from_bytes,
    resolve_audio_from_source,
    load_audio,
    AudioLoader,
)


def test_get_file_type_from_source_extensions():
    assert get_file_type_from_source("song.mp3") == "mp3"
    assert get_file_type_from_source("/path/to/audio.WAV") == "wav"
    assert get_file_type_from_source("https://example.com/stream.m4a?query=1") == "m4a"
    assert get_file_type_from_source("no_extension") is None


def test_get_file_type_from_source_data_uri():
    data_uri = "data:audio/x-wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."
    assert get_file_type_from_source(data_uri) == "wav"

    mp3_data_uri = "data:audio/mp3;base64,SUQzBAAAAAAAI..."
    assert get_file_type_from_source(mp3_data_uri) == "mp3"


def test_get_file_type_from_bytes_signatures():
    # Test RIFF WAV header
    wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 40
    assert get_file_type_from_bytes(wav_bytes) == "wav"

    # Test FLAC header
    flac_bytes = b"fLaC\x00\x00\x00\x22" + b"\x00" * 40
    assert get_file_type_from_bytes(flac_bytes) == "flac"

    # Test OGG header
    ogg_bytes = b"OggS\x00\x02\x00\x00" + b"\x00" * 40
    assert get_file_type_from_bytes(ogg_bytes) == "ogg"

    # Test invalid unsupported bytes
    with pytest.raises(ValueError, match="not supported filetype"):
        get_file_type_from_bytes(b"INVALID_HEADER_BYTES_1234567890")


def test_resolve_audio_from_source_local_file(tmp_path: Path):
    dummy_file = tmp_path / "test.wav"
    dummy_file.write_text("dummy content")

    resolved = resolve_audio_from_source(str(dummy_file))
    assert resolved == str(dummy_file)


def test_resolve_audio_from_source_base64():
    raw_str = "Hello World Audio"
    b64_encoded = base64.b64encode(raw_str.encode()).decode()
    
    # Test data URI format
    data_uri = f"data:audio/wav;base64,{b64_encoded}"
    decoded = resolve_audio_from_source(data_uri)
    assert isinstance(decoded, bytes)
    assert decoded.decode() == raw_str

    # Test raw base64 format
    decoded_raw = resolve_audio_from_source(b64_encoded)
    assert isinstance(decoded_raw, bytes)
    assert decoded_raw.decode() == raw_str


@patch("requests.get")
def test_resolve_audio_from_source_url(mock_get):
    mock_response = MagicMock()
    mock_response.content = b"fake_audio_bytes_from_url"
    mock_get.return_value = mock_response

    url = "https://example.com/audio.mp3"
    result = resolve_audio_from_source(url)
    
    mock_get.assert_called_once_with(url, follow_redirects=True, timeout=10.0)
    assert result == b"fake_audio_bytes_from_url"


def test_load_audio_numpy_passthrough():
    arr = np.zeros((2, 16000), dtype=np.float32)
    result = load_audio(arr)
    np.testing.assert_array_equal(result, arr)


def test_load_audio_invalid_type_raises():
    with pytest.raises(TypeError, match="Incorrect format used for `audio`"):
        load_audio(12345)  # Invalid type (int instead of path/string/array)


def test_audio_loader_bytes_handling():
    # Verify that AudioLoader correctly creates a temp file when given raw bytes
    raw_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 40
    loader = AudioLoader(raw_bytes)
    
    assert isinstance(loader.audio_source, str)
    assert os.path.exists(loader.audio_source)
    
    temp_file_path = loader.audio_source
    del loader  # Triggers cleanup destructor
    assert not os.path.exists(temp_file_path)
