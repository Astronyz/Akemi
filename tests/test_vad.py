import pytest

from akemi.audio.vad import VAD, vad_frame_bytes


def test_vad_frame_bytes_20ms_16k():
    assert vad_frame_bytes(16000, 20) == 640


def test_vad_rejects_bad_aggressiveness():
    with pytest.raises(ValueError):
        VAD(aggressiveness=9)


def test_vad_silence_is_not_speech():
    vad = VAD(aggressiveness=2, sample_rate=16000)
    silence = b"\x00" * vad_frame_bytes(16000, 20)
    assert vad.is_speech(silence) is False


def test_vad_invalid_frame_size_returns_false():
    vad = VAD(sample_rate=16000)
    assert vad.is_speech(b"\x00\x01") is False
