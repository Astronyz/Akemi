import numpy as np
import pytest

from akemi.audio.process import (
    FrameAssembler,
    StreamResampler,
    downmix_to_mono_int16,
    resample_int16,
)


def test_downmix_stereo_averages_channels():
    left = np.array([1000, 2000], dtype=np.int16)
    right = np.array([3000, 4000], dtype=np.int16)
    interleaved = np.column_stack((left, right)).reshape(-1).tobytes()
    mono = downmix_to_mono_int16(interleaved, channels=2)
    assert list(mono) == [2000, 3000]


def test_downmix_mono_passthrough():
    pcm = np.array([1, 2, 3], dtype=np.int16).tobytes()
    assert list(downmix_to_mono_int16(pcm, 1)) == [1, 2, 3]


def test_resample_48k_to_16k_is_exact_third():
    src = np.arange(480, dtype=np.int16)
    out, leftover = resample_int16(src, 48000, 16000)
    assert len(out) == 159  # (480-1)/3
    assert leftover.dtype == np.int16


def test_resample_same_rate_passthrough():
    src = np.array([1, 2, 3, 4], dtype=np.int16)
    out, leftover = resample_int16(src, 16000, 16000)
    assert list(out) == [1, 2, 3, 4]
    assert len(leftover) == 0


def test_stream_resampler_keeps_continuity():
    resampler = StreamResampler(48000, 16000)
    chunk = np.arange(300, dtype=np.int16)
    first = resampler.push(chunk)
    second = resampler.push(chunk)
    assert first.dtype == np.int16
    assert len(first) + len(second) > 0


def test_frame_assembler_splits_and_keeps_remainder():
    assembler = FrameAssembler(4)
    frames = assembler.push(b"abcdefg")
    assert frames == [b"abcd"]
    frames = assembler.push(b"hij")
    assert frames == [b"efgh"]


def test_frame_assembler_rejects_invalid_size():
    with pytest.raises(ValueError):
        FrameAssembler(0)
