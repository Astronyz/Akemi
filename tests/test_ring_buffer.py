from akemi.audio.ring_buffer import RingBuffer
import pytest


def test_ring_buffer_drops_oldest():
    buf = RingBuffer(2)
    buf.append(b"a")
    buf.append(b"b")
    buf.append(b"c")
    assert buf.get() == [b"b", b"c"]
    assert len(buf) == 2


def test_ring_buffer_clear():
    buf = RingBuffer(3)
    buf.append(b"x")
    buf.clear()
    assert buf.is_empty()


def test_ring_buffer_rejects_invalid_size():
    with pytest.raises(ValueError):
        RingBuffer(0)
