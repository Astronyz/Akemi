from akemi.audio.capture import AudioCapture


def test_get_recent_audio_uses_output_frame_size():
    capture = AudioCapture(buffer_size_chunks=50)
    frame = b"\x00" * 640
    for i in range(10):
        capture._ring_buffer.append(frame + bytes([i % 256]))
    recent = capture.get_recent_audio(seconds=0.06)
    # 60 ms → 3 frames of 20 ms
    assert len(recent) == 3


def test_stop_is_safe_before_start():
    capture = AudioCapture()
    capture.stop()
    assert capture.is_running() is False
