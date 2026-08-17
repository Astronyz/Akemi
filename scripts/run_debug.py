import time
import structlog
from akemi.akemi.audio.capture import AudioCapture
from akemi.akemi.audio.vad import VAD

logger = structlog.get_logger()

def main():
    # Initialize VAD
    vad = VAD(aggressiveness=2, sample_rate=16000)
    # Initialize audio capture
    audio_capture = AudioCapture(
        chunk_size=1024,  # 20ms at 16kHz (16000 * 0.02 = 320 samples, but we use 1024 for better performance? Actually, we need to match the VAD frame size.
        sample_rate=16000,
        channels=1,
        buffer_size_chunks=100,
    )

    # We need to adjust the chunk size to be a multiple of the VAD frame size.
    # For 16kHz, VAD expects 10, 20, or 30 ms frames.
    # Let's set the chunk size to 320 samples (20ms) for 16kHz mono 16-bit -> 640 bytes.
    # But note: the AudioCapture class uses the chunk_size for the stream and the ring buffer.
    # We'll change the AudioCapture initialization to use a chunk size that is suitable for VAD.
    # However, to avoid changing the AudioCapture class, we can process the chunks in the callback
    # by splitting them into VAD-sized frames if necessary.
    # For simplicity, let's set the chunk size in AudioCapture to 320 samples (20ms) -> 640 bytes.
    # We'll recreate the audio_capture with the appropriate chunk size.

    # Actually, let's adjust: we'll keep the AudioCapture as is and in the callback, we'll split the chunk into VAD frames.
    # But note: the VAD expects a specific frame size. We'll do that in the callback.

    def audio_callback(chunk: bytes):
        # For 16kHz, 16-bit mono, 20ms frame is 320 samples = 640 bytes.
        frame_size = 640  # bytes for 20ms at 16kHz mono 16-bit
        if len(chunk) < frame_size:
            # Not enough for a single frame, we can ignore or pad? We'll just return.
            return
        # Process the chunk in frame_size increments
        for i in range(0, len(chunk), frame_size):
            frame = chunk[i:i+frame_size]
            if len(frame) == frame_size:
                if vad.is_speech(frame):
                    logger.info("Speech detected")

    # Start capture with the callback
    audio_capture.start(callback=audio_callback)

    try:
        logger.info("Listening for speech... Press Ctrl+C to stop")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        audio_capture.stop()

if __name__ == "__main__":
    main()