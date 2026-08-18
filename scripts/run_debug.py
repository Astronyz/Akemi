import time
import structlog
from akemi.akemi.audio.capture import AudioCapture
from akemi.akemi.audio.vad import VAD

logger = structlog.get_logger()


def main():
    # Initialize VAD (aggressiveness 0-3, higher = more aggressive filtering)
    vad = VAD(aggressiveness=2, sample_rate=16000)

    # Initialize audio capture - agora já entrega frames de 20ms @ 16kHz mono
    audio_capture = AudioCapture(
        chunk_size=320,      # 20ms @ 16kHz (alignado com VAD)
        sample_rate=16000,   # Forçado internamente
        channels=1,          # Forçado internamente
        buffer_size_chunks=100,
    )

    speech_frames = 0
    total_frames = 0

    def audio_callback(chunk: bytes):
        nonlocal speech_frames, total_frames
        total_frames += 1

        # chunk já vem no formato correto: 320 samples (20ms) @ 16kHz mono, 16-bit PCM
        if vad.is_speech(chunk):
            speech_frames += 1
            logger.info("Speech detected", frame=total_frames, speech_frames=speech_frames)
        elif total_frames % 50 == 0:  # Log periódico de silêncio
            logger.debug("Listening...", frame=total_frames, speech_ratio=f"{speech_frames/total_frames:.2%}")

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
        logger.info("Session stats", total_frames=total_frames, speech_frames=speech_frames)


if __name__ == "__main__":
    main()