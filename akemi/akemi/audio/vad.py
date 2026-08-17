import webrtcvad
import structlog
from typing import Optional

logger = structlog.get_logger()


class VAD:
    """Voice Activity Detection using webrtcvad."""

    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000):
        """Initialize the VAD.

        Args:
            aggressiveness: An integer between 0 and 3. Higher means more aggressive in filtering non-speech.
            sample_rate: Audio sample rate in Hz. Must be 8000, 16000, 32000, or 48000.
        """
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("Sample rate must be 8000, 16000, 32000, or 48000 Hz")
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        logger.info("VAD initialized", aggressiveness=aggressiveness, sample_rate=sample_rate)

    def is_speech(self, frame: bytes, sample_rate: Optional[int] = None) -> bool:
        """Check if the given frame contains speech.

        Args:
            frame: Audio frame as bytes (16-bit PCM, mono).
            sample_rate: Sample rate of the frame. If None, uses the instance's sample_rate.

        Returns:
            True if speech is detected, False otherwise.
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        # webrtcvad expects a specific frame length: 10, 20, or 30 ms.
        # We'll assume the frame is 20 ms (as we set in the capture chunk size for 16kHz: 320 samples -> 20ms)
        # But we can also compute the required length in bytes: 2 bytes per sample * sample_rate * (frame_duration / 1000)
        # We'll let the caller ensure the frame is the correct length for the sample rate.
        try:
            return self.vad.is_speech(frame, sample_rate)
        except Exception as e:
            logger.error("Error in VAD processing", error=str(e))
            return False