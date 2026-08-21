from __future__ import annotations

from typing import Optional

import structlog
import webrtcvad

logger = structlog.get_logger()

ALLOWED_RATES = (8000, 16000, 32000, 48000)
FRAME_DURATION_MS = (10, 20, 30)
BYTES_PER_SAMPLE = 2


def vad_frame_bytes(sample_rate: int, duration_ms: int = 20) -> int:
    """Return the exact PCM16 mono frame size webrtcvad expects."""
    if sample_rate not in ALLOWED_RATES:
        raise ValueError(f"Sample rate must be one of {ALLOWED_RATES}")
    if duration_ms not in FRAME_DURATION_MS:
        raise ValueError(f"Frame duration must be one of {FRAME_DURATION_MS} ms")
    return sample_rate * duration_ms // 1000 * BYTES_PER_SAMPLE


class VAD:
    """Voice Activity Detection using webrtcvad."""

    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000):
        if aggressiveness not in (0, 1, 2, 3):
            raise ValueError("aggressiveness must be an integer between 0 and 3")
        if sample_rate not in ALLOWED_RATES:
            raise ValueError(f"Sample rate must be one of {ALLOWED_RATES}")
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        logger.info(
            "VAD initialized",
            aggressiveness=aggressiveness,
            sample_rate=sample_rate,
        )

    def is_speech(self, frame: bytes, sample_rate: Optional[int] = None) -> bool:
        """Return True if *frame* is 10/20/30 ms of speech-like PCM16 mono."""
        rate = sample_rate if sample_rate is not None else self.sample_rate
        if rate not in ALLOWED_RATES:
            logger.error("Unsupported VAD sample rate", sample_rate=rate)
            return False
        valid_sizes = {vad_frame_bytes(rate, ms) for ms in FRAME_DURATION_MS}
        if len(frame) not in valid_sizes:
            logger.error(
                "Invalid VAD frame size",
                got=len(frame),
                expected=sorted(valid_sizes),
                sample_rate=rate,
            )
            return False
        try:
            return self.vad.is_speech(frame, rate)
        except Exception as exc:
            logger.error("Error in VAD processing", error=str(exc))
            return False
