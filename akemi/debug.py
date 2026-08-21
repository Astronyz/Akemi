"""Local debug entrypoint: loopback → 16 kHz frames → VAD."""

from __future__ import annotations

import sys
import time

import structlog

from akemi.audio.capture import AudioCapture
from akemi.audio.vad import VAD

logger = structlog.get_logger()


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]
    )


def main() -> None:
    _configure_logging()
    if sys.platform != "win32":
        logger.error("Akemi audio debug only runs on Windows")
        raise SystemExit(1)

    vad = VAD(aggressiveness=2, sample_rate=16000)
    capture = AudioCapture()

    def on_frame(frame: bytes) -> None:
        if vad.is_speech(frame):
            logger.info("Speech detected")

    capture.start(callback=on_frame)
    try:
        logger.info("Listening for speech... Press Ctrl+C to stop")
        while capture.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        capture.stop()


if __name__ == "__main__":
    main()
