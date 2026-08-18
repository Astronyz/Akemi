import time
import wave
import io
from typing import AsyncIterator
import structlog

from akemi.akemi.tts.base import TTSEngine, TTSResult, TTSEngineFactory

logger = structlog.get_logger()


class PiperEngine(TTSEngine):
    """Piper TTS engine (local, offline)."""

    def __init__(
        self,
        model_path: str,
        speaker_id: int = 0,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        sentence_silence: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.speaker_id = speaker_id
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.sentence_silence = sentence_silence
        self._voice = None
        self._sample_rate = 0

    @property
    def engine_name(self) -> str:
        return "piper"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def initialize(self) -> None:
        if self._initialized:
            return

        try:
            from piper import PiperVoice
        except ImportError:
            raise RuntimeError(
                "piper-tts not installed. Install with: pip install piper-tts"
            )

        import os
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Piper model not found: {self.model_path}")

        import asyncio
        loop = asyncio.get_event_loop()

        def _load_voice():
            return PiperVoice.load(self.model_path)

        self._voice = await loop.run_in_executor(None, _load_voice)
        self._sample_rate = self._voice.config.sample_rate
        self._initialized = True
        logger.info(
            "Piper engine initialized",
            model=self.model_path,
            sample_rate=self._sample_rate,
            speakers=self._voice.config.num_speakers,
        )

    def _synthesize_internal(self, text: str) -> bytes:
        """Synthesize text to WAV bytes (blocking)."""
        if not self._initialized:
            raise RuntimeError("Piper engine not initialized")

        # Create in-memory WAV file
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self._sample_rate)

            # Synthesize
            self._voice.synthesize(
                text,
                wav_file,
                speaker_id=self.speaker_id,
                length_scale=self.length_scale,
                noise_scale=self.noise_scale,
                noise_w=self.noise_w,
                sentence_silence=self.sentence_silence,
            )

        return buffer.getvalue()

    async def synthesize(self, text: str, **kwargs) -> TTSResult:
        if not self._initialized:
            await self.initialize()

        import asyncio
        loop = asyncio.get_event_loop()

        start = time.perf_counter()
        audio_data = await loop.run_in_executor(None, self._synthesize_internal, text)
        duration = time.perf_counter() - start

        # Calculate audio duration from WAV header
        import wave
        import io
        with wave.open(io.BytesIO(audio_data), "rb") as wav:
            frames = wav.getnframes()
            audio_duration = frames / wav.getframerate()

        return TTSResult(
            audio_data=audio_data,
            sample_rate=self._sample_rate,
            duration=audio_duration,
            text=text,
            engine=self.engine_name,
        )

    async def stream_synthesize(self, text: str, **kwargs) -> AsyncIterator[bytes]:
        """Stream synthesis in chunks (sentence by sentence)."""
        if not self._initialized:
            await self.initialize()

        import asyncio
        loop = asyncio.get_event_loop()

        # Simple sentence splitting
        sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
        if not sentences:
            sentences = [text]

        for sentence in sentences:
            audio_data = await loop.run_in_executor(None, self._synthesize_internal, sentence)
            # Yield chunks (skip WAV header on subsequent chunks)
            yield audio_data

    async def close(self) -> None:
        self._voice = None
        self._initialized = False
        self._sample_rate = 0


# Register the engine
TTSEngineFactory.register("piper", PiperEngine)