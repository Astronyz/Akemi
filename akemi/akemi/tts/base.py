from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, AsyncIterator
import structlog

logger = structlog.get_logger()


@dataclass
class TTSResult:
    """Result of TTS synthesis."""

    audio_data: bytes  # Raw audio bytes (WAV format)
    sample_rate: int
    duration: float  # seconds
    text: str
    engine: str


class TTSEngine(ABC):
    """Abstract base class for TTS engines."""

    def __init__(self, **kwargs):
        self.config = kwargs
        self._initialized = False

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Return engine name (e.g., 'piper', 'cloud')."""
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Return output sample rate."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the engine."""
        pass

    @abstractmethod
    async def synthesize(self, text: str, **kwargs) -> TTSResult:
        """Synthesize text to speech."""
        pass

    @abstractmethod
    async def stream_synthesize(self, text: str, **kwargs) -> AsyncIterator[bytes]:
        """Stream audio chunks as they're generated."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class TTSEngineFactory:
    """Factory for creating TTS engines."""

    _engines: dict = {}

    @classmethod
    def register(cls, name: str, engine_class: type) -> None:
        cls._engines[name] = engine_class
        logger.info("Registered TTS engine", name=name)

    @classmethod
    def create(cls, engine_name: str, **kwargs) -> TTSEngine:
        if engine_name not in cls._engines:
            raise ValueError(f"Unknown TTS engine: {engine_name}. Available: {list(cls._engines.keys())}")
        return cls._engines[engine_name](**kwargs)

    @classmethod
    def get_available(cls) -> list:
        return list(cls._engines.keys())