from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional, List, Dict, Any
import structlog

logger = structlog.get_logger()


@dataclass
class BrainResponse:
    """Response from a brain provider."""

    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    model: str = ""
    provider: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BrainMessage:
    """Message for brain conversation."""

    role: str  # "system", "user", "assistant"
    content: str


class BrainProvider(ABC):
    """Abstract base class for brain providers."""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.config = kwargs
        self._initialized = False

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'anthropic', 'ollama')."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (load model, verify connection, etc.)."""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[BrainMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> BrainResponse:
        """Generate a response from the model."""
        pass

    @abstractmethod
    async def stream_generate(
        self,
        messages: List[BrainMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a response from the model."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy/available."""
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


class BrainProviderFactory:
    """Factory for creating brain providers."""

    _providers: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, provider_class: type) -> None:
        """Register a provider class."""
        cls._providers[name] = provider_class
        logger.info("Registered brain provider", name=name)

    @classmethod
    def create(cls, provider_name: str, model: str, **kwargs) -> BrainProvider:
        """Create a provider instance."""
        if provider_name not in cls._providers:
            raise ValueError(f"Unknown brain provider: {provider_name}. "
                           f"Available: {list(cls._providers.keys())}")
        provider_class = cls._providers[provider_name]
        return provider_class(model=model, **kwargs)

    @classmethod
    def get_available(cls) -> List[str]:
        """Get list of available provider names."""
        return list(cls._providers.keys())