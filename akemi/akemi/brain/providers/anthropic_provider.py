import time
from typing import List, AsyncIterator, Dict, Any
import anthropic
import structlog

from akemi.akemi.brain.providers.base import BrainProvider, BrainResponse, BrainMessage, BrainProviderFactory

logger = structlog.get_logger()


class AnthropicProvider(BrainProvider):
    """Anthropic Claude API provider."""

    def __init__(
        self,
        model: str,
        api_key: str,
        max_retries: int = 3,
        timeout: float = 60.0,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self._client: anthropic.AsyncAnthropic | None = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._client = anthropic.AsyncAnthropic(
            api_key=self.api_key,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )
        # Test connection
        await self.health_check()
        self._initialized = True
        logger.info("Anthropic provider initialized", model=self.model)

    async def generate(
        self,
        messages: List[BrainMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> BrainResponse:
        if not self._initialized:
            await self.initialize()

        start = time.perf_counter()

        # Separate system message
        system_msg = None
        chat_messages = []
        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                chat_messages.append({"role": msg.role, "content": msg.content})

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg,
            messages=chat_messages,
        )

        latency_ms = int((time.perf_counter() - start) * 1000)

        return BrainResponse(
            text=response.content[0].text if response.content else "",
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
            latency_ms=latency_ms,
            model=self.model,
            provider=self.provider_name,
            metadata={"stop_reason": response.stop_reason},
        )

    async def stream_generate(
        self,
        messages: List[BrainMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncIterator[str]:
        if not self._initialized:
            await self.initialize()

        system_msg = None
        chat_messages = []
        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                chat_messages.append({"role": msg.role, "content": msg.content})

        async with self._client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg,
            messages=chat_messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def health_check(self) -> bool:
        try:
            if not self._client:
                return False
            # Simple test request
            await self._client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception as e:
            logger.error("Anthropic health check failed", error=str(e))
            return False

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
        self._initialized = False


# Register the provider
BrainProviderFactory.register("anthropic", AnthropicProvider)