import time
from typing import List, AsyncIterator
import openai
import structlog

from akemi.akemi.brain.providers.base import BrainProvider, BrainResponse, BrainMessage, BrainProviderFactory

logger = structlog.get_logger()


class OpenAIProvider(BrainProvider):
    """OpenAI API provider (GPT-4, GPT-3.5, etc.)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self._client: openai.AsyncOpenAI | None = None

    @property
    def provider_name(self) -> str:
        return "openai"

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )
        await self.health_check()
        self._initialized = True
        logger.info("OpenAI provider initialized", model=self.model)

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

        chat_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        latency_ms = int((time.perf_counter() - start) * 1000)
        choice = response.choices[0]

        return BrainResponse(
            text=choice.message.content or "",
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            latency_ms=latency_ms,
            model=self.model,
            provider=self.provider_name,
            metadata={"finish_reason": choice.finish_reason},
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

        chat_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def health_check(self) -> bool:
        try:
            if not self._client:
                return False
            await self._client.chat.completions.create(
                model=self.model,
                max_tokens=5,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception as e:
            logger.error("OpenAI health check failed", error=str(e))
            return False

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
        self._initialized = False


# Register the provider
BrainProviderFactory.register("openai", OpenAIProvider)