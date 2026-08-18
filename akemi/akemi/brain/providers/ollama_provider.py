import time
from typing import List, AsyncIterator
import httpx
import structlog

from akemi.akemi.brain.providers.base import BrainProvider, BrainResponse, BrainMessage, BrainProviderFactory

logger = structlog.get_logger()


class OllamaProvider(BrainProvider):
    """Ollama local LLM provider."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._client = httpx.AsyncClient(timeout=self.timeout)
        await self.health_check()
        self._initialized = True
        logger.info("Ollama provider initialized", model=self.model, host=self.host)

    def _format_messages(self, messages: List[BrainMessage]) -> List[Dict]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

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

        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        response = await self._client.post(
            f"{self.host}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        latency_ms = int((time.perf_counter() - start) * 1000)

        return BrainResponse(
            text=data.get("message", {}).get("content", ""),
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
            latency_ms=latency_ms,
            model=self.model,
            provider=self.provider_name,
            metadata={"done_reason": data.get("done_reason")},
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

        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        async with self._client.stream(
            "POST",
            f"{self.host}/api/chat",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    import json
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                    except json.JSONDecodeError:
                        continue

    async def health_check(self) -> bool:
        try:
            if not self._client:
                return False
            # Check if model is available
            response = await self._client.get(f"{self.host}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            # Check if our model (or base name) is available
            model_base = self.model.split(":")[0]
            available = any(model_base in name for name in model_names)
            if not available:
                logger.warning("Ollama model not found locally", model=self.model, available=model_names)
            return True
        except Exception as e:
            logger.error("Ollama health check failed", error=str(e))
            return False

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        self._initialized = False


# Register the provider
BrainProviderFactory.register("ollama", OllamaProvider)