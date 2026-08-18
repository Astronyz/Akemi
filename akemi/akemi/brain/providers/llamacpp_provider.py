import time
from typing import List, AsyncIterator, Dict, Any
import structlog

from akemi.akemi.brain.providers.base import BrainProvider, BrainResponse, BrainMessage, BrainProviderFactory

logger = structlog.get_logger()


class LlamaCppProvider(BrainProvider):
    """llama.cpp Python bindings provider for local GGUF models."""

    def __init__(
        self,
        model: str,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        n_batch: int = 512,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.verbose = verbose
        self._llama = None

    @property
    def provider_name(self) -> str:
        return "llamacpp"

    async def initialize(self) -> None:
        if self._initialized:
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
            )

        import os
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Run in thread pool since llama.cpp is blocking
        import asyncio
        loop = asyncio.get_event_loop()

        self._llama = await loop.run_in_executor(
            None,
            lambda: Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                n_batch=self.n_batch,
                verbose=self.verbose,
            )
        )

        # Test inference
        await self.health_check()
        self._initialized = True
        logger.info("llama.cpp provider initialized", model=self.model, path=self.model_path)

    def _format_prompt(self, messages: List[BrainMessage]) -> str:
        """Format messages into a prompt for llama.cpp (chat template)."""
        # Simple chat template - can be extended for specific models
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"<|system|>\n{msg.content}")
            elif msg.role == "user":
                parts.append(f"<|user|>\n{msg.content}")
            elif msg.role == "assistant":
                parts.append(f"<|assistant|>\n{msg.content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    async def generate(
        self,
        messages: List[BrainMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> BrainResponse:
        if not self._initialized:
            await self.initialize()

        import asyncio
        loop = asyncio.get_event_loop()

        prompt = self._format_prompt(messages)

        start = time.perf_counter()

        def _generate():
            return self._llama(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|user|>", "<|system|>"],
                echo=False,
            )

        result = await loop.run_in_executor(None, _generate)

        latency_ms = int((time.perf_counter() - start) * 1000)

        text = result["choices"][0]["text"] if result["choices"] else ""
        usage = result.get("usage", {})

        return BrainResponse(
            text=text,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            model=self.model,
            provider=self.provider_name,
            metadata={"finish_reason": result["choices"][0].get("finish_reason") if result["choices"] else None},
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

        import asyncio
        loop = asyncio.get_event_loop()

        prompt = self._format_prompt(messages)

        def _stream():
            stream = self._llama(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|user|>", "<|system|>"],
                echo=False,
                stream=True,
            )
            for chunk in stream:
                if chunk["choices"]:
                    yield chunk["choices"][0]["text"]

        # Run generator in thread pool and yield results
        queue = asyncio.Queue()

        def _producer():
            try:
                for token in _stream():
                    asyncio.run_coroutine_threadsafe(queue.put(token), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        await loop.run_in_executor(None, _producer)

        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

    async def health_check(self) -> bool:
        try:
            if not self._llama:
                return False
            import asyncio
            loop = asyncio.get_event_loop()

            def _test():
                return self._llama("Hi", max_tokens=5, temperature=0.1)

            result = await loop.run_in_executor(None, _test)
            return bool(result["choices"])
        except Exception as e:
            logger.error("llama.cpp health check failed", error=str(e))
            return False

    async def close(self) -> None:
        self._llama = None
        self._initialized = False


# Register the provider
BrainProviderFactory.register("llamacpp", LlamaCppProvider)