import time
from typing import Optional, Dict, Any, List
import httpx
import structlog

from akemi.akemi.core.config import get_settings

logger = structlog.get_logger()


class HermesClient:
    """Client for communicating with Hermes agent."""

    def __init__(
        self,
        base_url: str = "http://localhost:5000",
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Hermes client.

        Args:
            base_url: Base URL of Hermes API
            timeout: Request timeout in seconds
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        """Initialize HTTP client."""
        if self._client is not None:
            return

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=headers,
        )

        # Test connection
        await self.health_check()
        logger.info("Hermes client initialized", base_url=self.base_url)

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if Hermes is reachable."""
        try:
            if not self._client:
                await self.initialize()
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception as e:
            logger.error("Hermes health check failed", error=str(e))
            return False

    async def send_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send a message to Hermes.

        Args:
            message: Message text
            session_id: Optional session ID
            context: Optional context data

        Returns:
            Response from Hermes
        """
        if not self._client:
            await self.initialize()

        payload = {
            "message": message,
            "session_id": session_id,
            "context": context or {},
        }

        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        return response.json()

    async def stream_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Stream a message to Hermes (SSE)."""
        if not self._client:
            await self.initialize()

        payload = {
            "message": message,
            "session_id": session_id,
            "context": context or {},
            "stream": True,
        }

        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    import json
                    yield json.loads(line[6:])

    async def get_status(self) -> Dict[str, Any]:
        """Get Hermes agent status."""
        if not self._client:
            await self.initialize()

        response = await self._client.get("/api/status")
        response.raise_for_status()
        return response.json()

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List active sessions."""
        if not self._client:
            await self.initialize()

        response = await self._client.get("/api/sessions")
        response.raise_for_status()
        return response.json()

    async def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new session."""
        if not self._client:
            await self.initialize()

        response = await self._client.post("/api/sessions", json=metadata or {})
        response.raise_for_status()
        return response.json()

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session details."""
        if not self._client:
            await self.initialize()

        response = await self._client.get(f"/api/sessions/{session_id}")
        response.raise_for_status()
        return response.json()

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if not self._client:
            await self.initialize()

        response = await self._client.delete(f"/api/sessions/{session_id}")
        return response.status_code == 204

    async def notify_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Notify Hermes of an event (fire and forget).

        Args:
            event_type: Type of event (e.g., "speech_detected", "screen_change")
            data: Event data
            session_id: Optional session ID

        Returns:
            True if notification was sent successfully
        """
        if not self._client:
            await self.initialize()

        try:
            payload = {
                "event_type": event_type,
                "data": data,
                "session_id": session_id,
                "timestamp": time.time(),
            }
            response = await self._client.post("/api/events", json=payload)
            return response.status_code == 200
        except Exception as e:
            logger.error("Failed to notify Hermes", event_type=event_type, error=str(e))
            return False


# Global client instance
_hermes_client: Optional[HermesClient] = None


async def get_hermes_client() -> HermesClient:
    """Get the global Hermes client instance."""
    global _hermes_client
    if _hermes_client is None:
        settings = get_settings()
        _hermes_client = HermesClient(
            base_url=settings.network.hermes_api_url,
            timeout=settings.network.hermes_timeout,
        )
        await _hermes_client.initialize()
    return _hermes_client


async def close_hermes_client() -> None:
    """Close the global Hermes client."""
    global _hermes_client
    if _hermes_client:
        await _hermes_client.close()
        _hermes_client = None