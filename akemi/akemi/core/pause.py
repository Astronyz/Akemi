import threading
from typing import Optional
import structlog

logger = structlog.get_logger()


class PauseController:
    """Thread-safe pause/resume controller for the agent."""

    def __init__(self):
        self._paused = False
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._pause_reason: Optional[str] = None

    def pause(self, reason: str = "Manual pause") -> None:
        """Pause the agent."""
        with self._lock:
            if not self._paused:
                self._paused = True
                self._pause_reason = reason
                logger.info("Agent paused", reason=reason)

    def resume(self) -> None:
        """Resume the agent."""
        with self._lock:
            if self._paused:
                self._paused = False
                self._pause_reason = None
                self._condition.notify_all()
                logger.info("Agent resumed")

    def is_paused(self) -> bool:
        """Check if agent is paused."""
        with self._lock:
            return self._paused

    def get_pause_reason(self) -> Optional[str]:
        """Get the reason for pause, if paused."""
        with self._lock:
            return self._pause_reason if self._paused else None

    def wait_if_paused(self, timeout: Optional[float] = None) -> bool:
        """
        Block until resumed or timeout.

        Args:
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            True if resumed, False if timed out while still paused
        """
        with self._condition:
            if not self._paused:
                return True
            return self._condition.wait(timeout=timeout)

    def toggle(self, reason: str = "Toggled") -> bool:
        """Toggle pause state. Returns new state (True = paused)."""
        with self._lock:
            if self._paused:
                self.resume()
                return False
            else:
                self.pause(reason)
                return True


# Global instance
pause_controller = PauseController()