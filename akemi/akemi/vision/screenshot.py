import time
import mss
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import structlog

logger = structlog.get_logger()


@dataclass
class ScreenshotResult:
    """Result of a screenshot capture."""

    image: np.ndarray  # RGB array (H, W, 3)
    timestamp: float
    monitor_index: int
    monitor_info: dict
    file_path: Optional[str] = None


class ScreenshotCapture:
    """Screen capture using mss (cross-platform, fast)."""

    def __init__(
        self,
        monitor_index: int = 0,
        output_dir: Optional[str] = None,
        image_format: str = "PNG",
    ):
        """
        Initialize screenshot capture.

        Args:
            monitor_index: Monitor to capture (0 = primary, 1+ = additional)
            output_dir: Directory to save screenshots (None = don't save)
            image_format: Image format for saving (PNG, JPEG)
        """
        self.monitor_index = monitor_index
        self.output_dir = Path(output_dir) if output_dir else None
        self.image_format = image_format
        self._sct = None
        self._monitors = []
        self._initialized = False

    def initialize(self) -> None:
        """Initialize mss and get monitor info."""
        if self._initialized:
            return

        self._sct = mss.mss()
        self._monitors = self._sct.monitors

        if self.monitor_index >= len(self._monitors):
            raise ValueError(
                f"Monitor index {self.monitor_index} not available. "
                f"Available: {len(self._monitors)} monitors (0 = all)"
            )

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self._initialized = True
        logger.info(
            "Screenshot capture initialized",
            monitor_index=self.monitor_index,
            monitor_info=self.get_monitor_info(),
        )

    def get_monitor_info(self) -> dict:
        """Get information about the selected monitor."""
        if not self._initialized:
            self.initialize()
        monitor = self._monitors[self.monitor_index]
        return {
            "index": self.monitor_index,
            "left": monitor["left"],
            "top": monitor["top"],
            "width": monitor["width"],
            "height": monitor["height"],
        }

    def capture(self, save: bool = False) -> ScreenshotResult:
        """
        Capture a screenshot.

        Args:
            save: Whether to save to disk (if output_dir is set)

        Returns:
            ScreenshotResult with image array and metadata
        """
        if not self._initialized:
            self.initialize()

        monitor = self._monitors[self.monitor_index]
        timestamp = time.time()

        # Capture
        screenshot = self._sct.grab(monitor)

        # Convert to numpy array (BGRA -> RGB)
        img = np.array(screenshot)
        img_rgb = img[:, :, :3][:, :, ::-1]  # BGRA -> RGB

        file_path = None
        if save and self.output_dir:
            file_path = self._save_image(img_rgb, timestamp)

        return ScreenshotResult(
            image=img_rgb,
            timestamp=timestamp,
            monitor_index=self.monitor_index,
            monitor_info=self.get_monitor_info(),
            file_path=file_path,
        )

    def _save_image(self, img_rgb: np.ndarray, timestamp: float) -> str:
        """Save image to disk."""
        from datetime import datetime
        dt = datetime.fromtimestamp(timestamp)
        filename = f"screenshot_{dt.strftime('%Y%m%d_%H%M%S_%f')}.{self.image_format.lower()}"
        file_path = self.output_dir / filename

        pil_img = Image.fromarray(img_rgb, "RGB")
        pil_img.save(file_path, format=self.image_format)

        return str(file_path)

    def capture_to_bytes(self, format: str = "PNG") -> bytes:
        """Capture and return as bytes."""
        result = self.capture()
        pil_img = Image.fromarray(result.image, "RGB")
        buffer = io.BytesIO()
        pil_img.save(buffer, format=format)
        return buffer.getvalue()

    def close(self) -> None:
        """Clean up resources."""
        if self._sct:
            self._sct.close()
            self._sct = None
        self._initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


import io  # For capture_to_bytes