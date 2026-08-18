import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from collections import deque
import structlog

logger = structlog.get_logger()


@dataclass
class FrameDiffResult:
    """Result of frame difference analysis."""

    has_change: bool
    change_score: float  # 0.0 to 1.0
    changed_regions: list  # List of (x, y, w, h) bounding boxes
    diff_image: Optional[np.ndarray] = None  # Visualization of differences


class FrameDifferencer:
    """Detect visual changes between frames using multiple methods."""

    def __init__(
        self,
        threshold: float = 0.05,
        min_region_size: int = 100,
        method: str = "mse",  # "mse", "ssim", "pixel_diff"
        history_size: int = 5,
    ):
        """
        Initialize frame differencer.

        Args:
            threshold: Change threshold (0.0 to 1.0)
            min_region_size: Minimum pixel count for a changed region
            method: Comparison method
            history_size: Number of previous frames to keep for adaptive thresholding
        """
        self.threshold = threshold
        self.min_region_size = min_region_size
        self.method = method
        self.history_size = history_size
        self._frame_history = deque(maxlen=history_size)
        self._initialized = False

    def initialize(self) -> None:
        self._initialized = True
        logger.info("Frame differencer initialized", method=self.method, threshold=self.threshold)

    def compare(
        self,
        current: np.ndarray,
        previous: Optional[np.ndarray] = None,
        return_diff: bool = False,
    ) -> FrameDiffResult:
        """
        Compare current frame with previous frame.

        Args:
            current: Current frame (H, W, 3) RGB uint8
            previous: Previous frame (if None, uses last from history)
            return_diff: Whether to return diff visualization

        Returns:
            FrameDiffResult with change detection info
        """
        if not self._initialized:
            self.initialize()

        if previous is None:
            if not self._frame_history:
                # First frame - no comparison possible
                self._frame_history.append(current)
                return FrameDiffResult(
                    has_change=False,
                    change_score=0.0,
                    changed_regions=[],
                    diff_image=None,
                )
            previous = self._frame_history[-1]

        # Ensure same shape
        if current.shape != previous.shape:
            logger.warning("Frame shape mismatch", current=current.shape, previous=previous.shape)
            previous = self._resize_to_match(previous, current.shape)

        # Compute difference based on method
        if self.method == "mse":
            change_score, diff_map = self._mse_diff(current, previous)
        elif self.method == "pixel_diff":
            change_score, diff_map = self._pixel_diff(current, previous)
        else:
            change_score, diff_map = self._mse_diff(current, previous)

        # Find changed regions
        changed_regions = self._find_changed_regions(diff_map)

        # Filter by size
        changed_regions = [
            r for r in changed_regions
            if r[2] * r[3] >= self.min_region_size
        ]

        has_change = change_score > self.threshold and len(changed_regions) > 0

        # Add to history
        self._frame_history.append(current)

        diff_image = None
        if return_diff:
            diff_image = self._create_diff_visualization(current, previous, diff_map, changed_regions)

        return FrameDiffResult(
            has_change=has_change,
            change_score=change_score,
            changed_regions=changed_regions,
            diff_image=diff_image,
        )

    def _mse_diff(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[float, np.ndarray]:
        """Mean Squared Error difference."""
        # Convert to float for computation
        img1_f = img1.astype(np.float32) / 255.0
        img2_f = img2.astype(np.float32) / 255.0

        # MSE per pixel (average across channels)
        mse = np.mean((img1_f - img2_f) ** 2, axis=2)  # (H, W)

        # Overall score (normalized to 0-1)
        change_score = float(np.mean(mse))

        return change_score, mse

    def _pixel_diff(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[float, np.ndarray]:
        """Simple pixel difference (L1 norm)."""
        img1_f = img1.astype(np.float32) / 255.0
        img2_f = img2.astype(np.float32) / 255.0

        diff = np.mean(np.abs(img1_f - img2_f), axis=2)
        change_score = float(np.mean(diff))

        return change_score, diff

    def _find_changed_regions(self, diff_map: np.ndarray) -> list:
        """Find connected components of changed pixels."""
        # Threshold the diff map
        binary = (diff_map > self.threshold).astype(np.uint8)

        if np.sum(binary) == 0:
            return []

        # Find connected components using scipy if available, else simple approach
        try:
            from scipy import ndimage
            labeled, num_features = ndimage.label(binary)
            regions = []
            for i in range(1, num_features + 1):
                coords = np.where(labeled == i)
                if len(coords[0]) >= self.min_region_size:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    regions.append((int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)))
            return regions
        except ImportError:
            # Fallback: simple bounding box of all changes
            coords = np.where(binary)
            if len(coords[0]) >= self.min_region_size:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                return [(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))]
            return []

    def _resize_to_match(self, img: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Resize image to match target shape."""
        from PIL import Image
        pil_img = Image.fromarray(img, "RGB")
        pil_img = pil_img.resize((target_shape[1], target_shape[0]), Image.Resampling.LANCZOS)
        return np.array(pil_img)

    def _create_diff_visualization(
        self,
        current: np.ndarray,
        previous: np.ndarray,
        diff_map: np.ndarray,
        regions: list,
    ) -> np.ndarray:
        """Create visualization of differences."""
        # Normalize diff map to 0-255
        diff_vis = (diff_map / (diff_map.max() + 1e-6) * 255).astype(np.uint8)

        # Create RGB heatmap
        heatmap = np.zeros_like(current)
        heatmap[:, :, 0] = diff_vis  # Red channel

        # Blend with current
        alpha = 0.5
        blended = (current * (1 - alpha) + heatmap * alpha).astype(np.uint8)

        # Draw region boxes
        for x, y, w, h in regions:
            # Draw rectangle
            blended[y:y+h, x:x+2] = [0, 255, 0]  # Left
            blended[y:y+h, x+w-2:x+w] = [0, 255, 0]  # Right
            blended[y:y+2, x:x+w] = [0, 255, 0]  # Top
            blended[y+h-2:y+h, x:x+w] = [0, 255, 0]  # Bottom

        return blended

    def reset_history(self) -> None:
        """Clear frame history."""
        self._frame_history.clear()

    def close(self) -> None:
        self._frame_history.clear()
        self._initialized = False