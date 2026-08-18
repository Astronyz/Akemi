import pytesseract
from PIL import Image
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class OCRResult:
    """Result of OCR processing."""

    text: str
    confidence: float  # Average confidence (0-100)
    words: List[Dict[str, Any]]  # Word-level details
    lines: List[Dict[str, Any]]  # Line-level details
    language: str
    processing_time: float


class OCREngine:
    """OCR using Tesseract."""

    def __init__(
        self,
        language: str = "por",
        config: str = "--psm 6",
        tessdata_dir: Optional[str] = None,
    ):
        """
        Initialize OCR engine.

        Args:
            language: Tesseract language code(s), e.g., "por", "eng", "por+eng"
            config: Tesseract config options (--psm, --oem, etc.)
            tessdata_dir: Path to tessdata directory (if not in default location)
        """
        self.language = language
        self.config = config
        self.tessdata_dir = tessdata_dir
        self._initialized = False

        # Verify tesseract is available
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.warning("Tesseract not found in PATH", error=str(e))

    def initialize(self) -> None:
        """Initialize OCR engine."""
        if self._initialized:
            return

        # Test with a small image
        test_img = Image.new("RGB", (100, 30), color="white")
        try:
            pytesseract.image_to_string(test_img, lang=self.language, config=self.config)
            self._initialized = True
            logger.info("OCR engine initialized", language=self.language, config=self.config)
        except Exception as e:
            logger.error("OCR initialization failed", error=str(e))
            raise

    def recognize(
        self,
        image: np.ndarray | Image.Image,
        detail: int = 0,
    ) -> OCRResult:
        """
        Perform OCR on an image.

        Args:
            image: Image as numpy array (RGB) or PIL Image
            detail: Level of detail (0=text only, 1=dict, 2=dict with word boxes, 3=dict with line boxes)

        Returns:
            OCRResult with text and metadata
        """
        if not self._initialized:
            self.initialize()

        import time
        start = time.perf_counter()

        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image, "RGB")
        else:
            pil_image = image

        # Run OCR
        if detail == 0:
            text = pytesseract.image_to_string(
                pil_image,
                lang=self.language,
                config=self.config,
            )
            words = []
            lines = []
            confidence = 0.0
        else:
            data = pytesseract.image_to_data(
                pil_image,
                lang=self.language,
                config=self.config,
                output_type=pytesseract.Output.DICT,
            )

            # Extract text
            text_parts = []
            confidences = []
            words = []
            lines = []

            for i in range(len(data["text"])):
                if data["text"][i].strip():
                    text_parts.append(data["text"][i])
                    conf = float(data["conf"][i])
                    if conf > 0:
                        confidences.append(conf)

                    words.append({
                        "text": data["text"][i],
                        "confidence": conf,
                        "bbox": {
                            "x": data["left"][i],
                            "y": data["top"][i],
                            "width": data["width"][i],
                            "height": data["height"][i],
                        },
                        "block_num": data["block_num"][i],
                        "par_num": data["par_num"][i],
                        "line_num": data["line_num"][i],
                        "word_num": data["word_num"][i],
                    })

            text = " ".join(text_parts)
            confidence = np.mean(confidences) if confidences else 0.0

            # Group by lines
            line_dict = {}
            for word in words:
                line_key = (word["block_num"], word["par_num"], word["line_num"])
                if line_key not in line_dict:
                    line_dict[line_key] = {"words": [], "text": "", "bbox": None}
                line_dict[line_key]["words"].append(word)
                line_dict[line_key]["text"] += word["text"] + " "

            for line_data in line_dict.values():
                line_data["text"] = line_data["text"].strip()
                # Calculate line bbox
                if line_data["words"]:
                    xs = [w["bbox"]["x"] for w in line_data["words"]]
                    ys = [w["bbox"]["y"] for w in line_data["words"]]
                    ws = [w["bbox"]["width"] for w in line_data["words"]]
                    hs = [w["bbox"]["height"] for w in line_data["words"]]
                    line_data["bbox"] = {
                        "x": min(xs),
                        "y": min(ys),
                        "width": max(xs) + max(ws) - min(xs),
                        "height": max(ys) + max(hs) - min(ys),
                    }
                lines.append(line_data)

        processing_time = time.perf_counter() - start

        return OCRResult(
            text=text.strip(),
            confidence=confidence,
            words=words,
            lines=lines,
            language=self.language,
            processing_time=processing_time,
        )

    def recognize_fast(self, image: np.ndarray | Image.Image) -> str:
        """Fast OCR returning only text."""
        result = self.recognize(image, detail=0)
        return result.text

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        try:
            return pytesseract.get_languages(config=self.config)
        except Exception:
            return []

    def close(self) -> None:
        """Clean up."""
        self._initialized = False