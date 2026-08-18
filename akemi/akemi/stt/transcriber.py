import time
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class TranscriptionResult:
    """Result of a transcription."""

    text: str
    language: str
    language_probability: float
    duration: float  # seconds
    segments: List[Dict[str, Any]]
    confidence: float


class Transcriber:
    """Speech-to-text using faster-whisper."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        language: Optional[str] = "pt",
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_parameters: Optional[Dict[str, Any]] = None,
        download_root: Optional[str] = None,
    ):
        """
        Initialize the transcriber.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3)
            device: Device to run on (auto, cpu, cuda)
            compute_type: Quantization (auto, int8, float16, float32)
            language: Language code (None = auto-detect)
            beam_size: Beam size for decoding
            vad_filter: Use VAD filter to remove silence
            vad_parameters: VAD filter parameters
            download_root: Directory to download models to
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.vad_parameters = vad_parameters or {
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "max_speech_duration_s": float("inf"),
            "min_silence_duration_ms": 100,
            "window_size_samples": 1024,
        }
        self.download_root = download_root
        self._model = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load the model."""
        if self._initialized:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError(
                "faster-whisper not installed. Install with: pip install faster-whisper"
            )

        import asyncio
        loop = asyncio.get_event_loop()

        def _load_model():
            return WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
            )

        self._model = await loop.run_in_executor(None, _load_model)
        self._initialized = True
        logger.info(
            "Transcriber initialized",
            model=self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """
        Transcribe audio array.

        Args:
            audio: Audio as float32 numpy array (mono, -1.0 to 1.0)
            sample_rate: Sample rate of audio (must be 16000)

        Returns:
            TranscriptionResult with text and metadata
        """
        if not self._initialized:
            raise RuntimeError("Transcriber not initialized. Call initialize() first.")

        if sample_rate != 16000:
            raise ValueError(f"Sample rate must be 16000, got {sample_rate}")

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32768.0

        start = time.perf_counter()

        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            vad_parameters=self.vad_parameters,
        )

        # Collect segments
        segment_list = []
        full_text = []
        for segment in segments:
            segment_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
            })
            full_text.append(segment.text)

        duration = time.perf_counter() - start

        # Calculate average confidence
        confidences = [seg["avg_logprob"] for seg in segment_list]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return TranscriptionResult(
            text=" ".join(full_text).strip(),
            language=info.language,
            language_probability=info.language_probability,
            duration=duration,
            segments=segment_list,
            confidence=avg_confidence,
        )

    async def transcribe_async(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """Async wrapper for transcribe."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe, audio, sample_rate)

    def transcribe_file(self, file_path: str) -> TranscriptionResult:
        """Transcribe an audio file directly."""
        if not self._initialized:
            raise RuntimeError("Transcriber not initialized. Call initialize() first.")

        start = time.perf_counter()

        segments, info = self._model.transcribe(
            file_path,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            vad_parameters=self.vad_parameters,
        )

        segment_list = []
        full_text = []
        for segment in segments:
            segment_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
            })
            full_text.append(segment.text)

        duration = time.perf_counter() - start
        confidences = [seg["avg_logprob"] for seg in segment_list]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return TranscriptionResult(
            text=" ".join(full_text).strip(),
            language=info.language,
            language_probability=info.language_probability,
            duration=duration,
            segments=segment_list,
            confidence=avg_confidence,
        )

    def is_initialized(self) -> bool:
        return self._initialized

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": self.language,
            "initialized": self._initialized,
        }