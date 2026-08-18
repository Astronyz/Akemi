import pyaudiowpatch as pyaudio
import numpy as np
from typing import Optional, Callable
import structlog
from scipy.signal import resample_poly

from .ring_buffer import RingBuffer

logger = structlog.get_logger()

# Constantes para VAD
VAD_SAMPLE_RATE = 16000
VAD_FRAME_MS = 20  # 10, 20, ou 30ms
VAD_FRAME_SAMPLES = VAD_SAMPLE_RATE * VAD_FRAME_MS // 1000  # 320 samples
VAD_FRAME_BYTES = VAD_FRAME_SAMPLES * 2  # 16-bit = 2 bytes


class AudioCapture:
    """Captures loopback audio using PyAudioWPatch and resamples to 16kHz mono for VAD."""

    def __init__(
        self,
        chunk_size: int = VAD_FRAME_SAMPLES,  # Alinhado ao frame VAD (320 samples @ 16kHz)
        sample_rate: int = VAD_SAMPLE_RATE,   # Taxa de saída para VAD/STT
        channels: int = 1,                    # Sempre mono na saída
        format: int = pyaudio.paInt16,
        buffer_size_chunks: int = 100,
    ):
        """
        Initialize the audio capture.

        Args:
            chunk_size: Number of frames per buffer (at output sample_rate).
                       Default: 320 (20ms @ 16kHz) - aligned to VAD frame.
            sample_rate: Output sampling rate in Hz (fixed at 16000 for VAD compatibility).
            channels: Number of output audio channels (fixed at 1 for mono).
            format: Audio format (paInt16, etc.).
            buffer_size_chunks: Size of the ring buffer in number of chunks.
        """
        # Forçar parâmetros de saída compatíveis com VAD
        self.chunk_size = VAD_FRAME_SAMPLES
        self.sample_rate = VAD_SAMPLE_RATE
        self.channels = 1
        self.format = format
        self.buffer_size_chunks = buffer_size_chunks

        self._audio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._ring_buffer = RingBuffer(buffer_size_chunks)
        self._is_running = False
        self._callback: Optional[Callable[[bytes], None]] = None

        # Estado para resampling
        self._device_sample_rate: Optional[int] = None
        self._device_channels: Optional[int] = None
        self._resample_buffer: np.ndarray = np.array([], dtype=np.int16)

    def _resample_to_vad_rate(self, audio_data: np.ndarray, in_rate: int, in_channels: int) -> np.ndarray:
        """
        Resample audio to 16kHz mono.

        Args:
            audio_data: Input audio as int16 numpy array (interleaved if multi-channel)
            in_rate: Input sample rate
            in_channels: Input number of channels

        Returns:
            Resampled audio as int16 numpy array (mono, 16kHz)
        """
        # Converter para mono se necessário
        if in_channels > 1:
            # Downmix: média dos canais (melhor que pegar só o esquerdo)
            audio_data = audio_data.reshape(-1, in_channels).mean(axis=1).astype(np.int16)

        # Resample se taxa diferente
        if in_rate != VAD_SAMPLE_RATE:
            # Usar resample_poly para conversão racional de taxa
            # Calcular fatores de up/down sampling
            from math import gcd
            g = gcd(in_rate, VAD_SAMPLE_RATE)
            up = VAD_SAMPLE_RATE // g
            down = in_rate // g
            audio_data = resample_poly(audio_data, up, down).astype(np.int16)

        return audio_data

    def _audio_callback(
        self, in_data, frame_count, time_info, status
    ) -> tuple[bytes, int]:
        """Callback for the audio stream."""
        if status:
            logger.warning("Audio stream status", status=status)

        # Converter bytes para numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)

        # Resample para 16kHz mono
        resampled = self._resample_to_vad_rate(
            audio_data,
            self._device_sample_rate,
            self._device_channels
        )

        # Acumular no buffer de resampling e emitir frames alinhados ao VAD
        self._resample_buffer = np.concatenate([self._resample_buffer, resampled])

        # Processar frames completos de VAD
        while len(self._resample_buffer) >= VAD_FRAME_SAMPLES:
            frame = self._resample_buffer[:VAD_FRAME_SAMPLES]
            self._resample_buffer = self._resample_buffer[VAD_FRAME_SAMPLES:]

            frame_bytes = frame.tobytes()
            self._ring_buffer.append(frame_bytes)

            if self._callback:
                self._callback(frame_bytes)

        return (in_data, pyaudio.paContinue)

    def start(self, callback: Optional[Callable[[bytes], None]] = None) -> None:
        """Start audio capture."""
        if self._is_running:
            logger.warning("Audio capture already started")
            return

        self._callback = callback
        self._audio = pyaudio.PyAudio()

        try:
            # Get the default WASAPI loopback device
            wasapi_info = self._audio.get_host_api_info_by_type(pyaudio.paWASAPIHostApiHostId)
            default_speakers = self._audio.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

            if not default_speakers["isLoopbackDevice"]:
                loopback_device = None
                for loopback in self._audio.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        loopback_device = loopback
                        break
                if loopback_device is None:
                    logger.error("No suitable loopback device found")
                    raise RuntimeError("No suitable loopback device found")
            else:
                loopback_device = default_speakers

            # Guardar taxa e canais do dispositivo para resampling
            self._device_sample_rate = int(loopback_device["defaultSampleRate"])
            self._device_channels = loopback_device["maxInputChannels"]

            logger.info(
                "Using loopback device",
                name=loopback_device["name"],
                device_sample_rate=self._device_sample_rate,
                device_channels=self._device_channels,
                output_sample_rate=self.sample_rate,
                output_channels=self.channels,
                vad_frame_ms=VAD_FRAME_MS,
            )

            # Abrir stream na taxa nativa do dispositivo
            self._stream = self._audio.open(
                format=self.format,
                channels=self._device_channels,
                rate=self._device_sample_rate,
                input=True,
                input_device_index=loopback_device["index"],
                frames_per_buffer=self.chunk_size,  # Tamanho do buffer do driver
                stream_callback=self._audio_callback,
            )
            self._stream.start_stream()
            self._is_running = True
            logger.info("Audio capture started")
        except Exception as e:
            logger.error("Failed to start audio capture", error=str(e))
            self.stop()
            raise

    def stop(self) -> None:
        """Stop audio capture."""
        if not self._is_running:
            return
        self._is_running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._audio:
            self._audio.terminate()
            self._audio = None
        self._ring_buffer.clear()
        self._resample_buffer = np.array([], dtype=np.int16)
        logger.info("Audio capture stopped")

    def get_recent_audio(self, seconds: float = 1.0) -> list[bytes]:
        """Get recent audio chunks from the ring buffer."""
        chunks_needed = int((self.sample_rate * seconds) / self.chunk_size)
        all_chunks = self._ring_buffer.get()
        return all_chunks[-chunks_needed:] if chunks_needed <= len(all_chunks) else all_chunks

    def is_running(self) -> bool:
        """Check if the capture is running."""
        return self._is_running