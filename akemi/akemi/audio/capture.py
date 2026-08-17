import pyaudiowpatch as pyaudio
import numpy as np
from typing import Optional, Callable
import structlog

from .ring_buffer import RingBuffer

logger = structlog.get_logger()


class AudioCapture:
    """Captures loopback audio using PyAudioWPatch."""

    def __init__(
        self,
        chunk_size: int = 1024,
        sample_rate: int = 16000,
        channels: int = 1,
        format: int = pyaudio.paInt16,
        buffer_size_chunks: int = 100,
    ):
        """Initialize the audio capture.

        Args:
            chunk_size: Number of frames per buffer.
            sample_rate: Sampling rate in Hz.
            channels: Number of audio channels.
            format: Audio format (paInt16, etc.).
            buffer_size_chunks: Size of the ring buffer in number of chunks.
        """
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = format
        self.buffer_size_chunks = buffer_size_chunks

        self._audio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._ring_buffer = RingBuffer(buffer_size_chunks)
        self._is_running = False
        self._callback: Optional[Callable[[bytes], None]] = None

    def _audio_callback(
        self, in_data, frame_count, time_info, status
    ) -> tuple[bytes, int]:
        """Callback for the audio stream.

        Args:
            in_data: Input audio data as bytes.
            frame_count: Number of frames in the buffer.
            time_info: Time information.
            status: Status flag.

        Returns:
            Tuple of (data, flag) where flag is pyaudio.paContinue.
        """
        if status:
            logger.warning("Audio stream status", status=status)
        # Convert to mono if needed
        if self.channels > 1:
            # Assuming stereo, take left channel
            # Note: This is a simple conversion; for production, consider proper downmixing.
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            audio_data = audio_data[:: self.channels]  # Take every nth sample for first channel
            in_data = audio_data.tobytes()
        self._ring_buffer.append(in_data)
        if self._callback:
            self._callback(in_data)
        return (in_data, pyaudio.paContinue)

    def start(self, callback: Optional[Callable[[bytes], None]] = None) -> None:
        """Start audio capture.

        Args:
            callback: Optional function to call with each audio chunk.
        """
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
                for loopback in self._audio.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        loopback_device = loopback
                        break
                else:
                    logger.error("No suitable loopback device found")
                    raise RuntimeError("No suitable loopback device found")
            else:
                loopback_device = default_speakers

            logger.info(
                "Using loopback device",
                name=loopback_device["name"],
                sample_rate=int(loopback_device["defaultSampleRate"]),
                channels=loopback_device["maxInputChannels"],
            )

            self._stream = self._audio.open(
                format=self.format,
                channels=self.channels,
                rate=int(loopback_device["defaultSampleRate"]),
                input=True,
                input_device_index=loopback_device["index"],
                frames_per_buffer=self.chunk_size,
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
        logger.info("Audio capture stopped")

    def get_recent_audio(self, seconds: float = 1.0) -> list[bytes]:
        """Get recent audio chunks from the ring buffer.

        Args:
            seconds: Number of seconds of audio to retrieve.

        Returns:
            List of audio chunks.
        """
        # Calculate how many chunks we need
        chunks_needed = int((self.sample_rate * seconds) / self.chunk_size)
        # Get all chunks and return the last `chunks_needed`
        all_chunks = self._ring_buffer.get()
        return all_chunks[-chunks_needed:] if chunks_needed <= len(all_chunks) else all_chunks

    def is_running(self) -> bool:
        """Check if the capture is running."""
        return self._is_running