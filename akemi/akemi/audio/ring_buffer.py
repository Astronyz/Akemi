import collections
from typing import Deque, Optional


class RingBuffer:
    """A ring buffer for audio chunks."""

    def __init__(self, max_size_chunks: int):
        """Initialize the ring buffer.

        Args:
            max_size_chunks: Maximum number of chunks to store.
        """
        self._buffer: Deque[bytes] = collections.deque(maxlen=max_size_chunks)

    def append(self, chunk: bytes) -> None:
        """Append a chunk to the buffer.

        Args:
            chunk: Audio chunk as bytes.
        """
        self._buffer.append(chunk)

    def get(self) -> list[bytes]:
        """Get a copy of all chunks in the buffer.

        Returns:
            List of chunks in the order they were added.
        """
        return list(self._buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        """Return the number of chunks in the buffer."""
        return len(self._buffer)

    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        return len(self._buffer) == 0