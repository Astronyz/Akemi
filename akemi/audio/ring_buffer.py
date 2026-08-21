from __future__ import annotations

import collections
from typing import Deque


class RingBuffer:
    """A ring buffer for audio chunks."""

    def __init__(self, max_size_chunks: int):
        if max_size_chunks <= 0:
            raise ValueError("max_size_chunks must be positive")
        self._buffer: Deque[bytes] = collections.deque(maxlen=max_size_chunks)

    def append(self, chunk: bytes) -> None:
        self._buffer.append(chunk)

    def get(self) -> list[bytes]:
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def is_empty(self) -> bool:
        return len(self._buffer) == 0
