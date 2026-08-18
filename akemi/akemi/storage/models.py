from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class EventType(str, Enum):
    """Types of events stored in the database."""

    AUDIO_SPEECH_START = "audio_speech_start"
    AUDIO_SPEECH_END = "audio_speech_end"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    VISION_SCREENSHOT = "vision_screenshot"
    VISION_OCR = "vision_ocr"
    VISION_CHANGE_DETECTED = "vision_change_detected"
    BRAIN_REQUEST = "brain_request"
    BRAIN_RESPONSE = "brain_response"
    BRAIN_ERROR = "brain_error"
    TTS_SPEAK = "tts_speak"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SYSTEM_PAUSE = "system_pause"
    SYSTEM_RESUME = "system_resume"
    ERROR = "error"


@dataclass
class BaseEvent:
    """Base event model."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: EventType = EventType.ERROR
    session_id: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, Enum):
                data[key] = value.value
            else:
                data[key] = value
        return data


@dataclass
class AudioEvent(BaseEvent):
    """Audio-related event."""

    event_type: EventType = EventType.AUDIO_TRANSCRIPTION
    transcript: str = ""
    confidence: float = 0.0
    duration_ms: int = 0
    audio_level: float = 0.0


@dataclass
class VisionEvent(BaseEvent):
    """Vision/screen capture event."""

    event_type: EventType = EventType.VISION_SCREENSHOT
    screenshot_path: str = ""
    ocr_text: str = ""
    change_score: float = 0.0
    monitor_index: int = 0


@dataclass
class BrainEvent(BaseEvent):
    """LLM brain event."""

    event_type: EventType = EventType.BRAIN_REQUEST
    provider: str = ""
    model: str = ""
    prompt: str = ""
    response: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    error: str = ""


@dataclass
class TTSEvent(BaseEvent):
    """TTS event."""

    event_type: EventType = EventType.TTS_SPEAK
    engine: str = ""
    text: str = ""
    audio_path: str = ""
    duration_ms: int = 0


@dataclass
class SystemEvent(BaseEvent):
    """System lifecycle event."""

    event_type: EventType = EventType.SYSTEM_START
    details: str = ""


@dataclass
class ErrorEvent(BaseEvent):
    """Error event."""

    event_type: EventType = EventType.ERROR
    error_type: str = ""
    message: str = ""
    traceback: str = ""
    context: str = ""


# Union type for all events
Event = AudioEvent | VisionEvent | BrainEvent | TTSEvent | SystemEvent | ErrorEvent


def create_table_schema() -> str:
    """Return SQL schema for events table."""
    return """
    CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        event_type TEXT NOT NULL,
        session_id TEXT NOT NULL,
        -- Audio fields
        transcript TEXT,
        confidence REAL,
        duration_ms INTEGER,
        audio_level REAL,
        -- Vision fields
        screenshot_path TEXT,
        ocr_text TEXT,
        change_score REAL,
        monitor_index INTEGER,
        -- Brain fields
        provider TEXT,
        model TEXT,
        prompt TEXT,
        response TEXT,
        tokens_in INTEGER,
        tokens_out INTEGER,
        latency_ms INTEGER,
        error TEXT,
        -- TTS fields
        engine TEXT,
        text TEXT,
        audio_path TEXT,
        -- System/Error fields
        details TEXT,
        error_type TEXT,
        message TEXT,
        traceback TEXT,
        context TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
    CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
    CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
    """