from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import structlog

logger = structlog.get_logger()


class TriggerType(str, Enum):
    """Types of triggers for proactive behavior."""

    AUDIO_SPEECH_DETECTED = "audio_speech_detected"
    AUDIO_SILENCE_TIMEOUT = "audio_silence_timeout"
    VISION_CHANGE_DETECTED = "vision_change_detected"
    VISION_TEXT_DETECTED = "vision_text_detected"
    TIME_INTERVAL = "time_interval"
    SYSTEM_STARTUP = "system_startup"
    USER_IDLE = "user_idle"
    ERROR_OCCURRED = "error_occurred"
    CUSTOM = "custom"


@dataclass
class TriggerCondition:
    """Condition for a trigger to fire."""

    trigger_type: TriggerType
    # Audio conditions
    min_speech_duration_ms: Optional[int] = None
    silence_timeout_ms: Optional[int] = None
    # Vision conditions
    min_change_score: float = 0.05
    ocr_keywords: List[str] = field(default_factory=list)
    # Time conditions
    interval_seconds: Optional[int] = None
    # Custom conditions
    custom_check: Optional[str] = None  # Python expression as string


@dataclass
class TriggerAction:
    """Action to execute when trigger fires."""

    action_type: str  # "speak", "log", "notify_hermes", "run_brain", "custom"
    # Speak parameters
    text: Optional[str] = None
    # Brain parameters
    prompt_template: Optional[str] = None
    use_fallback: bool = False
    # Hermes notification
    hermes_endpoint: Optional[str] = None
    hermes_payload: Optional[Dict[str, Any]] = None
    # Custom
    custom_handler: Optional[str] = None  # Function name to call


@dataclass
class Trigger:
    """A complete trigger definition."""

    id: str
    name: str
    description: str
    condition: TriggerCondition
    action: TriggerAction
    enabled: bool = True
    cooldown_seconds: int = 0  # Minimum time between firings
    max_firings_per_hour: Optional[int] = None
    priority: int = 0  # Higher = more important

    # Runtime state
    _last_fired: float = 0
    _fire_count: int = 0

    def can_fire(self, current_time: float) -> bool:
        """Check if trigger can fire (cooldown, rate limit)."""
        if not self.enabled:
            return False
        if self.cooldown_seconds > 0:
            if current_time - self._last_fired < self.cooldown_seconds:
                return False
        if self.max_firings_per_hour:
            # Simple hour-window rate limiting
            if self._fire_count >= self.max_firings_per_hour:
                return False
        return True

    def mark_fired(self, current_time: float) -> None:
        """Mark trigger as fired."""
        self._last_fired = current_time
        self._fire_count += 1


# Default triggers configuration
DEFAULT_TRIGGERS = [
    Trigger(
        id="speech_detected",
        name="Speech Detected",
        description="User started speaking",
        condition=TriggerCondition(
            trigger_type=TriggerType.AUDIO_SPEECH_DETECTED,
            min_speech_duration_ms=500,
        ),
        action=TriggerAction(
            action_type="run_brain",
            prompt_template="O usuário disse: {transcript}\nResponda de forma natural e concisa.",
        ),
        cooldown_seconds=2,
        priority=10,
    ),
    Trigger(
        id="silence_timeout",
        name="Silence Timeout",
        description="Extended silence after speech",
        condition=TriggerCondition(
            trigger_type=TriggerType.AUDIO_SILENCE_TIMEOUT,
            silence_timeout_ms=30000,  # 30 seconds
        ),
        action=TriggerAction(
            action_type="speak",
            text="Precisa de algo mais?",
        ),
        cooldown_seconds=60,
        priority=5,
    ),
    Trigger(
        id="screen_change",
        name="Screen Change Detected",
        description="Significant visual change on screen",
        condition=TriggerCondition(
            trigger_type=TriggerType.VISION_CHANGE_DETECTED,
            min_change_score=0.15,
        ),
        action=TriggerAction(
            action_type="run_brain",
            prompt_template="Detectei uma mudança na tela. OCR: {ocr_text}\nComente se relevante.",
        ),
        cooldown_seconds=10,
        max_firings_per_hour=20,
        priority=7,
    ),
    Trigger(
        id="text_on_screen",
        name="Text Detected on Screen",
        description="OCR found specific keywords",
        condition=TriggerCondition(
            trigger_type=TriggerType.VISION_TEXT_DETECTED,
            ocr_keywords=["erro", "error", "falha", "falhou", "atenção", "warning"],
        ),
        action=TriggerAction(
            action_type="run_brain",
            prompt_template="Texto importante detectado na tela: {ocr_text}\nAnalise e avise se precisar de ação.",
        ),
        cooldown_seconds=30,
        priority=8,
    ),
    Trigger(
        id="periodic_checkin",
        name="Periodic Check-in",
        description="Proactive check-in every hour",
        condition=TriggerCondition(
            trigger_type=TriggerType.TIME_INTERVAL,
            interval_seconds=3600,
        ),
        action=TriggerAction(
            action_type="speak",
            text="Tudo bem por aí? Precisa de ajuda com algo?",
        ),
        cooldown_seconds=3600,
        priority=3,
    ),
]


def load_triggers_from_yaml(yaml_path: str) -> List[Trigger]:
    """Load triggers from YAML file."""
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    triggers = []
    for t in data.get("triggers", []):
        condition = TriggerCondition(
            trigger_type=TriggerType(t["condition"]["type"]),
            min_speech_duration_ms=t["condition"].get("min_speech_duration_ms"),
            silence_timeout_ms=t["condition"].get("silence_timeout_ms"),
            min_change_score=t["condition"].get("min_change_score", 0.05),
            ocr_keywords=t["condition"].get("ocr_keywords", []),
            interval_seconds=t["condition"].get("interval_seconds"),
            custom_check=t["condition"].get("custom_check"),
        )
        action = TriggerAction(
            action_type=t["action"]["type"],
            text=t["action"].get("text"),
            prompt_template=t["action"].get("prompt_template"),
            use_fallback=t["action"].get("use_fallback", False),
            hermes_endpoint=t["action"].get("hermes_endpoint"),
            hermes_payload=t["action"].get("hermes_payload"),
            custom_handler=t["action"].get("custom_handler"),
        )
        trigger = Trigger(
            id=t["id"],
            name=t["name"],
            description=t.get("description", ""),
            condition=condition,
            action=action,
            enabled=t.get("enabled", True),
            cooldown_seconds=t.get("cooldown_seconds", 0),
            max_firings_per_hour=t.get("max_firings_per_hour"),
            priority=t.get("priority", 0),
        )
        triggers.append(trigger)

    return triggers


def get_default_triggers() -> List[Trigger]:
    """Get the default trigger set."""
    return DEFAULT_TRIGGERS.copy()