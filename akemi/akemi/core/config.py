import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AudioSettings(BaseSettings):
    """Audio capture settings."""

    sample_rate: int = Field(default=16000, description="Output sample rate for VAD/STT (Hz)")
    channels: int = Field(default=1, description="Output channels (mono=1)")
    vad_aggressiveness: int = Field(default=2, ge=0, le=3, description="VAD aggressiveness 0-3")
    vad_frame_ms: int = Field(default=20, description="VAD frame duration in ms (10, 20, or 30)")
    chunk_size: int = Field(default=320, description="Frames per callback (aligned to VAD frame)")
    buffer_size_chunks: int = Field(default=100, description="Ring buffer size in chunks")

    model_config = SettingsConfigDict(env_prefix="AUDIO_")


class BrainSettings(BaseSettings):
    """LLM brain provider settings."""

    provider: Literal["anthropic", "openai", "ollama", "llamacpp"] = Field(
        default="anthropic", description="Primary brain provider"
    )
    fallback_provider: Literal["anthropic", "openai", "ollama", "llamacpp"] = Field(
        default="ollama", description="Fallback provider if primary fails"
    )

    # Cloud API keys
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")

    # Local model settings
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama host URL")
    ollama_model: str = Field(default="llama3.1:8b", description="Ollama model name")
    llamacpp_model_path: Optional[str] = Field(default=None, description="Path to .gguf model file")
    llamacpp_n_ctx: int = Field(default=4096, description="llama.cpp context size")
    llamacpp_n_gpu_layers: int = Field(default=-1, description="GPU layers (-1 = all)")

    # Generation parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    system_prompt: str = Field(
        default="Você é a Akemi, um assistente pessoal autônomo que roda em background no Windows. "
                "Responda em português de forma natural e concisa.",
        description="System prompt for the LLM"
    )

    model_config = SettingsConfigDict(env_prefix="BRAIN_")


class STTSettings(BaseSettings):
    """Speech-to-text settings."""

    model_size: Literal["tiny", "base", "small", "medium", "large-v3"] = Field(
        default="base", description="faster-whisper model size"
    )
    device: Literal["auto", "cpu", "cuda"] = Field(default="auto", description="Inference device")
    compute_type: Literal["auto", "int8", "float16", "float32"] = Field(
        default="auto", description="Quantization type"
    )
    language: Optional[str] = Field(default="pt", description="Language code (None = auto-detect)")
    beam_size: int = Field(default=5, ge=1, description="Beam size for decoding")
    vad_filter: bool = Field(default=True, description="Use VAD filter in whisper")
    vad_parameters: dict = Field(
        default_factory=lambda: {"threshold": 0.5, "min_speech_duration_ms": 250},
        description="VAD filter parameters"
    )

    model_config = SettingsConfigDict(env_prefix="STT_")


class TTSSettings(BaseSettings):
    """Text-to-speech settings."""

    engine: Literal["piper", "cloud"] = Field(default="piper", description="TTS engine")
    # Piper (local)
    piper_model_path: Optional[str] = Field(default=None, description="Path to Piper .onnx model")
    piper_speaker_id: int = Field(default=0, description="Piper speaker ID")
    piper_length_scale: float = Field(default=1.0, description="Speech speed (1.0 = normal)")
    piper_noise_scale: float = Field(default=0.667, description="Noise scale")
    piper_noise_w: float = Field(default=0.8, description="Noise width")
    # Cloud (placeholder for future)
    cloud_provider: Optional[str] = Field(default=None, description="Cloud TTS provider")
    cloud_api_key: Optional[str] = Field(default=None, description="Cloud TTS API key")
    cloud_voice: Optional[str] = Field(default=None, description="Cloud TTS voice")

    model_config = SettingsConfigDict(env_prefix="TTS_")


class VisionSettings(BaseSettings):
    """Screen capture / vision settings."""

    screenshot_interval: float = Field(default=1.0, ge=0.1, description="Seconds between screenshots")
    screenshot_cache_size: int = Field(default=10, ge=1, description="Number of screenshots to keep in memory")
    ocr_language: str = Field(default="por", description="Tesseract OCR language code")
    ocr_config: str = Field(default="--psm 6", description="Tesseract config options")
    frame_diff_threshold: float = Field(default=0.05, description="Frame difference threshold (0-1)")
    monitor_index: int = Field(default=0, description="Monitor index to capture (0 = primary)")

    model_config = SettingsConfigDict(env_prefix="VISION_")


class NetworkSettings(BaseSettings):
    """Network / API settings."""

    control_api_host: str = Field(default="127.0.0.1", description="Control API bind address")
    control_api_port: int = Field(default=8765, description="Control API port")
    hermes_api_url: str = Field(default="http://localhost:5000", description="Hermes agent base URL")
    hermes_timeout: float = Field(default=30.0, description="Hermes API timeout (seconds)")

    model_config = SettingsConfigDict(env_prefix="NET_")


class StorageSettings(BaseSettings):
    """Database/storage settings."""

    db_path: str = Field(default="data/akemi.db", description="SQLite database path")
    log_retention_days: int = Field(default=30, description="Days to keep logs in DB")
    max_audio_logs: int = Field(default=10000, description="Max audio event logs to keep")
    max_vision_logs: int = Field(default=5000, description="Max vision event logs to keep")

    model_config = SettingsConfigDict(env_prefix="STORAGE_")


class SelfImproveSettings(BaseSettings):
    """Self-improvement / auto-PR settings."""

    enabled: bool = Field(default=False, description="Enable self-improvement")
    interval_hours: int = Field(default=24, ge=1, description="Hours between improvement runs")
    github_token: Optional[str] = Field(default=None, description="GitHub token for creating PRs")
    github_repository: str = Field(default="astronyz/akemi", description="Target repository for PRs")
    max_prs_per_day: int = Field(default=1, description="Maximum PRs per day")
    test_before_pr: bool = Field(default=True, description="Run tests before opening PR")
    rollback_on_failure: bool = Field(default=True, description="Auto-rollback if service crashes after merge")

    model_config = SettingsConfigDict(env_prefix="SELF_IMPROVE_")


class LoggingSettings(BaseSettings):
    """Logging settings."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    format: Literal["json", "console"] = Field(default="json", description="Log format")
    file_path: Optional[str] = Field(default="logs/akemi.log", description="Log file path (None = stdout only)")
    max_bytes: int = Field(default=10_000_000, description="Max log file size before rotation")
    backup_count: int = Field(default=5, description="Number of rotated log files to keep")

    model_config = SettingsConfigDict(env_prefix="LOG_")


class Settings(BaseSettings):
    """Main settings aggregator."""

    audio: AudioSettings = Field(default_factory=AudioSettings)
    brain: BrainSettings = Field(default_factory=BrainSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    network: NetworkSettings = Field(default_factory=NetworkSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    self_improve: SelfImproveSettings = Field(default_factory=SelfImproveSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Global
    app_name: str = "Akemi"
    version: str = "0.1.0"
    debug: bool = Field(default=False, description="Debug mode")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("logging", mode="before")
    @classmethod
    def ensure_log_dir(cls, v):
        if isinstance(v, LoggingSettings) and v.file_path:
            Path(v.file_path).parent.mkdir(parents=True, exist_ok=True)
        return v


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment (useful for testing)."""
    global settings
    settings = Settings()
    return settings