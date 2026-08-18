import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
import structlog

from akemi.akemi.core.config import get_settings, Settings
from akemi.akemi.core.logging_setup import setup_logging, get_logger
from akemi.akemi.core.pause import pause_controller
from akemi.akemi.storage import get_db, close_db
from akemi.akemi.storage.models import (
    Event, EventType, AudioEvent, VisionEvent, BrainEvent, TTSEvent, SystemEvent
)
from akemi.akemi.audio.capture import AudioCapture
from akemi.akemi.audio.vad import VAD
from akemi.akemi.brain import (
    BrainProviderFactory, BrainProvider, BrainMessage, BrainResponse,
    Trigger, TriggerType, get_default_triggers
)
from akemi.akemi.stt import Transcriber
from akemi.akemi.tts import TTSEngineFactory, TTSEngine
from akemi.akemi.vision import ScreenshotCapture, OCREngine, FrameDifferencer
from akemi.akemi.net import get_hermes_client, close_hermes_client, run_server

logger = structlog.get_logger()


@dataclass
class ComponentStatus:
    """Status of a component."""
    name: str
    healthy: bool
    details: str = ""
    last_error: Optional[str] = None


class Orchestrator:
    """Main orchestrator integrating all Akemi components."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.session_id = str(uuid.uuid4())
        self._running = False
        self._start_time = time.time()

        # Components
        self._audio_capture: Optional[AudioCapture] = None
        self._vad: Optional[VAD] = None
        self._transcriber: Optional[Transcriber] = None
        self._brain: Optional[BrainProvider] = None
        self._fallback_brain: Optional[BrainProvider] = None
        self._tts: Optional[TTSEngine] = None
        self._screenshot: Optional[ScreenshotCapture] = None
        self._ocr: Optional[OCREngine] = None
        self._frame_diff: Optional[FrameDifferencer] = None
        self._hermes = None
        self._api_server_task: Optional[asyncio.Task] = None

        # State
        self._speech_buffer: List[bytes] = []
        self._is_speaking = False
        self._last_speech_time = 0.0
        self._last_screenshot: Optional[Any] = None
        self._trigger_cooldowns: Dict[str, float] = {}

        # Callbacks
        self._on_transcription: Optional[Callable[[str], None]] = None
        self._on_brain_response: Optional[Callable[[str], None]] = None

    @property
    def start_time(self) -> float:
        return _start_time

    def get_uptime(self) -> float:
        return time.time() - self._start_time

    def is_running(self) -> bool:
        return self._running

    def get_component_status(self) -> Dict[str, bool]:
        return {
            "audio_capture": self._audio_capture is not None and self._audio_capture.is_running(),
            "vad": self._vad is not None,
            "transcriber": self._transcriber is not None and self._transcriber.is_initialized(),
            "brain": self._brain is not None,
            "tts": self._tts is not None,
            "screenshot": self._screenshot is not None,
            "ocr": self._ocr is not None,
            "frame_diff": self._frame_diff is not None,
            "hermes": self._hermes is not None,
        }

    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing Akemi orchestrator", session_id=self.session_id)

        # Setup logging
        setup_logging()

        # Initialize database
        get_db()

        # Audio pipeline
        await self._init_audio()

        # Brain
        await self._init_brain()

        # STT
        await self._init_stt()

        # TTS
        await self._init_tts()

        # Vision
        await self._init_vision()

        # Hermes client
        await self._init_hermes()

        # Triggers
        self._triggers = get_default_triggers()

        logger.info("All components initialized")

    async def _init_audio(self) -> None:
        """Initialize audio capture and VAD."""
        audio_cfg = self.settings.audio
        self._vad = VAD(
            aggressiveness=audio_cfg.vad_aggressiveness,
            sample_rate=audio_cfg.sample_rate,
        )

        self._audio_capture = AudioCapture(
            chunk_size=audio_cfg.chunk_size,
            sample_rate=audio_cfg.sample_rate,
            channels=audio_cfg.channels,
            buffer_size_chunks=audio_cfg.buffer_size_chunks,
        )

        # Set callback
        self._audio_capture.start(callback=self._audio_callback)
        logger.info("Audio pipeline started")

    async def _init_brain(self) -> None:
        """Initialize brain provider(s)."""
        brain_cfg = self.settings.brain

        # Primary provider
        self._brain = BrainProviderFactory.create(
            brain_cfg.provider,
            model=getattr(brain_cfg, f"{brain_cfg.provider}_model", brain_cfg.provider) or brain_cfg.provider,
            api_key=getattr(brain_cfg, f"{brain_cfg.provider}_api_key", None),
            host=getattr(brain_cfg, "ollama_host", None),
            model_path=getattr(brain_cfg, "llamacpp_model_path", None),
            n_ctx=getattr(brain_cfg, "llamacpp_n_ctx", 4096),
            n_gpu_layers=getattr(brain_cfg, "llamacpp_n_gpu_layers", -1),
            temperature=brain_cfg.temperature,
            max_tokens=brain_cfg.max_tokens,
        )
        await self._brain.initialize()

        # Fallback provider
        if brain_cfg.fallback_provider != brain_cfg.provider:
            self._fallback_brain = BrainProviderFactory.create(
                brain_cfg.fallback_provider,
                model=getattr(brain_cfg, f"{brain_cfg.fallback_provider}_model", brain_cfg.fallback_provider),
                api_key=getattr(brain_cfg, f"{brain_cfg.fallback_provider}_api_key", None),
                host=getattr(brain_cfg, "ollama_host", None),
                model_path=getattr(brain_cfg, "llamacpp_model_path", None),
                temperature=brain_cfg.temperature,
                max_tokens=brain_cfg.max_tokens,
            )
            await self._fallback_brain.initialize()

        logger.info("Brain providers initialized",
                   primary=brain_cfg.provider,
                   fallback=brain_cfg.fallback_provider)

    async def _init_stt(self) -> None:
        """Initialize STT."""
        stt_cfg = self.settings.stt
        self._transcriber = Transcriber(
            model_size=stt_cfg.model_size,
            device=stt_cfg.device,
            compute_type=stt_cfg.compute_type,
            language=stt_cfg.language,
            beam_size=stt_cfg.beam_size,
            vad_filter=stt_cfg.vad_filter,
            vad_parameters=stt_cfg.vad_parameters,
        )
        await self._transcriber.initialize()
        logger.info("STT initialized", model=stt_cfg.model_size)

    async def _init_tts(self) -> None:
        """Initialize TTS."""
        tts_cfg = self.settings.tts
        self._tts = TTSEngineFactory.create(
            tts_cfg.engine,
            model_path=tts_cfg.piper_model_path,
            speaker_id=tts_cfg.piper_speaker_id,
            length_scale=tts_cfg.piper_length_scale,
            noise_scale=tts_cfg.piper_noise_scale,
            noise_w=tts_cfg.piper_noise_w,
        )
        await self._tts.initialize()
        logger.info("TTS initialized", engine=tts_cfg.engine)

    async def _init_vision(self) -> None:
        """Initialize vision components."""
        vision_cfg = self.settings.vision

        self._screenshot = ScreenshotCapture(
            monitor_index=vision_cfg.monitor_index,
            output_dir="data/screenshots",
        )
        self._screenshot.initialize()

        self._ocr = OCREngine(
            language=vision_cfg.ocr_language,
            config=vision_cfg.ocr_config,
        )
        self._ocr.initialize()

        self._frame_diff = FrameDifferencer(
            threshold=vision_cfg.frame_diff_threshold,
        )
        self._frame_diff.initialize()

        logger.info("Vision initialized")

    async def _init_hermes(self) -> None:
        """Initialize Hermes client."""
        try:
            self._hermes = await get_hermes_client()
            healthy = await self._hermes.health_check()
            if healthy:
                logger.info("Hermes client connected")
            else:
                logger.warning("Hermes client initialized but health check failed")
        except Exception as e:
            logger.warning("Hermes client initialization failed", error=str(e))
            self._hermes = None

    def _audio_callback(self, chunk: bytes) -> None:
        """Callback for audio chunks from capture."""
        # VAD processing
        if self._vad and self._vad.is_speech(chunk):
            self._speech_buffer.append(chunk)
            self._last_speech_time = time.time()
            if not self._is_speaking:
                self._is_speaking = True
                self._handle_trigger(TriggerType.AUDIO_SPEECH_DETECTED, {})
        else:
            # Check for silence timeout
            if self._is_speaking and self._speech_buffer:
                silence_ms = (time.time() - self._last_speech_time) * 1000
                if silence_ms > self.settings.audio.vad_frame_ms * 50:  # ~1 second silence
                    self._process_speech_buffer()

    def _process_speech_buffer(self) -> None:
        """Process accumulated speech buffer."""
        if not self._speech_buffer:
            return

        self._is_speaking = False

        # Combine chunks
        import numpy as np
        audio_data = b"".join(self._speech_buffer)
        self._speech_buffer = []

        # Convert to float32 array for transcriber
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Transcribe in background
        asyncio.create_task(self._transcribe_and_respond(audio_np))

    async def _transcribe_and_respond(self, audio: np.ndarray) -> None:
        """Transcribe audio and get brain response."""
        try:
            # Transcribe
            result = await self._transcriber.transcribe_async(audio)

            if not result.text.strip():
                return

            logger.info("Transcription", text=result.text, confidence=result.confidence)

            # Log event
            db = get_db()
            db.insert_event(AudioEvent(
                event_type=EventType.AUDIO_TRANSCRIPTION,
                session_id=self.session_id,
                transcript=result.text,
                confidence=result.confidence,
                duration_ms=int(result.duration * 1000),
            ))

            # Trigger brain
            await self._get_brain_response(result.text)

        except Exception as e:
            logger.error("Transcription failed", error=str(e))
            self._log_error("transcription", str(e))

    async def _get_brain_response(self, user_text: str) -> None:
        """Get response from brain provider."""
        try:
            messages = [
                BrainMessage(role="system", content=self.settings.brain.system_prompt),
                BrainMessage(role="user", content=user_text),
            ]

            # Try primary
            response = await self._brain.generate(
                messages,
                temperature=self.settings.brain.temperature,
                max_tokens=self.settings.brain.max_tokens,
            )

            # Fallback if needed
            if not response.text and self._fallback_brain:
                logger.warning("Primary brain failed, trying fallback")
                response = await self._fallback_brain.generate(
                    messages,
                    temperature=self.settings.brain.temperature,
                    max_tokens=self.settings.brain.max_tokens,
                )

            if response.text:
                logger.info("Brain response", text=response.text[:100], latency_ms=response.latency_ms)

                # Log event
                db = get_db()
                db.insert_event(BrainEvent(
                    event_type=EventType.BRAIN_RESPONSE,
                    session_id=self.session_id,
                    provider=response.provider,
                    model=response.model,
                    prompt=user_text,
                    response=response.text,
                    tokens_in=response.tokens_in,
                    tokens_out=response.tokens_out,
                    latency_ms=response.latency_ms,
                ))

                # Speak response
                await self.speak(response.text)

                # Notify Hermes
                if self._hermes:
                    await self._hermes.notify_event("brain_response", {
                        "prompt": user_text,
                        "response": response.text,
                        "provider": response.provider,
                    })

        except Exception as e:
            logger.error("Brain response failed", error=str(e))
            self._log_error("brain", str(e))

    async def speak(self, text: str, engine: Optional[str] = None) -> None:
        """Synthesize and play speech."""
        if not self._tts:
            logger.warning("TTS not available")
            return

        try:
            result = await self._tts.synthesize(text)

            # Log event
            db = get_db()
            db.insert_event(TTSEvent(
                event_type=EventType.TTS_SPEAK,
                session_id=self.session_id,
                engine=result.engine,
                text=text,
                duration_ms=int(result.duration * 1000),
            ))

            # Play audio (placeholder - would need audio output)
            logger.info("TTS synthesized", text=text[:50], duration=result.duration)

        except Exception as e:
            logger.error("TTS failed", error=str(e))
            self._log_error("tts", str(e))

    async def _vision_loop(self) -> None:
        """Periodic vision processing loop."""
        interval = self.settings.vision.screenshot_interval

        while self._running:
            await asyncio.sleep(interval)

            if pause_controller.is_paused():
                continue

            try:
                await self._process_vision()
            except Exception as e:
                logger.error("Vision loop error", error=str(e))

    async def _process_vision(self) -> None:
        """Capture screenshot, OCR, and detect changes."""
        # Capture screenshot
        screenshot = self._screenshot.capture(save=True)

        # OCR
        ocr_result = self._ocr.recognize(screenshot.image, detail=1)

        # Frame diff
        diff_result = None
        if self._last_screenshot is not None:
            diff_result = self._frame_diff.compare(
                screenshot.image,
                self._last_screenshot.image,
            )

        self._last_screenshot = screenshot

        # Log vision event
        db = get_db()
        db.insert_event(VisionEvent(
            event_type=EventType.VISION_SCREENSHOT,
            session_id=self.session_id,
            screenshot_path=screenshot.file_path or "",
            ocr_text=ocr_result.text[:500],
            change_score=diff_result.change_score if diff_result else 0.0,
            monitor_index=screenshot.monitor_index,
        ))

        # Check triggers
        if diff_result and diff_result.has_change:
            self._handle_trigger(TriggerType.VISION_CHANGE_DETECTED, {
                "change_score": diff_result.change_score,
                "regions": diff_result.changed_regions,
                "ocr_text": ocr_result.text,
            })

        # Check for keywords in OCR
        for keyword in ["erro", "error", "falha", "falhou", "atenção", "warning"]:
            if keyword.lower() in ocr_result.text.lower():
                self._handle_trigger(TriggerType.VISION_TEXT_DETECTED, {
                    "keyword": keyword,
                    "ocr_text": ocr_result.text,
                })
                break

    def _handle_trigger(self, trigger_type: TriggerType, context: Dict[str, Any]) -> None:
        """Check and fire matching triggers."""
        current_time = time.time()

        for trigger in self._triggers:
            if trigger.condition.trigger_type != trigger_type:
                continue

            if not trigger.can_fire(current_time):
                continue

            # Check additional conditions
            if not self._check_trigger_condition(trigger, context):
                continue

            trigger.mark_fired(current_time)
            asyncio.create_task(self._execute_trigger_action(trigger, context))

    def _check_trigger_condition(self, trigger: Trigger, context: Dict[str, Any]) -> bool:
        """Check additional trigger conditions."""
        cond = trigger.condition

        if trigger_type := TriggerType.VISION_CHANGE_DETECTED:
            if cond.min_change_score and context.get("change_score", 0) < cond.min_change_score:
                return False

        if trigger_type := TriggerType.VISION_TEXT_DETECTED:
            if cond.ocr_keywords:
                ocr_text = context.get("ocr_text", "").lower()
                if not any(kw.lower() in ocr_text for kw in cond.ocr_keywords):
                    return False

        return True

    async def _execute_trigger_action(self, trigger: Trigger, context: Dict[str, Any]) -> None:
        """Execute trigger action."""
        action = trigger.action

        try:
            if action.action_type == "speak" and action.text:
                await self.speak(action.text)

            elif action.action_type == "run_brain" and action.prompt_template:
                prompt = action.prompt_template.format(**context)
                await self._get_brain_response(prompt)

            elif action.action_type == "notify_hermes" and self._hermes:
                await self._hermes.notify_event(
                    action.hermes_endpoint or "trigger",
                    action.hermes_payload or context,
                )

        except Exception as e:
            logger.error("Trigger action failed", trigger_id=trigger.id, error=str(e))

    def _log_error(self, error_type: str, message: str) -> None:
        """Log error to database."""
        db = get_db()
        db.insert_event(SystemEvent(
            event_type=EventType.ERROR,
            session_id=self.session_id,
            details=f"{error_type}: {message}",
        ))

    async def run(self) -> None:
        """Main run loop."""
        self._running = True

        # Log startup
        db = get_db()
        db.insert_event(SystemEvent(
            event_type=EventType.SYSTEM_START,
            session_id=self.session_id,
            details=f"Akemi started (session: {self.session_id})",
        ))

        # Start vision loop
        vision_task = asyncio.create_task(self._vision_loop())

        # Start control API server
        net_cfg = self.settings.network
        self._api_server_task = asyncio.create_task(
            asyncio.to_thread(run_server, net_cfg.control_api_host, net_cfg.control_api_port)
        )

        logger.info("Akemi running", session_id=self.session_id)

        try:
            # Keep running
            while self._running:
                await asyncio.sleep(1)

                # Wait if paused
                if pause_controller.is_paused():
                    pause_controller.wait_if_paused(timeout=1.0)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        if not self._running:
            return

        self._running = False
        logger.info("Shutting down Akemi...")

        # Stop audio
        if self._audio_capture:
            self._audio_capture.stop()

        # Cancel tasks
        for task in [self._api_server_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close components
        if self._brain:
            await self._brain.close()
        if self._fallback_brain:
            await self._fallback_brain.close()
        if self._tts:
            await self._tts.close()
        if self._screenshot:
            self._screenshot.close()
        if self._ocr:
            self._ocr.close()
        if self._frame_diff:
            self._frame_diff.close()

        # Close connections
        await close_hermes_client()
        close_db()

        # Log shutdown
        db = get_db()
        db.insert_event(SystemEvent(
            event_type=EventType.SYSTEM_STOP,
            session_id=self.session_id,
            details="Akemi stopped gracefully",
        ))

        logger.info("Akemi shutdown complete")


# Global orchestrator instance
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Optional[Orchestrator]:
    return _orchestrator


async def create_orchestrator(settings: Optional[Settings] = None) -> Orchestrator:
    """Create and initialize the global orchestrator."""
    global _orchestrator
    _orchestrator = Orchestrator(settings)
    await _orchestrator.initialize()
    return _orchestrator


async def run_akemi(settings: Optional[Settings] = None) -> None:
    """Convenience function to create and run Akemi."""
    orchestrator = await create_orchestrator(settings)
    await orchestrator.run()