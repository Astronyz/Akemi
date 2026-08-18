from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn
import structlog

from akemi.akemi.core.config import get_settings
from akemi.akemi.core.pause import pause_controller
from akemi.akemi.storage import get_db
from akemi.akemi.storage.models import EventType, SystemEvent

logger = structlog.get_logger()

# Global reference to orchestrator (set by main)
orchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Control API starting up")
    yield
    logger.info("Control API shutting down")


app = FastAPI(
    title="Akemi Control API",
    description="Local control API for Akemi autonomous agent",
    version="0.1.0",
    lifespan=lifespan,
)


# Request/Response models
class StatusResponse(BaseModel):
    running: bool
    paused: bool
    pause_reason: Optional[str] = None
    uptime_seconds: float
    components: Dict[str, bool]


class PauseRequest(BaseModel):
    reason: str = "Manual pause via API"


class SpeakRequest(BaseModel):
    text: str
    engine: Optional[str] = None


class TriggerRequest(BaseModel):
    trigger_id: str
    context: Optional[Dict[str, Any]] = None


class ConfigResponse(BaseModel):
    audio: Dict[str, Any]
    brain: Dict[str, Any]
    stt: Dict[str, Any]
    tts: Dict[str, Any]
    vision: Dict[str, Any]
    network: Dict[str, Any]


# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "akemi-control-api"}


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current agent status."""
    global orchestrator
    components = {}
    if orchestrator:
        components = orchestrator.get_component_status()

    return StatusResponse(
        running=orchestrator is not None and orchestrator.is_running(),
        paused=pause_controller.is_paused(),
        pause_reason=pause_controller.get_pause_reason(),
        uptime_seconds=orchestrator.get_uptime() if orchestrator else 0.0,
        components=components,
    )


@app.post("/pause")
async def pause_agent(request: PauseRequest):
    """Pause the agent."""
    pause_controller.pause(request.reason)
    # Log event
    db = get_db()
    db.insert_event(SystemEvent(
        event_type=EventType.SYSTEM_PAUSE,
        session_id=orchestrator.session_id if orchestrator else "unknown",
        details=request.reason,
    ))
    return {"status": "paused", "reason": request.reason}


@app.post("/resume")
async def resume_agent():
    """Resume the agent."""
    pause_controller.resume()
    db = get_db()
    db.insert_event(SystemEvent(
        event_type=EventType.SYSTEM_RESUME,
        session_id=orchestrator.session_id if orchestrator else "unknown",
        details="Resumed via API",
    ))
    return {"status": "resumed"}


@app.post("/speak")
async def speak_text(request: SpeakRequest, background_tasks: BackgroundTasks):
    """Make the agent speak text."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not running")

    background_tasks.add_task(orchestrator.speak, request.text, request.engine)
    return {"status": "queued", "text": request.text}


@app.post("/trigger")
async def trigger_action(request: TriggerRequest, background_tasks: BackgroundTasks):
    """Manually trigger an action."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not running")

    background_tasks.add_task(orchestrator.handle_trigger, request.trigger_id, request.context)
    return {"status": "triggered", "trigger_id": request.trigger_id}


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration (sanitized - no secrets)."""
    settings = get_settings()
    return ConfigResponse(
        audio=settings.audio.model_dump(),
        brain={
            "provider": settings.brain.provider,
            "fallback_provider": settings.brain.fallback_provider,
            "model": getattr(settings.brain, 'ollama_model', ''),
            "temperature": settings.brain.temperature,
            "max_tokens": settings.brain.max_tokens,
        },
        stt=settings.stt.model_dump(),
        tts=settings.tts.model_dump(),
        vision=settings.vision.model_dump(),
        network=settings.network.model_dump(),
    )


@app.get("/logs")
async def get_logs(
    event_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """Get recent event logs."""
    db = get_db()
    etype = EventType(event_type) if event_type else None
    events = db.get_events(event_type=etype, limit=limit, offset=offset)
    return {"events": events, "count": len(events)}


@app.get("/stats")
async def get_stats():
    """Get database statistics."""
    db = get_db()
    return db.get_stats()


@app.post("/shutdown")
async def shutdown_agent(background_tasks: BackgroundTasks):
    """Shutdown the agent gracefully."""
    if orchestrator:
        background_tasks.add_task(orchestrator.shutdown)
    return {"status": "shutting_down"}


def create_app() -> FastAPI:
    """Create the FastAPI app (for testing)."""
    return app


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the control API server."""
    uvicorn.run(
        "akemi.akemi.net.control_api:app",
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )