import sys
from pathlib import Path
import structlog
from structlog.stdlib import ProcessorFormatter

from akemi.akemi.core.config import get_settings


def setup_logging() -> None:
    """Configure structlog with JSON or console output."""
    settings = get_settings()
    log_settings = settings.logging

    # Processors para formatação
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if log_settings.format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # Configurar structlog
    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configurar logging stdlib
    import logging
    from logging.handlers import RotatingFileHandler

    root_logger = logging.getLogger()
    root_logger.setLevel(log_settings.level)

    # Remover handlers existentes
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Handler de console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_settings.level)
    console_formatter = ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Handler de arquivo (se configurado)
    if log_settings.file_path:
        log_path = Path(log_settings.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=log_settings.max_bytes,
            backupCount=log_settings.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_settings.level)
        file_formatter = ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=shared_processors,
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Silenciar loggers barulhentos
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str = None):
    """Get a structlog logger instance."""
    return structlog.get_logger(name)