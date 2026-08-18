#!/usr/bin/env python3
"""
Akemi - Autonomous AI Agent for Windows
Main entry point.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from akemi.akemi.core import run_akemi, get_settings
from akemi.akemi.core.logging_setup import setup_logging


async def main():
    """Main entry point."""
    # Load settings (also validates .env)
    settings = get_settings()

    # Setup logging early
    setup_logging()

    print(f"Starting {settings.app_name} v{settings.version}")
    print(f"Session ID: {settings.debug and 'debug' or 'production'}")

    try:
        await run_akemi(settings)
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Fatal error: {e}")
        if settings.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Windows event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())