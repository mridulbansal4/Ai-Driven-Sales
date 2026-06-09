"""One-time Windows / CUDA setup before importing torch or kokoro.

Importing this module must be the very first thing the entrypoint does:
  1. Loads ``backend/.env`` (regardless of CWD) so subsequent settings reads
     reflect the user's configuration.
  2. Configures logging from ``LOG_LEVEL``.
  3. On Windows, switches the asyncio event-loop policy and prepends CUDA /
     espeak DLL directories to PATH so torch and kokoro can find them.
"""

from __future__ import annotations

import asyncio
import os
import sys

from pipeline.core.config import configure_logging, load_dotenv_from_backend

load_dotenv_from_backend()
configure_logging()

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import pipeline.core.cuda_env  # noqa: F401  — side effect: PATH/DLL setup
