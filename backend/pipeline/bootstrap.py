"""One-time Windows / CUDA setup before importing torch or kokoro."""

from __future__ import annotations

import asyncio
import os
import sys

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import pipeline.cuda_env  # noqa: F401
