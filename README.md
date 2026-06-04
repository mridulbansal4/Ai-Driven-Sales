# AI Sales Trainer — local voice agent (Whisper -> Qwen -> Kokoro)

Same CLI as `Agent/livekit_basic_agent.py`, local pipeline.

```powershell
cd backend
uv sync --python 3.11
uv run python livekit_sales_agent.py download-files
uv run python livekit_sales_agent.py console
```

Or from project root: `.\start.ps1`

See [backend/README.md](backend/README.md) for full setup.
