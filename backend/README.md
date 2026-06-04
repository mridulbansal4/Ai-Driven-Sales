# LiveKit Sales Trainer — local voice agent (same layout as ../Agent/)

A LiveKit voice agent using **your local pipeline** instead of OpenAI/Deepgram:

```
Mic → Whisper → Qwen (Ollama) → Kokoro → Speakers
```

## Prerequisites

- Python **3.11** (not 3.13)
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com) running (`ollama pull qwen2.5-coder:7b`)
- [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases) installed

## Quick Start

### 1. Install dependencies

```bash
cd backend
uv sync --python 3.11
```

### 2. Environment

```bash
cp .env.example .env
```

### 3. Download model files (Silero VAD)

```bash
uv run python livekit_sales_agent.py download-files
```

### 4. Run — console mode (same as Agent reference)

```bash
uv run python livekit_sales_agent.py console
```

Or from project root: `.\start.ps1`

**Controls:** `Ctrl+B` text/audio · `Q` quit

### Audio devices

```bash
uv run python livekit_sales_agent.py console --list-devices
```

```powershell
$env:INPUT_DEVICE = "1"
$env:OUTPUT_DEVICE = "5"
uv run python livekit_sales_agent.py console
```

## vs Agent/ reference

| Agent/livekit_basic_agent.py | backend/livekit_sales_agent.py |
|------------------------------|--------------------------------|
| `deepgram.STT()`             | `LocalWhisperSTT()`            |
| `openai.LLM()`               | `LocalOllamaLLM()`             |
| `openai.TTS()`               | `LocalKokoroTTS()`             |
| `silero.VAD.load()`          | `silero.VAD.load()`            |
| `session.generate_reply()`   | `session.generate_reply()`     |

## Other commands (same CLI as reference)

```bash
uv run python livekit_sales_agent.py dev      # LiveKit worker mode
uv run python livekit_sales_agent.py start    # production worker
```
