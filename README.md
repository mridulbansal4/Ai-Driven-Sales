# AI Sales Trainer

**A fully local, voice-first sales training simulator powered by LiveKit, Whisper, Ollama, and Kokoro.**

Bank officers practice live conversations with AI-driven customers — no cloud APIs, no data leaves the machine.

```
Mic → Whisper (STT) → Qwen via Ollama (LLM) → Kokoro (TTS) → Speakers
```

---

## Features

- **Realistic customer personas** — 15 hand-crafted banking scenarios covering home loans, car loans, credit cards, FDs, MSME loans, NRI accounts, and more
- **Adversarial roleplay** — each persona has distinct psychology: analytical engineers who demand exact APR, aggressive hagglers, anxious borrowers, skeptical investors
- **Fully local inference** — Whisper for transcription, Qwen via Ollama for LLM, Kokoro for TTS; zero cloud dependency
- **Multilingual** — auto-detects Hindi vs English per utterance; voice switches automatically
- **LiveKit-powered audio** — low-latency, VAD-gated turn-taking with configurable endpointing
- **Conversia AI dashboard** — React frontend for session management and scenario selection

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Bank Officer (Human)               │
│                  Microphone / Headset               │
└───────────────────────┬─────────────────────────────┘
                        │ audio
                        ▼
              ┌─────────────────┐
              │   Silero VAD    │  voice activity detection
              └────────┬────────┘
                       │ speech segments
                       ▼
              ┌─────────────────┐
              │  LocalWhisperSTT│  faster-whisper, GPU/CPU
              │  (large-v3-turbo│  language auto-detect
              └────────┬────────┘
                       │ transcript
                       ▼
              ┌─────────────────┐
              │  LocalOllamaLLM │  Qwen via Ollama
              │  CustomerAgent  │  persona + scenario context
              └────────┬────────┘
                       │ response text
                       ▼
              ┌─────────────────┐
              │  LocalKokoroTTS │  Kokoro-82M, voice per language
              └────────┬────────┘
                       │ audio
                       ▼
                  Speakers / Headset
```

---

## Prerequisites

| Requirement | Version / Notes |
|---|---|
| Python | **3.11** (not 3.12/3.13) |
| [uv](https://docs.astral.sh/uv/) | Package manager |
| [Ollama](https://ollama.com) | Must be running before you start the agent |
| [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases) | Required by Kokoro TTS |
| NVIDIA GPU (optional) | CUDA accelerates Whisper significantly |

Pull the LLM model before first run:

```powershell
ollama pull qwen-sales
```

---

## Quick Start

### 1. Clone and install dependencies

```powershell
git clone <repo-url>
cd sales-trainer\backend
uv sync --python 3.11
```

### 2. Configure environment

```powershell
Copy-Item .env.example .env
# Edit .env — LiveKit URL, API keys, model names, Whisper device, etc.
```

### 3. Start the agent

From the **project root**, run the launcher:

```powershell
cd ..
.\start.ps1
```

`start.ps1` handles the rest automatically:

- Verifies Ollama is running
- Creates `.env` from `.env.example` if missing
- Creates the virtualenv and syncs deps if needed
- Downloads Whisper, VAD, and Kokoro model files
- Starts the voice agent in console mode

**Console controls:** `Ctrl+B` — toggle text/audio input &nbsp;|&nbsp; `Q` — quit

**Stop the agent:**

```powershell
.\stop.ps1
```

### Manual start (without the launcher)

If you prefer to run commands yourself from `backend/`:

```powershell
uv run python livekit_sales_agent.py download-files
uv run python livekit_sales_agent.py console
```

---

## Environment Variables

```ini
# LiveKit
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=devsecret

# LLM (Ollama)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen-sales
OLLAMA_TEMPERATURE=0.6
OLLAMA_MAX_TOKENS=120

# Speech-to-Text (Whisper)
WHISPER_MODEL=large-v3-turbo
WHISPER_DEVICE=cuda          # or cpu
WHISPER_COMPUTE=int8
WHISPER_LANGUAGE=auto        # auto-detects Hindi/English; force with 'hi' or 'en'

# Text-to-Speech (Kokoro) — optional overrides
# KOKORO_VOICE_EN=af_heart
# KOKORO_VOICE_HI=hf_alpha

# Audio devices — optional substring match (used by start.ps1)
# INPUT_DEVICE=Realtek
# OUTPUT_DEVICE=Realtek

LOG_LEVEL=INFO
```

### Audio devices

List available mics and speakers:

```powershell
cd backend
uv run python livekit_sales_agent.py console --list-devices
```

Pin devices by name or index — `start.ps1` reads these automatically:

```powershell
$env:INPUT_DEVICE = "Realtek"
$env:OUTPUT_DEVICE = "Realtek"
.\start.ps1
```

---

## Other Run Modes

All commands run from `backend/`:

```powershell
# LiveKit worker — dev (file watch / hot reload)
uv run python livekit_sales_agent.py dev

# LiveKit worker — production
uv run python livekit_sales_agent.py start
```

---

## Training Scenarios

| Scenario | Customer | Challenge |
|---|---|---|
| Home Loan — Analytical | Rajesh, software engineer | Demands exact APR, compares HDFC/SBI rates |
| Car Loan — Haggler | Suresh, textile trader | Aggressive price negotiation, fee waivers |
| Credit Card — Skeptic | Anita, marketing manager | Hidden charges paranoia, demands written proof |
| Personal Loan — Medical Emergency | Imran, fleet supervisor | Time-pressed, needs ₹5L disbursed by tomorrow |
| Fixed Deposit — Senior Citizen | Mr. Krishnamurthy, retired teacher | Safety-first, confused by jargon |
| Student First Account | Priya, college student | Worries about minimum balance and small fees |
| ULIP — Investment Skeptic | Vikram, IT consultant | Compares TER vs term plan + mutual fund |
| MSME Working Capital | Harpreet, factory owner | Collateral, CGTMSE, cashflow focus |
| Education Loan — Study Abroad | Lakshmi, government employee | Moratorium, forex disbursement, Section 80E |
| Gold Loan — Distress | Ramesh, farmer | Fears losing family gold, asks about auction |
| Credit Card Balance Transfer | Neha, designer | Escaping 42% APR, EMI conversion maths |
| NRI Accounts | Arjun, Dubai engineer | NRE vs NRO, repatriation rules, 10-day window |
| Two-Wheeler Loan — First Timer | Deepak, call-center employee | No credit history, nervous about rejection |
| Home Loan Balance Transfer + Top-up | Sunita, doctor | Break-even calculation on switching cost |
| Loan Restructuring — Default | Mahesh, restaurant owner | Missed EMIs, wants moratorium not threats |

---

## Frontend — Conversia AI

A React + Tailwind dashboard for session management and scenario browsing.

```powershell
cd frontend
npm install
npm run dev
```

Runs at `http://localhost:5173` by default.

---

## Local vs Cloud Pipeline

| Component | Cloud (reference) | This project |
|---|---|---|
| STT | Deepgram | Whisper (`faster-whisper`, local) |
| LLM | OpenAI GPT | Qwen via Ollama |
| TTS | OpenAI TTS | Kokoro-82M (local) |
| VAD | Silero | Silero (unchanged) |

---

## Project Structure

```
sales-trainer/
├── start.ps1                    # one-click launcher (run from here)
├── stop.ps1                     # stops the agent process
├── backend/
│   ├── livekit_sales_agent.py   # agent entrypoint
│   ├── .env.example
│   └── pipeline/
│       ├── __init__.py          # re-exports LocalWhisperSTT / LocalOllamaLLM / LocalKokoroTTS
│       ├── core/                # CUDA / espeak bootstrap + audio device resolution
│       │   ├── bootstrap.py
│       │   ├── cuda_env.py
│       │   └── audio_devices.py
│       ├── whisper/             # faster-whisper STT (processor + LiveKit plugin)
│       ├── kokoro/              # Kokoro TTS (processor + LiveKit plugin)
│       ├── ollama/              # Ollama LLM (processor + LiveKit plugin)
│       └── livekit/             # CustomerAgent + customer personas
│           ├── agent.py
│           └── scenarios.py
└── frontend/                    # Conversia AI dashboard (React + Tailwind)
```

---

## License

Internal tooling — not licensed for redistribution.
