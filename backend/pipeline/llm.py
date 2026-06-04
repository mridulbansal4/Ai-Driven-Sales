import os
import logging
import httpx

logger = logging.getLogger(__name__)


class LLMProcessor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        self.model = os.getenv('OLLAMA_MODEL', 'qwen2.5-coder:7b')
        self.system_prompt = (
            "You are a hesitant banking customer considering a home loan. "
            "You have concerns about interest rates and monthly EMIs. "
            "Keep responses short — maximum 2-3 sentences. "
            "Stay in character at all times.\n\n"
            "IMPORTANT — Language rule: "
            "Detect the language the user spoke in (Hindi or English). "
            "If the user spoke in Hindi (or Hinglish), reply ONLY in Hindi (Devanagari script). "
            "If the user spoke in English, reply ONLY in English. "
            "Never mix scripts in a single reply. "
            "Do not translate the user's message — just respond naturally in the same language."
        )
        self._initialized = True

    async def generate_full(self, user_text: str, system_prompt: str = None) -> str:
        prompt = system_prompt if system_prompt is not None else self.system_prompt
        return await self.generate_messages([
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ])

    async def generate_messages(self, messages: list[dict]) -> str:
        normalized = []
        for msg in messages:
            role = msg.get('role', 'user')
            if role == 'developer':
                role = 'system'
            content = (msg.get('content') or '').strip()
            if content:
                normalized.append({'role': role, 'content': content})

        if not any(m['role'] == 'system' for m in normalized):
            normalized.insert(0, {'role': 'system', 'content': self.system_prompt})

        body = {'model': self.model, 'messages': normalized, 'stream': False}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json=body,
                    timeout=120.0,
                )
                resp.raise_for_status()
                return resp.json()['message']['content']
        except httpx.HTTPStatusError as e:
            logger.error('Ollama HTTP %s: %s', e.response.status_code, e.response.text)
        except (httpx.ConnectError, OSError) as e:
            logger.error('Ollama connection error: %s', e)
        return "Sorry, I'm having trouble responding right now."