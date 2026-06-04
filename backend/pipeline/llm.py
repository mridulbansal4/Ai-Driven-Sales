import os
import re
import logging
import httpx

logger = logging.getLogger(__name__)

# Replies that mean the model broke character.
_OUT_OF_CHARACTER = re.compile(
    r"language model|as an ai|i'?m an ai|openai|assistant here to help|"
    r"i cannot|i can't help|how can i assist",
    re.IGNORECASE,
)

_FALLBACK = "I'm not sure I follow — could you explain that again in simpler terms?"


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
        self.model = (os.getenv('OLLAMA_MODEL') or 'qwen2.5:7b').strip()
        if not self.model:
            self.model = 'qwen2.5:7b'
        self._initialized = True

    def _build_messages(self, messages: list[dict]) -> list[dict]:
        system_parts: list[str] = []
        conversation: list[dict] = []

        for msg in messages:
            role = msg.get('role', 'user')
            if role == 'developer':
                role = 'system'
            content = (msg.get('content') or '').strip()
            if not content:
                continue
            if role == 'system':
                system_parts.append(content)
            elif role in ('user', 'assistant'):
                conversation.append({'role': role, 'content': content})

        if not system_parts:
            system_parts.append(
                "You are a banking customer. The user is the bank officer. "
                "Stay in character. Reply in 2-3 sentences."
            )

        out = [{'role': 'system', 'content': '\n\n'.join(system_parts)}]
        out.extend(conversation)
        return out

    @staticmethod
    def _clean_reply(text: str) -> str:
        text = text.strip()
        if not text:
            return ''
        if _OUT_OF_CHARACTER.search(text):
            return ''
        return text

    async def generate_messages(self, messages: list[dict]) -> str:
        body_messages = self._build_messages(messages)
        body = {
            'model': self.model,
            'messages': body_messages,
            'stream': False,
            'options': {
                'temperature': float(os.getenv('OLLAMA_TEMPERATURE', '0.6')),
                'num_predict': int(os.getenv('OLLAMA_MAX_TOKENS', '120')),
            },
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json=body,
                    timeout=120.0,
                )
                resp.raise_for_status()
                text = self._clean_reply(resp.json()['message']['content'])
                if text:
                    return text

                # One retry with a stricter reminder.
                retry_messages = body_messages + [
                    {
                        'role': 'user',
                        'content': (
                            '[Stay in character as the banking customer only. '
                            'Reply in 2-3 sentences to what the officer just said.]'
                        ),
                    }
                ]
                resp2 = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={**body, 'messages': retry_messages},
                    timeout=120.0,
                )
                resp2.raise_for_status()
                text2 = self._clean_reply(resp2.json()['message']['content'])
                return text2 or _FALLBACK

        except httpx.HTTPStatusError as e:
            logger.error('Ollama HTTP %s: %s', e.response.status_code, e.response.text)
        except (httpx.ConnectError, OSError) as e:
            logger.error('Ollama connection error: %s', e)
        return _FALLBACK
