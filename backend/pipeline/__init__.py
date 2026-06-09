"""Pipeline package.

Re-exports the three LiveKit plugin adapters lazily so that importing the
package (or any submodule) does not pull in torch / kokoro / faster-whisper
before :mod:`pipeline.core.bootstrap` has had a chance to load ``.env`` and
configure DLL paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["LocalWhisperSTT", "LocalKokoroTTS", "LocalOllamaLLM"]

if TYPE_CHECKING:  # pragma: no cover - type-checker only
    from pipeline.kokoro.plugin import LocalKokoroTTS
    from pipeline.ollama.plugin import LocalOllamaLLM
    from pipeline.whisper.plugin import LocalWhisperSTT


_LAZY = {
    "LocalWhisperSTT": ("pipeline.whisper.plugin", "LocalWhisperSTT"),
    "LocalKokoroTTS": ("pipeline.kokoro.plugin", "LocalKokoroTTS"),
    "LocalOllamaLLM": ("pipeline.ollama.plugin", "LocalOllamaLLM"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'pipeline' has no attribute {name!r}")
    from importlib import import_module

    module = import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value
