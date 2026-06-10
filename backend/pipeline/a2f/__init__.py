"""NVIDIA ACE Audio2Face gRPC bridge.

Streams Kokoro TTS audio to a remote Audio2Face microservice so the same
utterance that goes to LiveKit also drives a MetaHuman avatar.

Public API is intentionally tiny:

    from pipeline.a2f import Audio2FaceClient, get_client

    client = get_client()           # process-wide singleton
    await client.animate(wav_bytes) # fan out one utterance
"""

from __future__ import annotations

from pipeline.a2f.client import Audio2FaceClient, get_client

__all__ = ["Audio2FaceClient", "get_client"]
