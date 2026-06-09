"""Resolve audio device names to a unique sounddevice index."""

from __future__ import annotations

import sys


def resolve_device(name_or_id: str | int, kind: str) -> int:
    """Return a single device index for input or output."""
    import sounddevice as sd

    if isinstance(name_or_id, int):
        return name_or_id

    text = str(name_or_id).strip()
    if text.isdigit():
        return int(text)

    needle = text.lower()
    matches: list[tuple[int, str, str]] = []

    for idx, dev in enumerate(sd.query_devices()):
        if kind == "input" and dev["max_input_channels"] < 1:
            continue
        if kind == "output" and dev["max_output_channels"] < 1:
            continue

        name = dev["name"]
        host = sd.query_hostapis(dev["hostapi"])["name"]
        if needle in name.lower():
            matches.append((idx, name, host))

    if not matches:
        raise SystemExit(
            f"No {kind} device matching {text!r}. "
            "Run: uv run python livekit_sales_agent.py console --list-devices"
        )

    if len(matches) == 1:
        return matches[0][0]

    default_idx = sd.default.device[0 if kind == "input" else 1]
    for idx, _, _ in matches:
        if idx == default_idx:
            return idx

    def score(item: tuple[int, str, str]) -> tuple[int, int]:
        _, _, host = item
        host_l = host.lower()
        if "wasapi" in host_l:
            return (0, item[0])
        if "directsound" in host_l:
            return (1, item[0])
        if "mme" in host_l:
            return (2, item[0])
        return (3, item[0])

    matches.sort(key=score)
    return matches[0][0]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python -m pipeline.core.audio_devices <name-or-id> input|output", file=sys.stderr)
        raise SystemExit(2)
    print(resolve_device(sys.argv[1], sys.argv[2]))
