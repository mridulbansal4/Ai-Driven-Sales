"""Windows PATH setup: CUDA DLLs + espeak-ng (must run before torch/kokoro)."""

from __future__ import annotations

import logging
import os
import site
import subprocess
from pathlib import Path

from pipeline.core.config import settings

logger = logging.getLogger(__name__)


def _prepend_path(*dirs: Path) -> None:
    prefix: list[str] = []
    for directory in dirs:
        if not directory.is_dir():
            continue
        path_str = str(directory)
        if path_str not in prefix:
            prefix.append(path_str)
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(path_str)
            except OSError:
                pass
    if prefix:
        os.environ['PATH'] = os.pathsep.join(prefix) + os.pathsep + os.environ.get('PATH', '')


def setup_cuda_dlls() -> None:
    if os.name != 'nt':
        return

    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

    venv_bins: list[Path] = []
    for sp in site.getsitepackages():
        for sub in ('nvidia/cudnn/bin', 'nvidia/cublas/bin'):
            directory = Path(sp) / sub
            if directory.is_dir():
                venv_bins.append(directory)

    legacy = Path(r'C:\Tools\cuda11-dlls')
    ordered = list(venv_bins)
    if legacy.is_dir() and settings.whisper.legacy_dlls:
        ordered.append(legacy)

    _prepend_path(*ordered)


def setup_espeak() -> None:
    if os.name != 'nt':
        return

    candidates = [
        Path(os.environ.get('ProgramFiles', r'C:\Program Files')) / 'eSpeak NG',
        Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')) / 'eSpeak NG',
    ]
    _prepend_path(*candidates)


def ensure_espeak() -> None:
    """Verify espeak-ng is callable; on Windows add default install dir first."""
    setup_espeak()
    try:
        subprocess.run(
            ['espeak-ng', '--version'],
            capture_output=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        message = (
            "espeak-ng not found on PATH. Install it before starting the agent:\n"
            "  Windows: winget install eSpeak-NG.eSpeak-NG\n"
            "  macOS:   brew install espeak-ng\n"
            "  Linux:   sudo apt install espeak-ng\n"
            "Or download a release: https://github.com/espeak-ng/espeak-ng/releases"
        )
        logger.error(message)
        raise RuntimeError(message) from exc


def setup_windows_env() -> None:
    setup_cuda_dlls()
    setup_espeak()


setup_windows_env()
