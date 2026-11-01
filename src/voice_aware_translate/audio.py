"""Decoding the audio of a video with ffmpeg."""

import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000  # what both models expect


class AudioError(RuntimeError):
    """ffmpeg could not decode the audio."""


def load_audio(video: Path, track: int = 0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """The `track`-th audio track of `video` as mono float32 samples at `sample_rate`.

    A 45-minute episode at 16 kHz is about 170 MB in memory.
    """
    command = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(video),
        "-map", f"0:a:{track}", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1",
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        message = result.stderr.decode(errors="replace").strip() or "no audio"
        raise AudioError(f"{video}: {message}")
    return np.frombuffer(result.stdout, dtype=np.float32)
