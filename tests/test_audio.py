import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from voice_aware_translate.audio import SAMPLE_RATE, AudioError, load_audio

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def test_load_audio_decodes_the_track_to_mono_16khz(tmp_path: Path):
    video = tmp_path / "clip.mkv"
    # 2 s of 48 kHz stereo: a 440 Hz tone, the format of a real release's audio track
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2", "-ac", "2", str(video))

    audio = load_audio(video)

    assert audio.dtype == np.float32
    assert len(audio) == pytest.approx(2 * SAMPLE_RATE, abs=SAMPLE_RATE // 50)
    assert 0.1 < np.abs(audio).max() <= 1.0  # a real signal, not silence


def test_load_audio_reports_a_video_without_sound(tmp_path: Path):
    video = tmp_path / "silent.mkv"
    _ffmpeg("-f", "lavfi", "-i", "color=black:size=64x64:duration=1", str(video))

    with pytest.raises(AudioError, match=r"silent\.mkv"):
        load_audio(video)
