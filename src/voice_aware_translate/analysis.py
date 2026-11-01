"""The audio analysis of an episode (speaker turns and voices), and its JSON cache."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from voice_aware_translate.audio import SAMPLE_RATE, load_audio
from voice_aware_translate.diarization import Turn, diarize
from voice_aware_translate.voices import SpeakerVoice, Voice, speaker_voices

_FORMAT = 1  # version of the cache file


@dataclass(frozen=True)
class Analysis:
    turns: list[Turn]
    voices: dict[str, SpeakerVoice]

    def voice_map(self) -> dict[str, Voice]:
        return {speaker: info.voice for speaker, info in self.voices.items()}


def analyze(
    video: Path,
    device: str,
    audio_track: int = 0,
    token: str | None = None,
    log: Callable[[str], None] = print,
) -> Analysis:
    """Find the speakers in the video's audio and decide their voices. Downloads the models on
    first use (about 1.5 GB)."""
    from voice_aware_translate.age_gender import AgeGenderClassifier  # loads torch

    log(f"🎧 Decoding the audio of {video.name}...")
    audio = load_audio(video, audio_track)
    log(f"🗣️  Finding the speakers ({len(audio) / SAMPLE_RATE / 60:.0f} min of audio, {device})...")
    turns = diarize(audio, SAMPLE_RATE, device, token)
    log(f"🔊 Deciding the voices of {len({turn.speaker for turn in turns})} speakers...")
    voices = speaker_voices(audio, SAMPLE_RATE, turns, AgeGenderClassifier(device))
    return Analysis(turns, voices)


def save(analysis: Analysis, path: Path) -> None:
    data = {
        "format": _FORMAT,
        "turns": [asdict(turn) for turn in analysis.turns],
        "voices": [asdict(voice) for voice in analysis.voices.values()],
    }
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def load(path: Path) -> Analysis | None:
    """The cached analysis, or None if there is none in this format."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if data.get("format") != _FORMAT:
        return None
    turns = [Turn(**turn) for turn in data["turns"]]
    voices = {
        voice["speaker"]: SpeakerVoice(
            speaker=voice["speaker"],
            voice=Voice(voice["voice"]),
            probabilities=tuple(voice["probabilities"]),
            seconds=voice["seconds"],
        )
        for voice in data["voices"]
    }
    return Analysis(turns, voices)
