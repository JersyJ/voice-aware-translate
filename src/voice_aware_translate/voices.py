"""Each speaker's voice: female, male or child, decided over many of their turns."""

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from voice_aware_translate.diarization import Turn


class Voice(StrEnum):
    FEMALE = "female"
    MALE = "male"
    CHILD = "child"
    UNKNOWN = "unknown"


# Order of the classifier's probabilities.
LABELS = (Voice.FEMALE, Voice.MALE, Voice.CHILD)

# Gives female, male and child probabilities for a mono segment.
Classifier = Callable[[np.ndarray], Sequence[float]]


@dataclass(frozen=True)
class SpeakerVoice:
    speaker: str
    voice: Voice
    probabilities: tuple[float, float, float]  # female, male, child
    seconds: float  # all speech of the speaker


def decide(probabilities: Sequence[float], threshold: float = 0.6) -> Voice:
    """The most likely voice, or UNKNOWN if it isn't likely enough."""
    best = max(range(len(LABELS)), key=lambda i: probabilities[i])
    return LABELS[best] if probabilities[best] >= threshold else Voice.UNKNOWN


def sample_turns(turns: Iterable[Turn], count: int, min_seconds: float) -> list[Turn]:
    """The speaker's longest turns: short ones are mostly noise. At least one turn if there is any."""
    longest = sorted(turns, key=lambda turn: turn.duration, reverse=True)
    long_enough = [turn for turn in longest[:count] if turn.duration >= min_seconds]
    return long_enough or longest[:1]


def speaker_voices(
    audio: np.ndarray,
    sample_rate: int,
    turns: Iterable[Turn],
    classify: Classifier,
    turns_per_speaker: int = 15,
    min_seconds: float = 1.0,
    max_seconds: float = 8.0,
) -> dict[str, SpeakerVoice]:
    """The voice of every speaker, from the duration-weighted average over their longest turns.

    One shout or a line over loud music can't flip a speaker this way.
    """
    by_speaker: dict[str, list[Turn]] = defaultdict(list)
    for turn in turns:
        by_speaker[turn.speaker].append(turn)

    result = {}
    for speaker, own_turns in by_speaker.items():
        total = np.zeros(len(LABELS))
        weight = 0.0
        for turn in sample_turns(own_turns, turns_per_speaker, min_seconds):
            seconds = min(turn.duration, max_seconds)
            start = int(turn.start * sample_rate)
            segment = audio[start : start + int(seconds * sample_rate)]
            if len(segment) < sample_rate // 10:  # under 0.1 s: nothing to hear
                continue
            total += seconds * np.asarray(classify(segment), dtype=float)
            weight += seconds
        probabilities = (total / weight if weight else total).tolist()
        female, male, child = probabilities
        result[speaker] = SpeakerVoice(
            speaker=speaker,
            voice=decide(probabilities) if weight else Voice.UNKNOWN,
            probabilities=(female, male, child),
            seconds=sum(turn.duration for turn in own_turns),
        )
    return result
