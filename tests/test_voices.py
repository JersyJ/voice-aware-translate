import numpy as np
import pytest

from voice_aware_translate.diarization import Turn
from voice_aware_translate.voices import Voice, decide, sample_turns, speaker_voices

RATE = 100  # samples per second, enough for the logic


def test_decide_needs_a_clear_winner():
    assert decide([0.8, 0.15, 0.05]) is Voice.FEMALE
    assert decide([0.1, 0.2, 0.7]) is Voice.CHILD
    assert decide([0.45, 0.45, 0.1]) is Voice.UNKNOWN


def test_sample_turns_prefers_long_turns_but_keeps_one():
    turns = [Turn(0, 0.5, "A"), Turn(1, 4, "A"), Turn(5, 7, "A"), Turn(8, 8.2, "A")]
    assert sample_turns(turns, count=2, min_seconds=1) == [Turn(1, 4, "A"), Turn(5, 7, "A")]
    assert sample_turns(turns[:1], count=2, min_seconds=1) == [Turn(0, 0.5, "A")]


def test_speaker_voices_averages_over_turns_weighted_by_duration():
    # The fake classifier reads the "voice" from the sample values: 1 = female, 2 = male.
    audio = np.concatenate([np.full(10 * RATE, 1.0), np.full(10 * RATE, 2.0)]).astype(np.float32)
    turns = [Turn(0, 8, "A"), Turn(8, 9.5, "A"), Turn(10, 20, "B")]

    def classify(segment: np.ndarray) -> list[float]:
        return [0.9, 0.1, 0.0] if segment.mean() < 1.5 else [0.1, 0.9, 0.0]

    voices = speaker_voices(audio, RATE, turns, classify, max_seconds=5)
    assert voices["A"].voice is Voice.FEMALE
    assert voices["A"].seconds == pytest.approx(9.5)
    assert voices["B"].voice is Voice.MALE
    assert voices["B"].probabilities == pytest.approx((0.1, 0.9, 0.0))


def test_speaker_voices_outvotes_a_single_wrong_turn():
    audio = np.concatenate([np.full(12 * RATE, 1.0), np.full(2 * RATE, 2.0)]).astype(np.float32)
    turns = [Turn(0, 6, "A"), Turn(6, 12, "A"), Turn(12, 14, "A")]

    def classify(segment: np.ndarray) -> list[float]:
        return [0.95, 0.05, 0.0] if segment.mean() < 1.5 else [0.0, 1.0, 0.0]

    assert speaker_voices(audio, RATE, turns, classify)["A"].voice is Voice.FEMALE
