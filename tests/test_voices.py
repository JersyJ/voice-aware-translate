import numpy as np
import pytest

from voice_aware_translate.diarization import Turn
from voice_aware_translate.voices import Voice, speaker_voices

RATE = 100  # samples per second, enough for the logic


def _audio(*parts: tuple[float, float]) -> np.ndarray:
    """Test audio from (seconds, level) parts; the fake classifier hears the level as the voice."""
    return np.concatenate([np.full(int(seconds * RATE), level) for seconds, level in parts]).astype(
        np.float32
    )


def _classify(segment: np.ndarray) -> list[float]:
    """Level 1 sounds female, level 2 male."""
    return [1.0, 0.0, 0.0] if segment.mean() < 1.5 else [0.0, 1.0, 0.0]


def test_speaker_voices_weights_turns_by_duration():
    # A 6 s turn that sounds female and a 2 s turn that sounds male: 75 % female by duration.
    # Counted per turn, it would be 50 %, and the voice unknown.
    result = speaker_voices(_audio((6, 1), (2, 2)), RATE, [Turn(0, 6, "A"), Turn(6, 8, "A")], _classify)
    assert result["A"].probabilities == pytest.approx((0.75, 0.25, 0.0))
    assert result["A"].voice is Voice.FEMALE
    assert result["A"].seconds == pytest.approx(8)


def test_speaker_voices_outvotes_a_single_wrong_turn():
    turns = [Turn(0, 6, "A"), Turn(6, 12, "A"), Turn(12, 14, "A")]
    assert speaker_voices(_audio((12, 1), (2, 2)), RATE, turns, _classify)["A"].voice is Voice.FEMALE
