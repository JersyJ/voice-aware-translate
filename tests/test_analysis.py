from pathlib import Path

from subtitle_voices import analysis
from subtitle_voices.analysis import Analysis
from subtitle_voices.diarization import Turn
from subtitle_voices.voices import SpeakerVoice, Voice


def test_cache_round_trips(tmp_path: Path):
    result = Analysis(
        turns=[Turn(0.0, 1.5, "SPEAKER_00"), Turn(2.0, 3.0, "SPEAKER_01")],
        voices={"SPEAKER_00": SpeakerVoice("SPEAKER_00", Voice.FEMALE, (0.9, 0.08, 0.02), 1.5)},
    )
    path = tmp_path / "e01.voices.json"
    analysis.save(result, path)
    loaded = analysis.load(path)
    assert loaded == result
    assert loaded is not None
    assert loaded.voice_map() == {"SPEAKER_00": Voice.FEMALE}


def test_cache_ignores_missing_and_other_formats(tmp_path: Path):
    assert analysis.load(tmp_path / "missing.json") is None
    (tmp_path / "old.json").write_text('{"format": 0}')
    assert analysis.load(tmp_path / "old.json") is None
