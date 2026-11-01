from pathlib import Path

from voice_aware_translate import analysis
from voice_aware_translate.analysis import Analysis
from voice_aware_translate.diarization import Turn
from voice_aware_translate.voices import SpeakerVoice, Voice


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
