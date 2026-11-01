from pathlib import Path

from voice_aware_translate import cli


def test_refuses_to_overwrite_without_force(tmp_path: Path):
    video = tmp_path / "e.mkv"
    (tmp_path / "e.cs.srt").write_text("old")
    assert cli.main([str(video), "--plain"]) == 1
    assert (tmp_path / "e.cs.srt").read_text() == "old"
