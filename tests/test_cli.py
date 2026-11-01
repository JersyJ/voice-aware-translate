from collections.abc import Sequence
from pathlib import Path

import pytest

from voice_aware_translate import cli
from voice_aware_translate.subtitles import parse_srt


class FakeDeepL:
    def __init__(self, auth_key: str):
        assert auth_key == "test-key"

    def __call__(self, texts: Sequence[str], context: str) -> list[str]:
        return [f"CS[{text}]" for text in texts]


def test_plain_run_translates_the_episode_subtitles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    video = tmp_path / "Show - S01E01.mkv"
    video.touch()
    (tmp_path / "Show - S01E01.en.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\n- Hi.\n- Hello.\n")
    monkeypatch.setattr(cli, "DeepL", FakeDeepL)
    monkeypatch.setenv("DEEPL_API_KEY", "test-key")

    assert cli.main([str(video), "--plain"]) == 0

    czech = parse_srt((tmp_path / "Show - S01E01.cs.srt").read_text())
    assert [cue.text for cue in czech] == ["- CS[Hi.]\n- CS[Hello.]"]


def test_refuses_to_overwrite_without_force(tmp_path: Path):
    video = tmp_path / "e.mkv"
    (tmp_path / "e.cs.srt").write_text("old")
    assert cli.main([str(video), "--plain"]) == 1
    assert (tmp_path / "e.cs.srt").read_text() == "old"
