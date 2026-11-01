from subtitle_voices.subtitles import Cue, dialogue_parts, format_srt, parse_srt, split_dash, wrap_like

SRT = """﻿1
00:00:01,000 --> 00:00:02,500
Hello.

2
00:01:02,250 --> 00:01:04,000
- Thanks.
- Oh, shit.

no number
00:02:00.000 --> 00:02:01,000
<i>Without an index.</i>

4
00:03:00,000 --> 00:03:01,000

"""


def test_parse_srt_reads_times_and_text_and_skips_empty_cues():
    assert parse_srt(SRT.replace("\n", "\r\n")) == [
        Cue(1.0, 2.5, "Hello."),
        Cue(62.25, 64.0, "- Thanks.\n- Oh, shit."),
        Cue(120.0, 121.0, "<i>Without an index.</i>"),
    ]


def test_format_srt_round_trips():
    cues = parse_srt(SRT)
    text = format_srt(cues)
    assert text.startswith("1\n00:00:01,000 --> 00:00:02,500\nHello.\n\n2\n00:01:02,250 --> 00:01:04,000\n")
    assert parse_srt(text) == cues


def test_dialogue_parts_splits_dashed_lines():
    assert dialogue_parts("- Thanks.\n- Oh, shit.") == ["- Thanks.", "- Oh, shit."]
    assert dialogue_parts("Hey.\n- What?") == ["Hey.", "- What?"]
    assert dialogue_parts("<i>- Run!</i>\n<i>- Where?</i>") == ["<i>- Run!</i>", "<i>- Where?</i>"]


def test_dialogue_parts_keeps_continued_lines_together():
    assert dialogue_parts("You should\ntrademark your face") == ["You should\ntrademark your face"]
    assert dialogue_parts("- I think\nwe should go.\n- No.") == ["- I think\nwe should go.", "- No."]


def test_split_dash_keeps_formatting_tags():
    assert split_dash("- Thanks.") == ("- ", "Thanks.")
    assert split_dash("<i>- Run!</i>") == ("- ", "<i>Run!</i>")
    assert split_dash("No dash - here.") == ("", "No dash - here.")


def test_wrap_like_balances_lines():
    text = "Měl by sis nechat zaregistrovat svou tvář"
    assert wrap_like(text, 2) == "Měl by sis nechat\nzaregistrovat svou tvář"
    assert wrap_like("Krátké   věta\n", 1) == "Krátké věta"
    assert wrap_like("Jedno", 2) == "Jedno"
