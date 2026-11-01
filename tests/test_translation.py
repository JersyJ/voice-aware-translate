from collections.abc import Sequence

import deepl
import pytest

from subtitle_voices import translation
from subtitle_voices.attribution import Line
from subtitle_voices.subtitles import Cue
from subtitle_voices.translation import DeepL, batches, build_context, translate_cues, voice_hint
from subtitle_voices.voices import Voice

LINES = [
    Line(0, "Where were you?", 0, 2, "A", Voice.FEMALE, Voice.MALE),
    Line(1, "- I was tired.", 3, 4, "B", Voice.MALE, Voice.FEMALE),
    Line(1, "- <i>Liar.</i>", 4, 5, "A", Voice.FEMALE, Voice.MALE),
    Line(2, "You should\ntrademark your face", 6, 8),
]
CUES = [
    Cue(0, 2, "Where were you?"),
    Cue(3, 5, "- I was tired.\n- <i>Liar.</i>"),
    Cue(6, 8, "You should\ntrademark your face"),
]
CZECH = {
    "Where were you?": "Kde jsi byl?",
    "I was tired.": "Byl jsem unavený.",
    "<i>Liar.</i>": "<i>Lhář.</i>",
    "You should trademark your face": "Měl by sis nechat zaregistrovat svou tvář",
}


def test_voice_hint_names_what_is_known():
    assert voice_hint(LINES[0]) == "The lines are said by a woman to a man."
    assert voice_hint(Line(0, "x", 0, 1, voice=Voice.CHILD)) == "The lines are said by a child."
    assert voice_hint(Line(0, "x", 0, 1, addressee=Voice.MALE)) == "The lines are said to a man."
    assert voice_hint(LINES[3]) == ""


def test_batches_group_lines_with_the_same_voices():
    same = Line(0, "x", 0, 1, "A", Voice.FEMALE, Voice.MALE)
    lines = [same, same, same, LINES[1], same]
    assert batches(lines) == [range(0, 3), range(3, 4), range(4, 5)]
    assert batches(lines[:3], size=2) == [range(0, 2), range(2, 3)]
    assert batches([]) == []


def test_build_context_has_the_hint_and_the_conversation_without_tags_and_dashes():
    assert build_context(LINES, range(1, 2), window=1) == (
        "The lines are said by a man to a woman.\n"
        "A woman: Where were you?\nA man: I was tired.\nA woman: Liar."
    )


def test_translate_cues_translates_each_part_with_its_context():
    calls: list[tuple[list[str], str]] = []

    def translate(texts: Sequence[str], context: str) -> list[str]:
        calls.append((list(texts), context))
        return [CZECH[text] for text in texts]

    czech = translate_cues(CUES, LINES, translate, window=0)
    assert [cue.text for cue in czech] == [
        "Kde jsi byl?",
        "- Byl jsem unavený.\n- <i>Lhář.</i>",
        "Měl by sis nechat\nzaregistrovat svou tvář",
    ]
    assert [cue.start for cue in czech] == [0, 3, 6]
    assert calls[1] == (["I was tired."], "The lines are said by a man to a woman.\nA man: I was tired.")


class _Result:
    def __init__(self, text: str):
        self.text = text


class _BusyClient:
    """Answers "too many requests" `busy` times, then translates."""

    def __init__(self, busy: int):
        self.busy = busy
        self.calls: list[dict[str, object]] = []

    def translate_text(self, texts: list[str], **options: object) -> list[_Result]:
        self.calls.append(options)
        if len(self.calls) <= self.busy:
            raise deepl.TooManyRequestsException("busy")
        return [_Result(CZECH[text]) for text in texts]


def _deepl_with(client: _BusyClient, retries: int) -> DeepL:
    translator = DeepL.__new__(DeepL)
    translator.__dict__.update(_client=client, _target_lang="CS", _retries=retries)
    return translator


def test_deepl_waits_and_retries_when_busy(monkeypatch: pytest.MonkeyPatch):
    waits: list[float] = []
    monkeypatch.setattr(translation.time, "sleep", waits.append)
    client = _BusyClient(busy=2)

    assert _deepl_with(client, retries=3)(["<i>Liar.</i>"], "ctx") == ["<i>Lhář.</i>"]
    assert waits == [1, 2]
    assert client.calls[-1]["context"] == "ctx"
    assert client.calls[-1]["tag_handling"] == "html"


def test_deepl_gives_up_after_the_retries(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(translation.time, "sleep", lambda _: None)
    with pytest.raises(deepl.TooManyRequestsException):
        _deepl_with(_BusyClient(busy=5), retries=2)(["Where were you?"], "")
