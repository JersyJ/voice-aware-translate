from collections.abc import Sequence

from voice_aware_translate.attribution import Line
from voice_aware_translate.subtitles import Cue
from voice_aware_translate.translation import batches, build_context, translate_cues
from voice_aware_translate.voices import Voice

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
