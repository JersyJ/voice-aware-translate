from voice_aware_translate.attribution import Line, Timeline, add_addressees, attribute
from voice_aware_translate.diarization import Turn
from voice_aware_translate.subtitles import Cue
from voice_aware_translate.voices import Voice

VOICES = {"A": Voice.FEMALE, "B": Voice.MALE, "C": Voice.CHILD}


def test_main_speaker_is_who_talks_most_in_the_range():
    timeline = Timeline([Turn(0, 2, "A"), Turn(2, 5, "B"), Turn(8, 9, "A")])
    assert timeline.main_speaker(1.5, 4) == "B"
    assert timeline.main_speaker(0, 1) == "A"
    assert timeline.main_speaker(5.5, 7.5) is None


def test_attribute_splits_a_two_speaker_cue_between_the_speakers():
    turns = [Turn(0, 1.5, "A"), Turn(1.5, 4, "B")]
    lines = attribute([Cue(0, 4, "- Thanks.\n- Oh, shit, man.")], turns, VOICES)
    assert [(line.speaker, line.voice, line.addressee) for line in lines] == [
        ("A", Voice.FEMALE, Voice.MALE),
        ("B", Voice.MALE, Voice.FEMALE),
    ]


def test_addressee_skips_the_speakers_own_lines_but_not_long_pauses():
    lines = [
        Line(0, "Where were you?", 0, 2, "A", Voice.FEMALE),
        Line(1, "I was waiting.", 2.5, 4, "B", Voice.MALE),
        Line(2, "For hours.", 4.5, 6, "B", Voice.MALE),
        Line(3, "(much later, alone)", 30, 32, "B", Voice.MALE),
        Line(4, "(unknown speaker)", 40, 41),
    ]
    assert [line.addressee for line in add_addressees(lines, VOICES)] == [
        Voice.MALE,
        Voice.FEMALE,
        Voice.FEMALE,  # B's second line: back over B's own line to A
        Voice.UNKNOWN,  # more than CONVERSATION_GAP after the others
        Voice.UNKNOWN,
    ]
