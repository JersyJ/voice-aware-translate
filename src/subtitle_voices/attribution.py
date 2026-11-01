"""Matching subtitle lines to speakers, and guessing whom each line is said to."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from subtitle_voices.diarization import Turn
from subtitle_voices.subtitles import Cue, dialogue_parts
from subtitle_voices.voices import Voice

# A neighboring line counts as the same conversation only if the pause is at most this long (s).
CONVERSATION_GAP = 5.0


@dataclass(frozen=True)
class Line:
    """One speaker's part of a cue, with who says it and to whom."""

    cue: int  # index of the cue it belongs to
    text: str
    start: float
    end: float
    speaker: str | None = None
    voice: Voice = Voice.UNKNOWN
    addressee: Voice = Voice.UNKNOWN


class Timeline:
    """Speaker turns, to look up who talks in a time range. The turns must not overlap."""

    def __init__(self, turns: Iterable[Turn]):
        self._turns = sorted(turns)
        self._starts = [turn.start for turn in self._turns]
        self._ends = [turn.end for turn in self._turns]  # sorted too, as turns don't overlap

    def main_speaker(self, start: float, end: float) -> str | None:
        """The speaker who talks most between `start` and `end`, if anybody talks."""
        first = bisect_right(self._ends, start)  # first turn that ends after `start`
        last = bisect_left(self._starts, end)  # first turn that starts at or after `end`
        talk: dict[str, float] = defaultdict(float)
        for turn in self._turns[first:last]:
            talk[turn.speaker] += min(end, turn.end) - max(start, turn.start)
        return max(talk, key=talk.__getitem__) if talk else None


def split_cue(index: int, cue: Cue) -> list[Line]:
    """The speakers' parts of a cue, each with a share of the cue's time by its text length."""
    parts = dialogue_parts(cue.text)
    total = sum(len(part) for part in parts)
    lines, start = [], cue.start
    for part in parts:
        end = start + (cue.end - cue.start) * len(part) / total
        lines.append(Line(index, part, start, end))
        start = end
    return lines


def attribute(cues: Sequence[Cue], turns: Iterable[Turn], voices: Mapping[str, Voice]) -> list[Line]:
    """All speakers' parts of the cues, with speaker, voice and addressee where they can be told."""
    timeline = Timeline(turns)
    lines = []
    for index, cue in enumerate(cues):
        for line in split_cue(index, cue):
            speaker = timeline.main_speaker(line.start, line.end)
            voice = voices.get(speaker, Voice.UNKNOWN) if speaker else Voice.UNKNOWN
            lines.append(replace(line, speaker=speaker, voice=voice))
    return add_addressees(lines, voices)


def add_addressees(lines: Sequence[Line], voices: Mapping[str, Voice]) -> list[Line]:
    """Whom each line is said to: the nearest other speaker before it in the same conversation,
    else the nearest one after it. Unknown in a monologue or after a long pause."""
    result = []
    for i, line in enumerate(lines):
        other = _other_speaker(lines, i, -1) or _other_speaker(lines, i, +1)
        addressee = voices.get(other, Voice.UNKNOWN) if other else Voice.UNKNOWN
        result.append(replace(line, addressee=addressee))
    return result


def _other_speaker(lines: Sequence[Line], index: int, step: int) -> str | None:
    """Walk from lines[index] in `step` direction over the same speaker's lines (and lines without a
    speaker) to the first line of somebody else, as long as no pause exceeds CONVERSATION_GAP."""
    current = lines[index]
    if current.speaker is None:
        return None
    edge, j = current, index + step
    while 0 <= j < len(lines):
        neighbor = lines[j]
        gap = edge.start - neighbor.end if step < 0 else neighbor.start - edge.end
        if gap > CONVERSATION_GAP:
            return None
        if neighbor.speaker is not None and neighbor.speaker != current.speaker:
            return neighbor.speaker
        edge, j = neighbor, j + step
    return None
