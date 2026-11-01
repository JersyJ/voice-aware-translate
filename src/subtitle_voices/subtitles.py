"""SRT subtitles: reading, writing, and the speakers' parts inside a cue."""

import re
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass

_TIMING = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})")
# A line that starts a speaker's part: "- Thanks.", also behind formatting tags ("<i>- Thanks.</i>").
_DASH = re.compile(r"^(\s*(?:<[^>]+>\s*)*)-\s*")


@dataclass(frozen=True)
class Cue:
    """One subtitle, shown from `start` to `end` (seconds)."""

    start: float
    end: float
    text: str


def _seconds(hours: str, minutes: str, seconds: str, millis: str) -> float:
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def _timestamp(seconds: float) -> str:
    hours, rest = divmod(round(seconds * 1000), 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_srt(content: str) -> list[Cue]:
    """Parse SRT text. Blocks without a timing line or without text are skipped."""
    cues = []
    for block in re.split(r"\n\s*\n", content.lstrip("﻿").replace("\r\n", "\n")):
        lines = block.strip("\n").split("\n")
        for i, line in enumerate(lines):
            if timing := _TIMING.search(line):
                times = timing.groups()
                text = "\n".join(lines[i + 1 :]).strip()
                if text:
                    cues.append(Cue(_seconds(*times[:4]), _seconds(*times[4:]), text))
                break
    return cues


def format_srt(cues: Iterable[Cue]) -> str:
    """SRT text, cues numbered from 1."""
    return "".join(
        f"{number}\n{_timestamp(cue.start)} --> {_timestamp(cue.end)}\n{cue.text}\n\n"
        for number, cue in enumerate(cues, 1)
    )


def dialogue_parts(text: str) -> list[str]:
    """The speakers' parts of a cue text.

    Two people in one cue are written as lines that start with a dash ("- Thanks.\\n- Oh, shit.");
    the first part may also have no dash ("Hey.\\n- What?"). A line without a dash continues the
    part before it, so a cue without dashed lines is one part.
    """
    parts: list[str] = []
    for line in text.split("\n"):
        if parts and not _DASH.match(line):
            parts[-1] += "\n" + line
        else:
            parts.append(line)
    return parts


def split_dash(part: str) -> tuple[str, str]:
    """Split a part into its dash prefix ("- " or "") and the spoken text (formatting tags kept)."""
    if match := _DASH.match(part):
        return "- ", match.group(1) + part[match.end() :]
    return "", part


def wrap_like(text: str, line_count: int) -> str:
    """Break `text` into `line_count` lines of similar length (at spaces), like the original cue."""
    text = " ".join(text.split())
    if line_count <= 1:
        return text
    width = len(text) // line_count + 1
    while len(lines := textwrap.wrap(text, width, break_long_words=False)) > line_count:
        width += 1
    return "\n".join(lines)
