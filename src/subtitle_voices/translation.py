"""Translating the lines with DeepL, with the voices as context."""

import re
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import replace

import deepl

from subtitle_voices.attribution import Line
from subtitle_voices.subtitles import Cue, split_dash, wrap_like
from subtitle_voices.voices import Voice

# Translates the texts (first argument) with one context (second, not translated) and returns the
# translations in the same order. Each text is translated on its own.
Translate = Callable[[Sequence[str], str], list[str]]

# Lines per request. Fewer requests keep DeepL from refusing them as "too many".
MAX_BATCH = 8

_PERSON = {Voice.FEMALE: "a woman", Voice.MALE: "a man", Voice.CHILD: "a child"}
_TAG = re.compile(r"<[^>]+>")


def voice_hint(line: Line) -> str:
    """One sentence on who says the lines to whom, or "" if neither is known."""
    speaker, addressee = _PERSON.get(line.voice), _PERSON.get(line.addressee)
    if speaker and addressee:
        return f"The lines are said by {speaker} to {addressee}."
    if speaker:
        return f"The lines are said by {speaker}."
    if addressee:
        return f"The lines are said to {addressee}."
    return ""


def batches(lines: Sequence[Line], size: int = MAX_BATCH) -> list[range]:
    """Runs of consecutive lines with the same speaker and addressee voices, at most `size` long.
    The lines of a run share one request and one context, which fits all of them."""
    runs: list[range] = []
    start = 0
    for index in range(1, len(lines) + 1):
        if (
            index == len(lines)
            or index - start == size
            or (lines[index].voice, lines[index].addressee) != (lines[start].voice, lines[start].addressee)
        ):
            runs.append(range(start, index))
            start = index
    return runs


def build_context(lines: Sequence[Line], run: range, window: int) -> str:
    """The context for the lines in `run`: who says them to whom, then the conversation around
    them, each line with its speaker ("A woman: ..."). DeepL doesn't translate or bill the context."""
    conversation = []
    for line in lines[max(run.start - window, 0) : run.stop + window]:
        who = _PERSON.get(line.voice, "someone").capitalize()
        text = " ".join(_TAG.sub("", split_dash(line.text)[1]).split())
        conversation.append(f"{who}: {text}")
    return "\n".join(filter(None, [voice_hint(lines[run.start]), *conversation]))


def translate_cues(
    cues: Sequence[Cue],
    lines: Sequence[Line],
    translate: Translate,
    window: int = 2,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Cue]:
    """The cues translated, each speaker's part on its own with its context.

    Dashes stay at the start of the parts, and each part keeps its number of lines.
    """
    translated: list[str] = []
    for run in batches(lines):
        texts = [" ".join(split_dash(lines[index].text)[1].split()) for index in run]
        translated += translate(texts, build_context(lines, run, window))
        if on_progress:
            on_progress(run.stop, len(lines))

    parts: dict[int, list[str]] = defaultdict(list)
    for line, czech in zip(lines, translated, strict=True):
        dash, text = split_dash(line.text)
        parts[line.cue].append(dash + wrap_like(czech, text.count("\n") + 1))
    return [replace(cue, text="\n".join(parts[index])) for index, cue in enumerate(cues)]


class DeepL:
    """DeepL translation from English, usable as `Translate`.

    When DeepL answers "too many requests", it waits (1 s, 2 s, ... up to a minute) and tries again.
    """

    def __init__(self, auth_key: str, target_lang: str = "CS", retries: int = 8):
        self._client = deepl.Translator(auth_key)
        self._target_lang = target_lang
        self._retries = retries

    def __call__(self, texts: Sequence[str], context: str) -> list[str]:
        attempt = 0
        while True:
            try:
                results = self._client.translate_text(
                    list(texts),
                    source_lang="EN",
                    target_lang=self._target_lang,
                    context=context or None,
                    tag_handling="html" if any(_TAG.search(text) for text in texts) else None,
                    preserve_formatting=True,
                )
                return [result.text for result in (results if isinstance(results, list) else [results])]
            except deepl.TooManyRequestsException:
                if attempt == self._retries:
                    raise
                time.sleep(min(2**attempt, 60))
                attempt += 1
