"""Command line: translate an episode's English subtitles to Czech with the speakers' voices."""

import argparse
import os
import sys
from collections import Counter
from collections.abc import Sequence
from getpass import getpass
from pathlib import Path

import deepl

from voice_aware_translate import analysis as cache
from voice_aware_translate.analysis import Analysis, analyze
from voice_aware_translate.attribution import Line, attribute
from voice_aware_translate.audio import AudioError
from voice_aware_translate.diarization import DiarizationError
from voice_aware_translate.subtitles import format_srt, parse_srt
from voice_aware_translate.translation import DeepL, translate_cues
from voice_aware_translate.voices import Voice

_SHORT = {Voice.FEMALE: "woman", Voice.MALE: "man", Voice.CHILD: "child", Voice.UNKNOWN: "?"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-aware-translate",
        description="Translate English subtitles to Czech with the gender of who speaks and who is "
        "addressed, detected from the audio.",
    )
    parser.add_argument("video", type=Path, help="The episode or movie (its audio is analyzed)")
    parser.add_argument("--subtitles", type=Path, help="English SRT (default: VIDEO.en.srt)")
    parser.add_argument("--output", type=Path, help="Czech SRT to write (default: VIDEO.cs.srt)")
    parser.add_argument("--force", action="store_true", help="Overwrite the output")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Without the audio analysis, for comparison (DeepL with the neighboring lines)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only show the speakers and the voice of each line, translate nothing",
    )
    parser.add_argument("--reanalyze", action="store_true", help="Ignore the cached analysis")
    parser.add_argument("--audio-track", type=int, default=0, help="Audio track number (default: 0)")
    parser.add_argument("--device", default="auto", help="cuda, cpu or auto (default)")
    parser.add_argument(
        "--context",
        type=int,
        default=2,
        help="Lines before and after each line given to DeepL as context (default: 2)",
    )
    return parser


def _sibling(video: Path, suffix: str) -> Path:
    return video.with_name(video.stem + suffix)


def _device(choice: str) -> str:
    if choice != "auto":
        return choice
    import torch  # slow to import, only for the analysis

    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_analysis(args: argparse.Namespace) -> Analysis:
    path = _sibling(args.video, ".voices.json")
    if not args.reanalyze and (cached := cache.load(path)):
        print(f"🗂️  Using the cached analysis {path.name}")
        return cached
    result = analyze(args.video, _device(args.device), args.audio_track)
    cache.save(result, path)
    return result


def _print_speakers(result: Analysis, lines: Sequence[Line]) -> None:
    line_counts = Counter(line.speaker for line in lines)
    print("\nSpeaker      voice   female  male  child   speech  lines")
    for info in sorted(result.voices.values(), key=lambda info: info.seconds, reverse=True):
        female, male, child = (f"{p:5.0%}" for p in info.probabilities)
        minutes, seconds = divmod(round(info.seconds), 60)
        print(
            f"{info.speaker:<12} {_SHORT[info.voice]:<6} {female}  {male} {child}  "
            f"{minutes:3d}:{seconds:02d}  {line_counts[info.speaker]:5d}"
        )
    known = sum(line.voice is not Voice.UNKNOWN for line in lines)
    print(f"\nVoice known for {known} of {len(lines)} lines.")


def _print_lines(lines: Sequence[Line]) -> None:
    for line in lines:
        minutes, seconds = divmod(int(line.start), 60)
        who = f"{_SHORT[line.voice]}→{_SHORT[line.addressee]}"
        print(f"{minutes:3d}:{seconds:02d}  {who:<12} {' '.join(line.text.split())}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    subtitles = args.subtitles or _sibling(args.video, ".en.srt")
    output = args.output or _sibling(args.video, ".cs.srt")
    if output.exists() and not (args.force or args.analyze_only):
        print(f"❌ {output} exists; --force overwrites it.", file=sys.stderr)
        return 1

    try:
        cues = parse_srt(subtitles.read_text(encoding="utf-8-sig"))
        if args.plain:
            lines = attribute(cues, [], {})
        else:
            result = _get_analysis(args)
            lines = attribute(cues, result.turns, result.voice_map())
            _print_speakers(result, lines)
        if args.analyze_only:
            _print_lines(lines)
            return 0

        key = os.environ.get("DEEPL_API_KEY") or getpass("DeepL API key: ")
        print(f"🌍 Translating {len(cues)} cues with DeepL...")
        czech = translate_cues(
            cues,
            lines,
            DeepL(key),
            window=args.context,
            on_progress=lambda done, total: print(f"\r   {done}/{total}", end=""),
        )
        output.write_text(format_srt(czech), encoding="utf-8")
        print(f"\n💾 Saved {output}")
    except (OSError, AudioError, DiarizationError, deepl.DeepLException) as error:
        print(f"\n❌ {error}", file=sys.stderr)
        return 1
    return 0
