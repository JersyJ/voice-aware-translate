# voice-aware-translate

Translates English subtitles to Czech so that the grammar fits **who speaks and to whom**.

Czech marks gender where English doesn't:

| English | said by a man | said by a woman |
|---|---|---|
| I was tired. | Byl jsem unavený. | Byla jsem unavená. |
| Did you see it? (to a man / to a woman) | Viděl jsi to? | Viděla jsi to? |

A text-only translator has to guess, mostly the masculine form. `voice-aware-translate` listens to the
episode instead: it finds the speakers in the audio, decides each speaker's voice (woman, man or
child) and gives DeepL that as context for every line.

## How it works

1. **Audio**: ffmpeg decodes the episode's audio track (mono, 16 kHz).
2. **Speakers**: [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   splits the audio into turns of "speaker A", "speaker B", ... (no limit on their number).
3. **Voices**: [audEERING's wav2vec2 age-gender model](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender)
   gives female / male / child probabilities. They are averaged over each speaker's longest turns,
   so one shout or a line over loud music doesn't flip a speaker.
4. **Lines**: each subtitle line goes to the speaker who talks most during it. Cues with two people
   ("- Thanks. / - Oh, shit.") are split into their parts. The addressee is the other speaker of the
   same conversation (the nearest one before, else after, without a long pause).
5. **Translation**: DeepL translates each part with a context like *"The line is said by a woman to
   a man"* plus the neighboring lines with their speakers. DeepL doesn't translate or bill the
   context. Dashes, italics and line breaks are kept.

The analysis is cached next to the video (`VIDEO.voices.json`), so translating again doesn't
analyze again.

## Setup

Needs ffmpeg and [uv](https://docs.astral.sh/uv/). An NVIDIA GPU makes the analysis take minutes
instead of about the length of the episode.

```bash
uv sync
```

The speaker model is free but gated: create a [Hugging Face](https://huggingface.co) account,
accept the conditions on the [model page](https://huggingface.co/pyannote/speaker-diarization-community-1),
create a read token (Settings → Access Tokens) and log in once:

```bash
uvx hf auth login
```

The models (about 1.5 GB) are downloaded on first use. The audio stays on your computer; pyannote's
anonymous usage statistics are turned off (`PYANNOTE_METRICS_ENABLED=0`).

The DeepL key is read from `DEEPL_API_KEY`, or asked for.

## Usage

```bash
# VIDEO.en.srt → VIDEO.cs.srt
uv run voice-aware-translate "Show - S01E01.mkv"

# See the speakers and the voice of every line, without translating
uv run voice-aware-translate "Show - S01E01.mkv" --analyze-only

# For comparison: the same translation without the audio analysis
uv run voice-aware-translate "Show - S01E01.mkv" --plain --output plain.cs.srt
```

Options: `--subtitles`, `--output`, `--force`, `--audio-track N` (a release with several
languages), `--device cuda|cpu`, `--context N` (neighboring lines for DeepL), `--reanalyze`.

## Limits

- The voice says a speaker's gender, not the addressee's; that is a guess from the conversation.
- Music under the dialogue, whispering, shouting, phone voices and two people at once make the
  speaker detection less certain. Such lines get no gender hint (`?` in `--analyze-only`).
- A child's voice doesn't tell a boy from a girl.
- DeepL takes the context as a hint; it doesn't always follow it.

## Development

```bash
uv run pytest
uv run ruff check
uv run ty check
```

## Licenses

The code is yours to use. The voice model is CC-BY-NC-SA 4.0 (non-commercial only), the speaker
model CC-BY-4.0.
