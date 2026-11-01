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

1. **Audio**: ffmpeg decodes the audio track of the episode (mono, 16 kHz).
2. **Speakers**: [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   divides the audio into turns of "speaker A", "speaker B" and more. The number of speakers has no
   limit.
3. **Voices**: The [audEERING wav2vec2 age-gender model](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender)
   gives the probabilities for female, male and child. The tool calculates the average over the
   longest turns of each speaker. Thus, one shout or one line with loud music does not change the
   voice of a speaker.
4. **Lines**: The tool gives each subtitle line to the speaker who speaks most during that line. If a
   cue has two persons ("- Thanks. / - Oh, shit."), the tool divides the cue into two parts. The
   addressee is the other speaker in the same conversation. The tool looks for the nearest speaker
   before the line, then after the line. A long pause stops the conversation.
5. **Translation**: DeepL translates each part with a context, for example *"The lines are said by a
   woman to a man"*. The context also contains the adjacent lines and their speakers. DeepL does not
   translate the context and does not count the context for payment. The tool keeps the dashes, the
   italic text and the line breaks.

The tool keeps the analysis in a file next to the video (`VIDEO.voices.json`). If you translate the
episode again, the tool uses this file and does not do the analysis again.

## Installation

Make sure that ffmpeg and [uv](https://docs.astral.sh/uv/) are on the computer. With an NVIDIA GPU,
the analysis takes some minutes. Without a GPU, the analysis takes approximately the length of the
episode.

Install the dependencies:

```bash
uv sync
```

The speaker model is free, but the model has restricted access. Do these steps one time:

1. Create a [Hugging Face](https://huggingface.co) account.
2. On the [model page](https://huggingface.co/pyannote/speaker-diarization-community-1), accept the
   conditions.
3. Create a token with read access (Settings → Access Tokens).
4. Log in with the token:

   ```bash
   uvx hf auth login
   ```

The tool downloads the models (approximately 1.5 GB) when you use the tool for the first time. The
audio stays on your computer. The tool disables the anonymous usage statistics of pyannote
(`PYANNOTE_METRICS_ENABLED=0`).

For the translation, the tool needs a DeepL API key. Do these steps one time:

1. Create an account for the DeepL API on [deepl.com/pro-api](https://www.deepl.com/pro-api).
   The Free plan translates 500,000 characters each month, approximately 25 episodes.
2. Copy the authentication key from the account page.
3. In each new shell, set the key. With `read -rs`, the key does not go into the shell history:

   ```bash
   read -rs DEEPL_API_KEY && export DEEPL_API_KEY
   ```

If you do not set `DEEPL_API_KEY`, the tool asks for the key.

## Docker

The Docker image contains Python, ffmpeg and all dependencies. You need only Docker. For the GPU, also
install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).

Build the image (approximately 11 GB on the disk, mostly PyTorch with CUDA):

```bash
docker build -t voice-aware-translate .
```

Do steps 1 to 3 for Hugging Face and the steps for the DeepL key from the installation. Then set the
Hugging Face token in the shell:

```bash
read -rs HF_TOKEN && export HF_TOKEN
```

Translate an episode in the current directory:

```bash
mkdir -p ~/.cache/huggingface
docker run --rm -it --gpus all --user "$(id -u):$(id -g)" \
  -v ~/.cache/huggingface:/cache -v "$PWD:/data" \
  -e HF_TOKEN -e DEEPL_API_KEY \
  voice-aware-translate "Show - S01E01.mkv"
```

- `-v ~/.cache/huggingface:/cache` keeps the models between runs. If you logged in with
  `hf auth login`, the container uses that login, and `-e HF_TOKEN` is not necessary.
- `-e HF_TOKEN -e DEEPL_API_KEY` give the keys from the shell to the container. The values do not
  go into the command line.
- `--user` makes the subtitle files yours. The video and the subtitles must be in the current
  directory.
- Without a GPU, remove `--gpus all`. Then the analysis takes approximately the length of the episode.

## Use

```bash
# Translate VIDEO.en.srt to VIDEO.cs.srt
uv run voice-aware-translate "Show - S01E01.mkv"

# Show the speakers and the voice of each line. Do not translate.
uv run voice-aware-translate "Show - S01E01.mkv" --analyze-only

# Translate without the audio analysis, for a comparison
uv run voice-aware-translate "Show - S01E01.mkv" --plain --output plain.cs.srt
```

| Option | Function |
|---|---|
| `--subtitles FILE` | English subtitle file (default: `VIDEO.en.srt`) |
| `--output FILE` | Czech subtitle file (default: `VIDEO.cs.srt`) |
| `--force` | Overwrite the output file |
| `--audio-track N` | Use audio track N, for a release with more than one language |
| `--device cuda\|cpu` | Select the processor for the analysis |
| `--context N` | Number of adjacent lines that DeepL gets as context (default: 2) |
| `--reanalyze` | Do the analysis again and ignore the file from the last analysis |

## Limits

- The voice identifies the gender of the speaker. The voice does not identify the gender of the
  addressee. The tool guesses the addressee from the conversation.
- These conditions decrease the accuracy of the speaker detection:
  - music below the dialogue
  - whispers and shouts
  - telephone voices
  - two persons who speak at the same time

  Lines with low accuracy get no gender hint (`?` in `--analyze-only`).
- The voice of a child does not identify a boy or a girl.
- DeepL uses the context as a hint. DeepL does not always obey the hint.

## Development

[prek](https://github.com/j178/prek) runs all checks: ruff, ty, pytest and a scan for secrets. Install
the Git hook one time. Then the checks run before each commit:

```bash
prek install
```

Run the checks on all files:

```bash
prek run --all-files
```
