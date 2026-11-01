"""Who speaks when: speaker diarization with pyannote."""

import os
from dataclasses import dataclass

import numpy as np

MODEL = "pyannote/speaker-diarization-community-1"


@dataclass(frozen=True, order=True)
class Turn:
    """A stretch of speech by one speaker (labels like "SPEAKER_03" are only per episode)."""

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


class DiarizationError(RuntimeError):
    """The diarization model could not be loaded."""


def diarize(audio: np.ndarray, sample_rate: int, device: str, token: str | None = None) -> list[Turn]:
    """Speaker turns in `audio`, without overlaps, sorted by time.

    The model is gated on Hugging Face: accept its conditions and log in (`hf auth login`), or pass
    `token`. Runs on the GPU in a few minutes per episode.
    """
    # pyannote sends anonymous usage statistics unless this is off. Set it to 1 to allow them.
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
    import torch  # heavy, only needed here
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput

    try:
        pipeline = Pipeline.from_pretrained(MODEL, token=token)
    except Exception as error:  # huggingface_hub raises several types for access problems
        raise DiarizationError(_access_help(error)) from error
    if pipeline is None:
        raise DiarizationError(_access_help(None))
    pipeline.to(torch.device(device))
    output = pipeline({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sample_rate})
    if not isinstance(output, DiarizeOutput):
        raise DiarizationError(f"{MODEL} returned {type(output).__name__}, expected DiarizeOutput")
    # "Exclusive": one speaker at a time, which is what a subtitle line needs.
    return sorted(
        Turn(segment.start, segment.end, str(speaker))
        for segment, speaker in output.exclusive_speaker_diarization
    )


def _access_help(error: Exception | None) -> str:
    detail = f" ({error})" if error else ""
    return (
        f"cannot load {MODEL}{detail}. Accept its conditions on https://huggingface.co/{MODEL} "
        "and log in with `uvx hf auth login` (or set HF_TOKEN)."
    )
