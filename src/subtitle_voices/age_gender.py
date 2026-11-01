"""The voice classifier: audEERING's wav2vec2 age and gender model.

https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender (CC-BY-NC-SA 4.0: for
non-commercial use). The model classes follow its model card; the checkpoint's weight names must
match them (`wav2vec2`, `age`, `gender`, `dense`, `out_proj`).
"""

import numpy as np
import torch
from torch import nn
from transformers import Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

MODEL = "audeering/wav2vec2-large-robust-24-ft-age-gender"


class _Head(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, labels: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, labels)

    # torch declares `forward` as an attribute; defining it as a method is the normal PyTorch way.
    def forward(self, features: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        return self.out_proj(self.dropout(x))


class _AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = _Head(config.hidden_size, config.final_dropout, 1)
        self.gender = _Head(config.hidden_size, config.final_dropout, 3)  # female, male, child
        self.post_init()

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        hidden = self.wav2vec2(input_values)[0].mean(dim=1)
        return torch.softmax(self.gender(hidden), dim=1)


class AgeGenderClassifier:
    """Female, male and child probabilities of a mono 16 kHz segment. The model is 1.3 GB."""

    def __init__(self, device: str):
        self._device = device
        # Only the feature extractor: it normalizes the signal (the model has no tokenizer to use).
        self._features = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
        model = _AgeGenderModel.from_pretrained(MODEL)
        nn.Module.to(model, torch.device(device))  # transformers' own .to() type hint is broken
        self._model = model.eval()

    def __call__(self, segment: np.ndarray) -> list[float]:
        values = self._features(segment, sampling_rate=16_000)["input_values"][0]
        tensor = torch.from_numpy(np.asarray(values, dtype=np.float32).reshape(1, -1)).to(self._device)
        with torch.inference_mode():
            return self._model(tensor)[0].tolist()
