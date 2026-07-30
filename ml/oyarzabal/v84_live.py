"""Minimal V8.4 sequence-residual inference used by the live service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import validate_probability_matrix
from .sequence import TOKEN_NUMERIC_COLUMNS, SequenceExamples, SequenceNormalizer

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


if nn is not None:

    class GlobalConditionedSequenceResidual(nn.Module):
        """Inference-compatible copy of the trained V8.4 architecture."""

        def __init__(
            self,
            description_vocab_size: int,
            *,
            length: int,
            current_numeric_width: int,
        ):
            super().__init__()
            self.length = length
            self.group_embedding = nn.Embedding(8, 8, padding_idx=0)
            self.family_embedding = nn.Embedding(5, 4, padding_idx=0)
            self.ball_embedding = nn.Embedding(5, 3, padding_idx=0)
            self.strike_embedding = nn.Embedding(4, 3, padding_idx=0)
            self.description_embedding = nn.Embedding(
                max(2, description_vocab_size), 8, padding_idx=0
            )
            self.stand_embedding = nn.Embedding(4, 3, padding_idx=0)
            token_width = 8 + 4 + 3 + 3 + 8 + 3 + 2
            token_width += 2 * len(TOKEN_NUMERIC_COLUMNS)
            self.token_projection = nn.Linear(token_width, 64)
            current_width = 3 + 3 + 3 + 2 * current_numeric_width + 6
            self.current_projection = nn.Linear(current_width, 64)
            self.position_embedding = nn.Embedding(length, 64)
            layer = nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=128,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=2,
                enable_nested_tensor=False,
            )
            self.normalization = nn.LayerNorm(64)
            self.family_delta = nn.Linear(64, 3)
            self.child_delta = nn.Linear(64, 6)

        def causal_mask(self, device: torch.device) -> torch.Tensor:
            return torch.triu(
                torch.ones(
                    self.length,
                    self.length,
                    dtype=torch.bool,
                    device=device,
                ),
                diagonal=1,
            )

        def forward(self, batch: dict[str, torch.Tensor]) -> tuple[Any, Any]:
            categorical = batch["token_categorical"].long()
            tokens = self.token_projection(
                torch.cat(
                    [
                        self.group_embedding(categorical[..., 0]),
                        self.family_embedding(categorical[..., 1]),
                        self.ball_embedding(categorical[..., 2]),
                        self.strike_embedding(categorical[..., 3]),
                        self.description_embedding(categorical[..., 4]),
                        self.stand_embedding(categorical[..., 5]),
                        batch["history_flags"].float(),
                        batch["token_numeric"].float(),
                        batch["token_observed"].float(),
                    ],
                    dim=-1,
                )
            )
            positions = torch.arange(self.length, device=tokens.device)
            encoded = self.encoder(
                tokens + self.position_embedding(positions)[None, :, :],
                mask=self.causal_mask(tokens.device),
                src_key_padding_mask=batch["padding_mask"].bool(),
                is_causal=True,
            )
            current = batch["current_categorical"].long()
            global_values = batch["global_probabilities"].float().clamp_min(1e-12)
            current_values = torch.cat(
                [
                    self.ball_embedding(current[..., 0]),
                    self.strike_embedding(current[..., 1]),
                    self.stand_embedding(current[..., 2]),
                    batch["current_numeric"].float(),
                    batch["current_observed"].float(),
                    global_values.log(),
                ],
                dim=-1,
            )
            state = self.normalization(
                encoded[:, -1] + self.current_projection(current_values)
            )
            return self.family_delta(state), self.child_delta(state).reshape(-1, 3, 2)

else:

    class GlobalConditionedSequenceResidual:
        def __init__(self, *_args: object, **_kwargs: object):
            raise ModuleNotFoundError(
                "V8.4 live inference requires `uv sync --extra sequence`"
            )


@dataclass(frozen=True)
class LoadedExpert:
    model: GlobalConditionedSequenceResidual
    normalizer: SequenceNormalizer


def _temperature(
    probabilities: np.ndarray,
    log_temperature: float,
) -> np.ndarray:
    logits = np.log(validate_probability_matrix(probabilities))
    logits /= np.exp(log_temperature)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return validate_probability_matrix(values / values.sum(axis=1, keepdims=True))


def _hierarchical_calibration(
    probabilities: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    values = validate_probability_matrix(probabilities).reshape(-1, 3, 2)
    family = values.sum(axis=2)
    child = values / family[:, :, None]
    family_bias = parameters[:3] - parameters[:3].mean()
    child_bias = parameters[3:].reshape(3, 2)
    child_bias -= child_bias.mean(axis=1, keepdims=True)
    family_logits = np.log(np.clip(family, 1e-12, 1)) + family_bias
    family_logits -= family_logits.max(axis=1, keepdims=True)
    adjusted_family = np.exp(family_logits)
    adjusted_family /= adjusted_family.sum(axis=1, keepdims=True)
    child_logits = np.log(np.clip(child, 1e-12, 1)) + child_bias
    child_logits -= child_logits.max(axis=2, keepdims=True)
    adjusted_child = np.exp(child_logits)
    adjusted_child /= adjusted_child.sum(axis=2, keepdims=True)
    return validate_probability_matrix(
        (adjusted_family[:, :, None] * adjusted_child).reshape(-1, 6)
    )


def _transform(
    global_probabilities: np.ndarray,
    family_delta: np.ndarray,
    child_delta: np.ndarray,
    transform: dict[str, Any],
) -> np.ndarray:
    values = validate_probability_matrix(global_probabilities)
    family_values = values.reshape(-1, 3, 2).sum(axis=2)
    child_values = values.reshape(-1, 3, 2) / family_values[:, :, None]
    scale = float(transform["scale"])
    family_delta = family_delta * np.asarray(
        transform["familyShrinkage"],
        dtype=float,
    )
    child_delta = child_delta * np.asarray(
        transform["childShrinkage"],
        dtype=float,
    ).reshape(3, 2)
    family_logits = np.log(np.clip(family_values, 1e-12, 1))
    family_logits += scale * family_delta
    family_logits -= family_logits.max(axis=1, keepdims=True)
    family = np.exp(family_logits)
    family /= family.sum(axis=1, keepdims=True)
    child_logits = np.log(np.clip(child_values, 1e-12, 1))
    child_logits += scale * child_delta
    child_logits -= child_logits.max(axis=2, keepdims=True)
    child = np.exp(child_logits)
    child /= child.sum(axis=2, keepdims=True)
    probabilities = validate_probability_matrix(
        (family[:, :, None] * child).reshape(-1, 6)
    )
    mode = str(transform["calibrationMode"])
    parameters = np.asarray(transform["calibrationParameters"], dtype=float)
    if mode == "identity":
        return probabilities
    if mode == "temperature":
        return _temperature(probabilities, float(parameters[0]))
    if mode == "hierarchical":
        return _hierarchical_calibration(probabilities, parameters)
    raise ValueError(f"unknown V8.4 calibration mode: {mode}")


class V84SequenceEnsemble:
    """Load the three frozen sequence experts and average their predictions."""

    def __init__(self, model_directory: Path):
        if torch is None:
            raise ModuleNotFoundError(
                "V8.4 live inference requires `uv sync --extra sequence`"
            )
        metadata_path = model_directory / "metadata.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoints = self.metadata.get("checkpoints") or [
            path.name for path in sorted(model_directory.glob("sequence-seed-*.pt"))
        ]
        transforms = self.metadata.get("transforms") or []
        if not checkpoints or len(checkpoints) != len(transforms):
            raise ValueError("V8.4 checkpoints and transforms must align")
        self.experts = [
            self._load(model_directory / str(name)) for name in checkpoints
        ]
        self.transforms = transforms

    @staticmethod
    def _load(path: Path) -> LoadedExpert:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = GlobalConditionedSequenceResidual(
            int(payload["description_vocab_size"]),
            length=int(payload["length"]),
            current_numeric_width=int(payload["current_numeric_width"]),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return LoadedExpert(model=model, normalizer=payload["normalizer"])

    def predict(
        self,
        examples: SequenceExamples,
        global_probabilities: np.ndarray,
    ) -> np.ndarray:
        index = np.array([len(examples) - 1], dtype=int)
        members = []
        with torch.no_grad():
            for expert, transform in zip(
                self.experts,
                self.transforms,
                strict=True,
            ):
                raw = examples.batch(index, expert.normalizer)
                batch = {
                    key: torch.as_tensor(value)
                    for key, value in raw.items()
                }
                batch["global_probabilities"] = torch.as_tensor(
                    global_probabilities,
                    dtype=torch.float32,
                )
                family, child = expert.model(batch)
                members.append(
                    _transform(
                        global_probabilities,
                        family.numpy(),
                        child.numpy(),
                        transform,
                    )
                )
        return validate_probability_matrix(np.mean(members, axis=0))
