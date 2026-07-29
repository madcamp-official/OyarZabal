"""Robust Global-conditioned hierarchical sequence residual."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .metrics import validate_probability_matrix
from .sequence import (
    SEQUENCE_LENGTH,
    TOKEN_NUMERIC_COLUMNS,
    SequenceExamples,
    SequenceNormalizer,
    _batches,
    _require_torch,
    _tensor_batch,
    torch,
)

if torch is not None:
    from torch import nn
    from torch.nn import functional as F
else:
    nn = None
    F = None

OPTIONAL_PHYSICAL_START = TOKEN_NUMERIC_COLUMNS.index("release_spin_rate")


@dataclass(frozen=True)
class SequenceObjective:
    soft_target_strength: float = 0.0
    focal_gamma: float = 0.0
    group_balanced: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.soft_target_strength <= 0.5:
            raise ValueError("soft target strength must be between zero and 0.5")
        if self.focal_gamma not in {0.0, 1.0}:
            raise ValueError("focal gamma must be zero or one")


def mild_class_weights(
    labels: np.ndarray,
    *,
    classes: int,
) -> np.ndarray:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=classes)
    if (counts == 0).any():
        raise ValueError("class weights require every class")
    raw = counts.astype(float) ** -0.25
    low, high = 0.0, 100.0
    for _ in range(60):
        scale = (low + high) / 2
        if np.clip(raw * scale, 0.75, 1.5).mean() < 1:
            low = scale
        else:
            high = scale
    return np.clip(raw * ((low + high) / 2), 0.75, 1.5).astype(np.float32)


def mild_family_weights(labels: np.ndarray) -> np.ndarray:
    return mild_class_weights(labels, classes=3)


def hierarchical_residual_probabilities(
    global_probabilities: np.ndarray,
    family_delta: np.ndarray,
    child_delta: np.ndarray,
    scale: float,
) -> np.ndarray:
    global_values = validate_probability_matrix(global_probabilities)
    family_values = global_values.reshape(-1, 3, 2).sum(axis=2)
    child_values = global_values.reshape(-1, 3, 2) / family_values[:, :, None]
    if not 0 <= scale <= 1:
        raise ValueError("sequence residual scale must be between zero and one")
    if scale == 0:
        return global_values.copy()
    family_logits = np.log(np.clip(family_values, 1e-12, 1))
    family_logits += scale * np.asarray(family_delta, dtype=float)
    family_logits -= family_logits.max(axis=1, keepdims=True)
    family = np.exp(family_logits)
    family /= family.sum(axis=1, keepdims=True)
    child_logits = np.log(np.clip(child_values, 1e-12, 1))
    child_logits += scale * np.asarray(child_delta, dtype=float)
    child_logits -= child_logits.max(axis=2, keepdims=True)
    child = np.exp(child_logits)
    child /= child.sum(axis=2, keepdims=True)
    return validate_probability_matrix((family[:, :, None] * child).reshape(-1, 6))


def apply_hierarchical_calibration(
    probabilities: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    values = validate_probability_matrix(probabilities).reshape(-1, 3, 2)
    params = np.asarray(parameters, dtype=float)
    if params.shape != (9,):
        raise ValueError("hierarchical calibration needs nine parameters")
    family = values.sum(axis=2)
    child = values / family[:, :, None]
    family_bias = params[:3] - params[:3].mean()
    child_bias = params[3:].reshape(3, 2)
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


def fit_hierarchical_calibration(
    actual: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(actual, dtype=int)
    values = validate_probability_matrix(probabilities)
    rows = np.arange(len(labels))

    def objective(parameters: np.ndarray) -> float:
        adjusted = apply_hierarchical_calibration(values, parameters)
        loss = -np.log(np.clip(adjusted[rows, labels], 1e-12, 1)).mean()
        return float(loss + 0.001 * np.square(parameters).mean())

    fitted = minimize(
        objective,
        np.zeros(9, dtype=float),
        method="L-BFGS-B",
        bounds=[(-1.5, 1.5)] * 9,
    )
    if not fitted.success:
        raise RuntimeError(f"hierarchical calibration failed: {fitted.message}")
    return np.asarray(fitted.x, dtype=float)


if nn is not None:

    class GlobalConditionedSequenceResidual(nn.Module):
        def __init__(
            self,
            description_vocab_size: int,
            *,
            length: int = SEQUENCE_LENGTH,
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
                layer, num_layers=2, enable_nested_tensor=False
            )
            self.normalization = nn.LayerNorm(64)
            self.family_delta = nn.Linear(64, 3)
            self.child_delta = nn.Linear(64, 6)
            nn.init.zeros_(self.family_delta.weight)
            nn.init.zeros_(self.family_delta.bias)
            nn.init.zeros_(self.child_delta.weight)
            nn.init.zeros_(self.child_delta.bias)

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

        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            categorical = batch["token_categorical"].long()
            pieces = [
                self.group_embedding(categorical[..., 0]),
                self.family_embedding(categorical[..., 1]),
                self.ball_embedding(categorical[..., 2]),
                self.strike_embedding(categorical[..., 3]),
                self.description_embedding(categorical[..., 4]),
                self.stand_embedding(categorical[..., 5]),
                batch["history_flags"].float(),
                batch["token_numeric"].float(),
                batch["token_observed"].float(),
            ]
            tokens = self.token_projection(torch.cat(pieces, dim=-1))
            positions = torch.arange(self.length, device=tokens.device)
            tokens = tokens + self.position_embedding(positions)[None, :, :]
            encoded = self.encoder(
                tokens,
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
            family_delta = self.family_delta(state)
            child_delta = self.child_delta(state).reshape(-1, 3, 2)
            global_child = global_values.reshape(-1, 3, 2)
            global_family = global_child.sum(dim=2)
            global_conditional = global_child / global_family[:, :, None]
            family_logits = global_family.log() + family_delta
            child_logits = global_conditional.log() + child_delta
            family = family_logits.softmax(dim=1)
            child = child_logits.softmax(dim=2)
            return {
                "family_delta": family_delta,
                "child_delta": child_delta,
                "family_logits": family_logits,
                "group_probabilities": (family[:, :, None] * child).reshape(-1, 6),
            }

        @staticmethod
        def loss(
            output: dict[str, torch.Tensor],
            batch: dict[str, torch.Tensor],
            *,
            balance_strength: float,
            family_weights: torch.Tensor,
            objective: SequenceObjective | None = None,
            group_weights: torch.Tensor | None = None,
        ) -> torch.Tensor:
            objective = objective or SequenceObjective()
            rows = torch.arange(
                len(batch["target_group"]), device=family_weights.device
            )
            group = output["group_probabilities"]
            truth = batch["target_group"].long()
            log_group = group.clamp_min(1e-12).log()
            truth_probability = group[rows, truth].clamp_min(1e-12)
            nll = -log_group[rows, truth]
            if objective.soft_target_strength:
                teacher = batch["global_probabilities"].float().detach()
                distillation = -(teacher * log_group).sum(dim=1)
                strength = objective.soft_target_strength
                nll = (1 - strength) * nll + strength * distillation
            if objective.focal_gamma:
                nll = nll * (1 - truth_probability).pow(objective.focal_gamma)
            if objective.group_balanced:
                if group_weights is None:
                    raise ValueError("group-balanced loss needs group weights")
                sample_weights = group_weights[truth]
                nll = nll * sample_weights / sample_weights.mean()
            balanced = F.cross_entropy(
                output["family_logits"],
                batch["target_family"].long(),
                weight=family_weights,
            )
            penalty = output["family_delta"].square().mean()
            penalty += output["child_delta"].square().mean()
            return nll.mean() + balance_strength * balanced + 0.001 * penalty

else:

    class GlobalConditionedSequenceResidual:
        def __init__(self, *_args: object, **_kwargs: object):
            _require_torch()


@dataclass
class FittedV83Expert:
    model: GlobalConditionedSequenceResidual
    normalizer: SequenceNormalizer
    validation_log_loss: float
    epochs: int
    balance_strength: float
    block_dropout: float
    family_weights: np.ndarray
    objective: SequenceObjective = field(default_factory=SequenceObjective)
    group_weights: np.ndarray = field(
        default_factory=lambda: np.ones(6, dtype=np.float32)
    )


def _batch(
    examples: SequenceExamples,
    indices: np.ndarray,
    normalizer: SequenceNormalizer,
    global_probabilities: np.ndarray,
    device: str,
) -> dict[str, torch.Tensor]:
    batch = _tensor_batch(examples, indices, normalizer, device)
    batch["global_probabilities"] = torch.as_tensor(
        global_probabilities[indices], device=device
    )
    return batch


def predict_v83_deltas(
    fitted: FittedV83Expert,
    examples: SequenceExamples,
    indices: np.ndarray,
    global_probabilities: np.ndarray,
    *,
    batch_size: int = 4096,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    _require_torch()
    selected_device = device or next(fitted.model.parameters()).device.type
    fitted.model.eval()
    family, child = [], []
    with torch.no_grad():
        for positions in _batches(indices, batch_size, shuffle=False, seed=0):
            output = fitted.model(
                _batch(
                    examples,
                    positions,
                    fitted.normalizer,
                    global_probabilities,
                    selected_device,
                )
            )
            family.append(output["family_delta"].cpu().numpy())
            child.append(output["child_delta"].cpu().numpy())
    return np.concatenate(family), np.concatenate(child)


def _train_epoch(
    model: GlobalConditionedSequenceResidual,
    examples: SequenceExamples,
    train_indices: np.ndarray,
    global_probabilities: np.ndarray,
    normalizer: SequenceNormalizer,
    optimizer: torch.optim.Optimizer,
    family_weights: torch.Tensor,
    group_weights: torch.Tensor,
    objective: SequenceObjective,
    *,
    balance_strength: float,
    block_dropout: float,
    batch_size: int,
    seed: int,
    epoch: int,
    device: str,
    rng: np.random.Generator,
) -> None:
    model.train()
    for positions in _batches(
        train_indices,
        batch_size,
        shuffle=True,
        seed=seed + epoch,
    ):
        batch = _batch(
            examples,
            positions,
            normalizer,
            global_probabilities,
            device,
        )
        if block_dropout and rng.random() < block_dropout:
            batch["token_numeric"][..., OPTIONAL_PHYSICAL_START:] = 0
            batch["token_observed"][..., OPTIONAL_PHYSICAL_START:] = 0
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(
            model(batch),
            batch,
            balance_strength=balance_strength,
            family_weights=family_weights,
            objective=objective,
            group_weights=group_weights,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()


def fit_v83_expert(
    examples: SequenceExamples,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    global_probabilities: np.ndarray,
    *,
    description_vocab_size: int,
    balance_strength: float,
    block_dropout: float,
    objective: SequenceObjective | None = None,
    epochs: int = 6,
    batch_size: int = 8192,
    seed: int = 83,
    device: str | None = None,
) -> FittedV83Expert:
    _require_torch()
    objective = objective or SequenceObjective()
    if balance_strength not in {0.0, 0.1, 0.2}:
        raise ValueError("balance strength must be 0, 0.1, or 0.2")
    if block_dropout not in {0.0, 0.2}:
        raise ValueError("physical block dropout must be 0 or 0.2")
    if len(global_probabilities) != len(examples):
        raise ValueError("global probabilities and examples differ")
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    normalizer = SequenceNormalizer.fit(examples, train_indices)
    weights = mild_family_weights(examples.target_families[train_indices])
    weight_tensor = torch.as_tensor(weights, device=selected_device)
    group_weights = mild_class_weights(
        examples.target_groups[train_indices],
        classes=6,
    )
    group_weight_tensor = torch.as_tensor(group_weights, device=selected_device)
    model = GlobalConditionedSequenceResidual(
        description_vocab_size,
        length=examples.history_indices.shape[1],
        current_numeric_width=examples.current_numeric.shape[1],
    ).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        _train_epoch(
            model,
            examples,
            train_indices,
            global_probabilities,
            normalizer,
            optimizer,
            weight_tensor,
            group_weight_tensor,
            objective,
            balance_strength=balance_strength,
            block_dropout=block_dropout,
            batch_size=batch_size,
            seed=seed,
            epoch=epoch,
            device=selected_device,
            rng=rng,
        )
        temporary = FittedV83Expert(
            model,
            normalizer,
            float("inf"),
            epoch,
            balance_strength,
            block_dropout,
            weights,
        )
        family, child = predict_v83_deltas(
            temporary,
            examples,
            validation_indices,
            global_probabilities,
            batch_size=batch_size,
            device=selected_device,
        )
        probabilities = hierarchical_residual_probabilities(
            global_probabilities[validation_indices], family, child, 1
        )
        truth = examples.target_groups[validation_indices]
        validation_loss = float(
            -np.log(
                np.clip(probabilities[np.arange(len(truth)), truth], 1e-12, 1)
            ).mean()
        )
        if validation_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= 2:
                break
    if best_state is None:
        raise RuntimeError("V8.3 training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return FittedV83Expert(
        model,
        normalizer,
        best_loss,
        best_epoch,
        balance_strength,
        block_dropout,
        weights,
        objective,
        group_weights,
    )


def refit_v83_expert(
    examples: SequenceExamples,
    train_indices: np.ndarray,
    global_probabilities: np.ndarray,
    *,
    description_vocab_size: int,
    balance_strength: float,
    block_dropout: float,
    objective: SequenceObjective,
    epochs: int,
    batch_size: int = 8192,
    seed: int = 83,
    device: str | None = None,
) -> FittedV83Expert:
    """Refit fixed V8.4 settings on every supplied pre-holdout row."""
    _require_torch()
    if epochs <= 0:
        raise ValueError("refit epochs must be positive")
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    normalizer = SequenceNormalizer.fit(examples, train_indices)
    family_weights = mild_family_weights(examples.target_families[train_indices])
    group_weights = mild_class_weights(
        examples.target_groups[train_indices],
        classes=6,
    )
    family_tensor = torch.as_tensor(family_weights, device=selected_device)
    group_tensor = torch.as_tensor(group_weights, device=selected_device)
    model = GlobalConditionedSequenceResidual(
        description_vocab_size,
        length=examples.history_indices.shape[1],
        current_numeric_width=examples.current_numeric.shape[1],
    ).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, epochs + 1):
        _train_epoch(
            model,
            examples,
            train_indices,
            global_probabilities,
            normalizer,
            optimizer,
            family_tensor,
            group_tensor,
            objective,
            balance_strength=balance_strength,
            block_dropout=block_dropout,
            batch_size=batch_size,
            seed=seed,
            epoch=epoch,
            device=selected_device,
            rng=rng,
        )
    return FittedV83Expert(
        model=model,
        normalizer=normalizer,
        validation_log_loss=float("nan"),
        epochs=epochs,
        balance_strength=balance_strength,
        block_dropout=block_dropout,
        family_weights=family_weights,
        objective=objective,
        group_weights=group_weights,
    )


def save_v83_expert(fitted: FittedV83Expert, path: Path) -> None:
    _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": fitted.model.state_dict(),
            "normalizer": fitted.normalizer,
            "validation_log_loss": fitted.validation_log_loss,
            "epochs": fitted.epochs,
            "balance_strength": fitted.balance_strength,
            "block_dropout": fitted.block_dropout,
            "family_weights": fitted.family_weights,
            "objective": {
                "soft_target_strength": fitted.objective.soft_target_strength,
                "focal_gamma": fitted.objective.focal_gamma,
                "group_balanced": fitted.objective.group_balanced,
            },
            "group_weights": fitted.group_weights,
            "length": fitted.model.length,
            "current_numeric_width": (fitted.model.current_projection.in_features - 15)
            // 2,
            "description_vocab_size": fitted.model.description_embedding.num_embeddings,
        },
        path,
    )


def load_v83_expert(path: Path, *, device: str = "cpu") -> FittedV83Expert:
    _require_torch()
    payload = torch.load(path, map_location=device, weights_only=False)
    model = GlobalConditionedSequenceResidual(
        int(payload["description_vocab_size"]),
        length=int(payload["length"]),
        current_numeric_width=int(payload["current_numeric_width"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    return FittedV83Expert(
        model=model,
        normalizer=payload["normalizer"],
        validation_log_loss=float(payload["validation_log_loss"]),
        epochs=int(payload["epochs"]),
        balance_strength=float(payload["balance_strength"]),
        block_dropout=float(payload["block_dropout"]),
        family_weights=np.asarray(payload["family_weights"], dtype=np.float32),
        objective=SequenceObjective(**payload.get("objective", {})),
        group_weights=np.asarray(
            payload.get("group_weights", np.ones(6)),
            dtype=np.float32,
        ),
    )
