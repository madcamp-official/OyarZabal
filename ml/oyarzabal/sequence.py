"""Leakage-safe V8 sequence examples and a small optional PyTorch expert."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import validate_probability_matrix
from .taxonomy import (
    PITCH_FAMILIES,
    PITCH_GROUP_FAMILIES,
    PITCH_GROUPS,
    PitchGroup,
    context_pitch,
    group_pitch,
)

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError:  # Optional `sequence` extra.
    torch = None
    nn = None
    F = None

SEQUENCE_LENGTH = 16
TOKEN_NUMERIC_COLUMNS = (
    "release_speed",
    "plate_x",
    "plate_z",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "sz_top",
    "sz_bot",
)
CURRENT_NUMERIC_COLUMNS = (
    "outs_when_up",
    "inning",
    "on_1b",
    "on_2b",
    "on_3b",
    "pitch_number",
    "n_thruorder_pitcher",
)
SORT_COLUMNS = ("game_date", "game_pk", "at_bat_number", "pitch_number")

_GROUP_INDEX = {group: index + 2 for index, group in enumerate(PITCH_GROUPS)}
_FAMILY_INDEX = {
    family: index + 2 for index, family in enumerate(PITCH_FAMILIES)
}
_TARGET_INDEX = {group: index for index, group in enumerate(PITCH_GROUPS)}
_FAMILY_TARGET = {
    group: PITCH_FAMILIES.index(PITCH_GROUP_FAMILIES[group])
    for group in PITCH_GROUPS
}
_STAND_INDEX = {"L": 1, "R": 2, "S": 3}


def _column(rows: pd.DataFrame, name: str, default: object) -> pd.Series:
    if name in rows:
        return rows[name]
    return pd.Series(default, index=rows.index)


def _numeric(rows: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(
        _column(rows, name, np.nan),
        errors="coerce",
    ).to_numpy(dtype=np.float32)


def _entity_map(values: pd.Series) -> dict[int, int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype("int64")
    return {
        int(value): index + 1
        for index, value in enumerate(sorted(numeric.unique()))
    }


@dataclass(frozen=True)
class SequenceVocabulary:
    """Fold-local vocabularies; index zero is padding/OOV."""

    descriptions: Mapping[str, int]
    pitchers: Mapping[int, int]
    batters: Mapping[int, int]
    catchers: Mapping[int, int]

    @classmethod
    def fit(cls, train_rows: pd.DataFrame) -> SequenceVocabulary:
        descriptions = sorted(
            _column(train_rows, "description", "UNKNOWN")
            .fillna("UNKNOWN")
            .astype(str)
            .unique()
        )
        return cls(
            descriptions={
                value: index + 1 for index, value in enumerate(descriptions)
            },
            pitchers=_entity_map(_column(train_rows, "pitcher", np.nan)),
            batters=_entity_map(_column(train_rows, "batter", np.nan)),
            catchers=_entity_map(_column(train_rows, "fielder_2", np.nan)),
        )


@dataclass(frozen=True)
class SequenceNormalizer:
    token_mean: np.ndarray
    token_scale: np.ndarray
    current_mean: np.ndarray
    current_scale: np.ndarray

    @staticmethod
    def _moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.nanmean(values, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0).astype(np.float32)
        filled = np.where(np.isfinite(values), values, mean)
        scale = filled.std(axis=0)
        scale = np.where(scale > 1e-6, scale, 1).astype(np.float32)
        return mean, scale

    @classmethod
    def fit(
        cls,
        examples: SequenceExamples,
        example_indices: np.ndarray,
    ) -> SequenceNormalizer:
        if not len(example_indices):
            raise ValueError("normalizer needs training examples")
        cutoff = examples.target_dates[example_indices].max()
        token_values = examples.source_numeric[
            examples.source_dates <= cutoff
        ]
        token_mean, token_scale = cls._moments(token_values)
        current_mean, current_scale = cls._moments(
            examples.current_numeric[example_indices]
        )
        return cls(token_mean, token_scale, current_mean, current_scale)


@dataclass(frozen=True)
class SequenceExamples:
    """Compact examples: history stores row indices, not duplicated tokens."""

    history_indices: np.ndarray
    history_flags: np.ndarray
    source_categorical: np.ndarray
    source_numeric: np.ndarray
    source_catcher_ids: np.ndarray
    source_dates: np.ndarray
    current_categorical: np.ndarray
    current_numeric: np.ndarray
    pitcher_ids: np.ndarray
    batter_ids: np.ndarray
    catcher_ids: np.ndarray
    target_groups: np.ndarray
    target_families: np.ndarray
    target_children: np.ndarray
    target_zones: np.ndarray
    target_dates: np.ndarray
    target_row_indices: np.ndarray

    def __len__(self) -> int:
        return len(self.target_groups)

    def batch(
        self,
        indices: np.ndarray,
        normalizer: SequenceNormalizer,
    ) -> dict[str, np.ndarray]:
        history = self.history_indices[indices]
        padding = history < 0
        safe = np.maximum(history, 0)
        categorical = self.source_categorical[safe].copy()
        numeric = self.source_numeric[safe].copy()
        numeric = np.where(
            np.isfinite(numeric),
            numeric,
            normalizer.token_mean,
        )
        numeric = (
            numeric - normalizer.token_mean
        ) / normalizer.token_scale
        flags = self.history_flags[indices].astype(np.float32, copy=True)
        categorical[padding] = 0
        numeric[padding] = 0
        flags[padding] = 0

        # Keep one valid BOS-like slot so all-padding rows cannot create NaNs.
        empty = padding.all(axis=1)
        padding[empty, -1] = False
        # A causal padded query otherwise has no valid key on PyTorch's eval
        # fast-path; an always-valid zero/BOS key prevents NaN propagation.
        padding[:, 0] = False
        current = self.current_numeric[indices].copy()
        current = np.where(
            np.isfinite(current),
            current,
            normalizer.current_mean,
        )
        current = (
            current - normalizer.current_mean
        ) / normalizer.current_scale
        return {
            "token_categorical": categorical,
            "token_numeric": numeric.astype(np.float32),
            "history_flags": flags,
            "padding_mask": padding,
            "current_categorical": self.current_categorical[indices],
            "current_numeric": current.astype(np.float32),
            "target_group": self.target_groups[indices],
            "target_family": self.target_families[indices],
            "target_child": self.target_children[indices],
            "target_zone": self.target_zones[indices],
        }


class SequenceExampleBuilder:
    """Build each target before appending the current pitch to pitcher history."""

    def __init__(self, length: int = SEQUENCE_LENGTH):
        if length <= 0:
            raise ValueError("sequence length must be positive")
        self.length = length

    def build(
        self,
        ordered_rows: pd.DataFrame,
        vocabulary: SequenceVocabulary,
    ) -> SequenceExamples:
        required = {"pitch_type", "game_date", "game_pk", "pitcher", "batter"}
        missing = required - set(ordered_rows)
        if missing:
            raise ValueError(f"missing sequence columns: {sorted(missing)}")
        rows = ordered_rows.copy()
        rows["game_date"] = pd.to_datetime(rows["game_date"], errors="coerce")
        rows["_context_group"] = rows["pitch_type"].map(context_pitch)
        rows["_target_group"] = rows["pitch_type"].map(group_pitch)
        rows = rows[
            rows["game_date"].notna() & rows["_context_group"].notna()
        ].copy()
        for name in ("at_bat_number", "pitch_number"):
            rows[name] = pd.to_numeric(
                _column(rows, name, 0),
                errors="coerce",
            ).fillna(0)
        rows = rows.sort_values(list(SORT_COLUMNS), kind="stable").reset_index(
            drop=True
        )
        target_count = int(rows["_target_group"].notna().sum())
        history_indices = np.full(
            (target_count, self.length),
            -1,
            dtype=np.int32,
        )
        history_flags = np.zeros(
            (target_count, self.length, 2),
            dtype=np.uint8,
        )

        groups = rows["_context_group"].to_numpy()
        group_tokens = np.array(
            [
                _GROUP_INDEX.get(value, 1)
                if isinstance(value, PitchGroup)
                else 1
                for value in groups
            ],
            dtype=np.int8,
        )
        family_tokens = np.array(
            [
                _FAMILY_INDEX[PITCH_GROUP_FAMILIES[value]]
                if isinstance(value, PitchGroup)
                else 1
                for value in groups
            ],
            dtype=np.int8,
        )
        balls = np.clip(
            np.nan_to_num(_numeric(rows, "balls"), nan=0),
            0,
            3,
        )
        strikes = np.clip(
            np.nan_to_num(_numeric(rows, "strikes"), nan=0),
            0,
            2,
        )
        description_tokens = np.array(
            [
                vocabulary.descriptions.get(str(value), 0)
                for value in _column(rows, "description", "UNKNOWN").fillna(
                    "UNKNOWN"
                )
            ],
            dtype=np.int16,
        )
        stands = np.array(
            [
                _STAND_INDEX.get(str(value).upper(), 0)
                for value in _column(rows, "stand", "UNKNOWN").fillna(
                    "UNKNOWN"
                )
            ],
            dtype=np.int8,
        )
        source_categorical = np.column_stack(
            [
                group_tokens,
                family_tokens,
                balls.astype(np.int8) + 1,
                strikes.astype(np.int8) + 1,
                description_tokens,
                stands,
            ]
        )
        source_numeric = np.column_stack(
            [_numeric(rows, name) for name in TOKEN_NUMERIC_COLUMNS]
        ).astype(np.float32)

        pitcher_raw = pd.to_numeric(rows["pitcher"], errors="coerce").fillna(-1)
        batter_raw = pd.to_numeric(rows["batter"], errors="coerce").fillna(-1)
        catcher_raw = pd.to_numeric(
            _column(rows, "fielder_2", np.nan),
            errors="coerce",
        ).fillna(-1)
        source_catcher_ids = np.array(
            [vocabulary.catchers.get(int(value), 0) for value in catcher_raw],
            dtype=np.int32,
        )
        game_values = rows["game_pk"].to_numpy()
        pa_values = rows["at_bat_number"].to_numpy()
        batter_values = batter_raw.to_numpy(dtype=np.int64)

        current_categorical = np.empty((target_count, 3), dtype=np.int16)
        current_numeric = np.empty(
            (target_count, len(CURRENT_NUMERIC_COLUMNS)),
            dtype=np.float32,
        )
        pitcher_ids = np.empty(target_count, dtype=np.int32)
        batter_ids = np.empty(target_count, dtype=np.int32)
        catcher_ids = np.empty(target_count, dtype=np.int32)
        target_groups = np.empty(target_count, dtype=np.int64)
        target_families = np.empty(target_count, dtype=np.int64)
        target_children = np.empty(target_count, dtype=np.int64)
        target_zones = np.empty(target_count, dtype=np.int64)
        target_dates = np.empty(target_count, dtype="datetime64[D]")
        target_row_indices = np.empty(target_count, dtype=np.int32)
        current_numeric_values = np.column_stack(
            [
                (
                    _column(rows, name, np.nan).notna().to_numpy(
                        dtype=np.float32
                    )
                    if name in {"on_1b", "on_2b", "on_3b"}
                    else _numeric(rows, name)
                )
                for name in CURRENT_NUMERIC_COLUMNS
            ]
        )
        zones = pd.to_numeric(
            _column(rows, "zone", np.nan),
            errors="coerce",
        ).to_numpy()
        dates = rows["game_date"].to_numpy(dtype="datetime64[D]")
        histories: dict[int, deque[int]] = defaultdict(
            lambda: deque(maxlen=self.length)
        )

        example = 0
        for position, target_group in enumerate(rows["_target_group"]):
            pitcher = int(pitcher_raw.iloc[position])
            history = histories[pitcher]
            if target_group is not None and not pd.isna(target_group):
                past = np.asarray(history, dtype=np.int32)
                start = self.length - len(past)
                history_indices[example, start:] = past
                if len(past):
                    same_pa = (
                        (game_values[past] == game_values[position])
                        & (pa_values[past] == pa_values[position])
                    )
                    batter_changed = (
                        batter_values[past] != batter_values[position]
                    )
                    history_flags[example, start:, 0] = same_pa
                    history_flags[example, start:, 1] = batter_changed
                current_categorical[example] = (
                    int(balls[position]) + 1,
                    int(strikes[position]) + 1,
                    stands[position],
                )
                current_numeric[example] = current_numeric_values[position]
                pitcher_ids[example] = vocabulary.pitchers.get(pitcher, 0)
                batter_ids[example] = vocabulary.batters.get(
                    int(batter_values[position]),
                    0,
                )
                catcher_ids[example] = vocabulary.catchers.get(
                    int(catcher_raw.iloc[position]),
                    0,
                )
                group_index = _TARGET_INDEX[target_group]
                target_groups[example] = group_index
                target_families[example] = _FAMILY_TARGET[target_group]
                target_children[example] = group_index % 2
                zone = zones[position]
                target_zones[example] = (
                    int(zone) - 1
                    if np.isfinite(zone) and 1 <= int(zone) <= 14
                    else -100
                )
                target_dates[example] = dates[position]
                target_row_indices[example] = position
                example += 1
            history.append(position)

        return SequenceExamples(
            history_indices=history_indices,
            history_flags=history_flags,
            source_categorical=source_categorical,
            source_numeric=source_numeric,
            source_catcher_ids=source_catcher_ids,
            source_dates=dates,
            current_categorical=current_categorical,
            current_numeric=current_numeric,
            pitcher_ids=pitcher_ids,
            batter_ids=batter_ids,
            catcher_ids=catcher_ids,
            target_groups=target_groups,
            target_families=target_families,
            target_children=target_children,
            target_zones=target_zones,
            target_dates=target_dates,
            target_row_indices=target_row_indices,
        )


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "V8 sequence training requires `uv sync --extra sequence`"
        )


if nn is not None:

    class HierarchicalSequenceTransformer(nn.Module):
        """64-wide, two-layer causal expert; entity IDs are intentionally absent."""

        def __init__(
            self,
            description_vocab_size: int,
            *,
            length: int = SEQUENCE_LENGTH,
            d_model: int = 64,
            layers: int = 2,
            heads: int = 4,
            feedforward: int = 128,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.length = length
            self.group_embedding = nn.Embedding(8, 8, padding_idx=0)
            self.family_embedding = nn.Embedding(5, 4, padding_idx=0)
            self.ball_embedding = nn.Embedding(5, 3, padding_idx=0)
            self.strike_embedding = nn.Embedding(4, 3, padding_idx=0)
            self.description_embedding = nn.Embedding(
                max(2, description_vocab_size),
                8,
                padding_idx=0,
            )
            self.stand_embedding = nn.Embedding(4, 3, padding_idx=0)
            token_width = 8 + 4 + 3 + 3 + 8 + 3 + 2 + len(
                TOKEN_NUMERIC_COLUMNS
            )
            self.token_projection = nn.Linear(token_width, d_model)
            current_width = 3 + 3 + 3 + len(CURRENT_NUMERIC_COLUMNS)
            self.current_projection = nn.Linear(current_width, d_model)
            self.position_embedding = nn.Embedding(length, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=layers,
                enable_nested_tensor=False,
            )
            self.normalization = nn.LayerNorm(d_model)
            self.family_head = nn.Linear(d_model, 3)
            self.child_head = nn.Linear(d_model, 6)
            self.zone_head = nn.Linear(d_model, 6 * 14)

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

        def forward(
            self,
            batch: Mapping[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
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
            ]
            tokens = self.token_projection(torch.cat(pieces, dim=-1))
            positions = torch.arange(
                self.length,
                device=tokens.device,
            )
            tokens = tokens + self.position_embedding(positions)[None, :, :]
            encoded = self.encoder(
                tokens,
                mask=self.causal_mask(tokens.device),
                src_key_padding_mask=batch["padding_mask"].bool(),
                is_causal=True,
            )
            current = batch["current_categorical"].long()
            current_values = torch.cat(
                [
                    self.ball_embedding(current[..., 0]),
                    self.strike_embedding(current[..., 1]),
                    self.stand_embedding(current[..., 2]),
                    batch["current_numeric"].float(),
                ],
                dim=-1,
            )
            state = self.normalization(
                encoded[:, -1] + self.current_projection(current_values)
            )
            family_logits = self.family_head(state)
            child_logits = self.child_head(state).reshape(-1, 3, 2)
            zone_logits = self.zone_head(state).reshape(-1, 6, 14)
            family_probabilities = family_logits.softmax(dim=-1)
            child_probabilities = child_logits.softmax(dim=-1)
            group_probabilities = (
                family_probabilities[:, :, None] * child_probabilities
            ).reshape(-1, 6)
            return {
                "family_logits": family_logits,
                "child_logits": child_logits,
                "zone_logits": zone_logits,
                "group_probabilities": group_probabilities,
            }

        @staticmethod
        def loss(
            output: Mapping[str, torch.Tensor],
            batch: Mapping[str, torch.Tensor],
            *,
            location_weight: float = 0.25,
        ) -> torch.Tensor:
            family_target = batch["target_family"].long()
            group_target = batch["target_group"].long()
            rows = torch.arange(len(group_target), device=group_target.device)
            family_loss = F.cross_entropy(
                output["family_logits"],
                family_target,
            )
            child_loss = F.cross_entropy(
                output["child_logits"][rows, family_target],
                batch["target_child"].long(),
            )
            zone_target = batch["target_zone"].long()
            valid_zone = zone_target != -100
            zone_loss = (
                F.cross_entropy(
                    output["zone_logits"][rows[valid_zone], group_target[valid_zone]],
                    zone_target[valid_zone],
                )
                if valid_zone.any()
                else family_loss.new_zeros(())
            )
            return family_loss + child_loss + location_weight * zone_loss

else:

    class HierarchicalSequenceTransformer:
        def __init__(self, *_args: object, **_kwargs: object):
            _require_torch()


def _tensor_batch(
    examples: SequenceExamples,
    indices: np.ndarray,
    normalizer: SequenceNormalizer,
    device: str,
) -> dict[str, torch.Tensor]:
    _require_torch()
    return {
        name: torch.as_tensor(value, device=device)
        for name, value in examples.batch(indices, normalizer).items()
    }


def _batches(
    indices: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> Iterator[np.ndarray]:
    values = indices.copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(values)
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


@dataclass
class FittedSequenceExpert:
    model: HierarchicalSequenceTransformer
    normalizer: SequenceNormalizer
    location_weight: float
    validation_log_loss: float
    epochs: int


def predict_sequence(
    fitted: FittedSequenceExpert,
    examples: SequenceExamples,
    indices: np.ndarray,
    *,
    batch_size: int = 4096,
    device: str | None = None,
) -> np.ndarray:
    _require_torch()
    selected_device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    fitted.model.to(selected_device).eval()
    predictions = []
    with torch.inference_mode():
        for batch_indices in _batches(
            indices,
            batch_size,
            shuffle=False,
            seed=0,
        ):
            output = fitted.model(
                _tensor_batch(
                    examples,
                    batch_indices,
                    fitted.normalizer,
                    selected_device,
                )
            )
            predictions.append(
                output["group_probabilities"].detach().cpu().numpy()
            )
    return validate_probability_matrix(np.concatenate(predictions))


def fit_sequence_expert(
    examples: SequenceExamples,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    description_vocab_size: int,
    location_weight: float = 0.25,
    epochs: int = 6,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
) -> FittedSequenceExpert:
    _require_torch()
    if location_weight not in {0.0, 0.1, 0.25, 0.5}:
        raise ValueError("location weight must be 0, 0.1, 0.25, or 0.5")
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("sequence train/validation rows cannot be empty")
    torch.manual_seed(seed)
    selected_device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    normalizer = SequenceNormalizer.fit(examples, train_indices)
    model = HierarchicalSequenceTransformer(
        description_vocab_size,
        length=examples.history_indices.shape[1],
    ).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_indices in _batches(
            train_indices,
            batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            batch = _tensor_batch(
                examples,
                batch_indices,
                normalizer,
                selected_device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(
                model(batch),
                batch,
                location_weight=location_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        temporary = FittedSequenceExpert(
            model,
            normalizer,
            location_weight,
            float("inf"),
            epoch,
        )
        probabilities = predict_sequence(
            temporary,
            examples,
            validation_indices,
            batch_size=batch_size,
            device=selected_device,
        )
        truth = examples.target_groups[validation_indices]
        validation_loss = float(
            -np.log(
                np.clip(
                    probabilities[np.arange(len(truth)), truth],
                    1e-12,
                    1,
                )
            ).mean()
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                break
    if best_state is None:
        raise RuntimeError("sequence training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return FittedSequenceExpert(
        model,
        normalizer,
        location_weight,
        best_loss,
        best_epoch,
    )


def blend_log_probabilities(
    global_probabilities: np.ndarray,
    sequence_probabilities: np.ndarray,
    weight: float,
) -> np.ndarray:
    if not 0 <= weight <= 1:
        raise ValueError("sequence blend weight must be between zero and one")
    global_values = validate_probability_matrix(global_probabilities)
    sequence_values = validate_probability_matrix(sequence_probabilities)
    if global_values.shape != sequence_values.shape:
        raise ValueError("global and sequence probabilities are misaligned")
    logits = (1 - weight) * np.log(np.clip(global_values, 1e-12, 1))
    logits += weight * np.log(np.clip(sequence_values, 1e-12, 1))
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return validate_probability_matrix(values)
