"""Model-selection helpers with explicit balance gates."""

from __future__ import annotations

import gc
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .features import (
    CATEGORICAL_FEATURES,
    GLOBAL_CATEGORICAL_FEATURES,
    GLOBAL_NUMERIC_FEATURES,
    LEGACY_CATEGORICAL_FEATURES,
    LEGACY_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
)
from .metrics import evaluate_diagnostics, validate_probability_matrix
from .resources import assert_safe, snapshot
from .taxonomy import PITCH_GROUPS


@dataclass(frozen=True)
class CandidateSpec:
    feature_set: str
    weight_mode: str
    max_depth: int
    min_child_weight: int

    @property
    def name(self) -> str:
        return (
            f"{self.feature_set}-{self.weight_mode}"
            f"-d{self.max_depth}-m{self.min_child_weight}"
        )

    @property
    def numeric_features(self) -> tuple[str, ...]:
        if self.feature_set == "legacy":
            return LEGACY_NUMERIC_FEATURES
        if self.feature_set == "global":
            return GLOBAL_NUMERIC_FEATURES
        return NUMERIC_FEATURES

    @property
    def categorical_features(self) -> tuple[str, ...]:
        if self.feature_set == "legacy":
            return LEGACY_CATEGORICAL_FEATURES
        if self.feature_set == "global":
            return GLOBAL_CATEGORICAL_FEATURES
        return CATEGORICAL_FEATURES


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_through: int
    evaluate_year: int


FOLD_SPECS = (
    FoldSpec("through-2021-to-2022", 2021, 2022),
    FoldSpec("through-2022-to-2023", 2022, 2023),
    FoldSpec("through-2023-to-2024", 2023, 2024),
)
REFERENCE_NAME = "legacy-none-d6-m3"


@dataclass
class FittedCandidate:
    spec: CandidateSpec
    preprocessor: ColumnTransformer
    model: XGBClassifier
    device: str
    class_indices: tuple[int, ...]


def candidate_specs() -> tuple[CandidateSpec, ...]:
    return tuple(
        CandidateSpec(feature_set, weight_mode, max_depth, min_child_weight)
        for feature_set, weight_mode, max_depth, min_child_weight in product(
            ("legacy", "enriched"),
            ("none", "light", "sqrt"),
            (4, 6),
            (3, 8),
        )
    )


def time_fold_masks(
    rows: pd.DataFrame,
) -> list[tuple[FoldSpec, pd.Series, pd.Series]]:
    years = rows["game_date"].dt.year
    folds = []
    for fold in FOLD_SPECS:
        train = years <= fold.train_through
        evaluation = years == fold.evaluate_year
        if not train.any() or not evaluation.any():
            raise ValueError(f"empty time fold: {fold.name}")
        train_end = rows.loc[train, "game_date"].max()
        evaluation_start = rows.loc[evaluation, "game_date"].min()
        if train_end >= evaluation_start:
            raise AssertionError(f"time fold overlaps: {fold.name}")
        folds.append((fold, train, evaluation))
    return folds


def _split_core_tuning(
    rows: pd.DataFrame, tuning_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(rows["game_date"].dt.normalize().unique()))
    split_index = max(1, min(len(dates) - 1, int(len(dates) * (1 - tuning_fraction))))
    cutoff = dates[split_index]
    core = rows[rows["game_date"].dt.normalize() < cutoff].copy()
    tuning = rows[rows["game_date"].dt.normalize() >= cutoff].copy()
    overlaps = core["game_date"].max() >= tuning["game_date"].min()
    if core.empty or tuning.empty or overlaps:
        raise ValueError("core/tuning split is not chronological")
    return core, tuning


def _preprocessor(spec: CandidateSpec) -> ColumnTransformer:
    categorical = Pipeline(
        [
            ("missing", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), list(spec.numeric_features)),
            ("categorical", categorical, list(spec.categorical_features)),
        ]
    )


def _model(
    spec: CandidateSpec,
    *,
    device: str,
    n_estimators: int,
    early_stopping: bool,
    num_classes: int = len(PITCH_GROUPS),
) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=spec.max_depth,
        min_child_weight=spec.min_child_weight,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=2.0,
        tree_method="hist",
        device=device,
        eval_metric="mlogloss",
        early_stopping_rounds=75 if early_stopping else None,
        n_jobs=8,
        random_state=737,
    )


def _fit_with_fallback(
    spec: CandidateSpec,
    x_train: Any,
    y_train: np.ndarray,
    train_weights: np.ndarray,
    *,
    x_tuning: Any | None = None,
    y_tuning: np.ndarray | None = None,
    n_estimators: int = 2_000,
    num_classes: int = len(PITCH_GROUPS),
) -> tuple[XGBClassifier, str]:
    early_stopping = x_tuning is not None
    for device in ("cuda", "cpu"):
        model = _model(
            spec,
            device=device,
            n_estimators=n_estimators,
            early_stopping=early_stopping,
            num_classes=num_classes,
        )
        fit_arguments: dict[str, object] = {
            "sample_weight": train_weights,
            "verbose": False,
        }
        if early_stopping:
            fit_arguments["eval_set"] = [(x_tuning, y_tuning)]
        try:
            model.fit(x_train, y_train, **fit_arguments)
            return model, device
        except Exception:
            if device == "cpu":
                raise
    raise AssertionError("unreachable")


def _best_tree_count(model: XGBClassifier) -> int:
    best_iteration = getattr(model, "best_iteration", None)
    return int(best_iteration + 1) if best_iteration is not None else model.n_estimators


def run_oof_experiment(rows: pd.DataFrame) -> dict[str, object]:
    """Evaluate the fixed candidate matrix on the three chronological folds."""
    specs = candidate_specs()
    predictions: dict[str, list[np.ndarray]] = {spec.name: [] for spec in specs}
    best_iterations: dict[str, list[int]] = {spec.name: [] for spec in specs}
    devices: dict[str, set[str]] = {spec.name: set() for spec in specs}
    actual_parts: list[np.ndarray] = []

    for fold, train_mask, evaluation_mask in time_fold_masks(rows):
        train_rows = rows[train_mask]
        evaluation = rows[evaluation_mask]
        core, tuning = _split_core_tuning(train_rows)
        actual_parts.append(evaluation["target"].to_numpy())
        print(
            f"[{fold.name}] core={len(core)} tuning={len(tuning)} "
            f"evaluation={len(evaluation)}"
        )

        for feature_set in ("legacy", "enriched"):
            feature_specs = [spec for spec in specs if spec.feature_set == feature_set]
            representative = feature_specs[0]
            feature_names = (
                *representative.numeric_features,
                *representative.categorical_features,
            )
            preprocessor = _preprocessor(representative)
            x_core = preprocessor.fit_transform(core[list(feature_names)])
            x_tuning = preprocessor.transform(tuning[list(feature_names)])
            x_evaluation = preprocessor.transform(evaluation[list(feature_names)])

            for spec in feature_specs:
                assert_safe(snapshot())
                model, device = _fit_with_fallback(
                    spec,
                    x_core,
                    core["target"].to_numpy(),
                    class_sample_weights(core["target"].to_numpy(), spec.weight_mode),
                    x_tuning=x_tuning,
                    y_tuning=tuning["target"].to_numpy(),
                )
                probabilities = model.predict_proba(x_evaluation)
                predictions[spec.name].append(probabilities)
                best_iterations[spec.name].append(_best_tree_count(model))
                devices[spec.name].add(device)
                print(f"  {spec.name}: trees={_best_tree_count(model)} device={device}")
                del model
                gc.collect()

            del preprocessor, x_core, x_tuning, x_evaluation
            gc.collect()

    actual = np.concatenate(actual_parts)
    names = [str(group) for group in PITCH_GROUPS]
    results = []
    for spec in specs:
        raw = np.concatenate(predictions[spec.name])
        temperature = fit_temperature(actual, raw)
        calibrated = apply_temperature(raw, temperature)
        metrics = evaluate_diagnostics(
            actual,
            calibrated,
            labels=range(len(PITCH_GROUPS)),
            names=names,
        )
        results.append(
            {
                "name": spec.name,
                "spec": {
                    "featureSet": spec.feature_set,
                    "weightMode": spec.weight_mode,
                    "maxDepth": spec.max_depth,
                    "minChildWeight": spec.min_child_weight,
                },
                "metrics": metrics,
                "temperature": temperature,
                "bestIterations": best_iterations[spec.name],
                "finalTreeCount": int(np.median(best_iterations[spec.name])),
                "devices": sorted(devices[spec.name]),
            }
        )

    reference = next(result for result in results if result["name"] == REFERENCE_NAME)
    selected = select_candidate(
        reference["metrics"],
        [result for result in results if result["name"] != REFERENCE_NAME],
    )
    accepted = selected is not None
    selected = selected or reference
    for result in results:
        result["accepted"] = accepted and result["name"] == selected["name"]
    return {
        "folds": [fold.name for fold in FOLD_SPECS],
        "referenceName": REFERENCE_NAME,
        "selectedName": selected["name"],
        "accepted": accepted,
        "candidates": results,
    }


def train_final_candidate(
    rows: pd.DataFrame, spec: CandidateSpec, n_estimators: int
) -> FittedCandidate:
    features = (*spec.numeric_features, *spec.categorical_features)
    preprocessor = _preprocessor(spec)
    transformed = preprocessor.fit_transform(rows[list(features)])
    classes = tuple(int(value) for value in sorted(rows["target"].unique()))
    labels = np.searchsorted(classes, rows["target"].to_numpy())
    model, device = _fit_with_fallback(
        spec,
        transformed,
        labels,
        class_sample_weights(labels, spec.weight_mode),
        n_estimators=n_estimators,
        num_classes=len(classes),
    )
    return FittedCandidate(spec, preprocessor, model, device, classes)


def train_candidate_with_tuning(
    rows: pd.DataFrame,
    spec: CandidateSpec,
    *,
    n_estimators: int = 2_000,
) -> tuple[FittedCandidate, int]:
    """Fit on a chronological core with an internal early-stopping tail."""
    core, tuning = _split_core_tuning(rows)
    classes = tuple(int(value) for value in sorted(core["target"].unique()))
    tuning = tuning[tuning["target"].isin(classes)]
    if tuning.empty:
        raise ValueError("tuning tail has no target classes seen in training core")
    features = (*spec.numeric_features, *spec.categorical_features)
    preprocessor = _preprocessor(spec)
    x_core = preprocessor.fit_transform(core[list(features)])
    x_tuning = preprocessor.transform(tuning[list(features)])
    core_labels = np.searchsorted(classes, core["target"].to_numpy())
    tuning_labels = np.searchsorted(classes, tuning["target"].to_numpy())
    model, device = _fit_with_fallback(
        spec,
        x_core,
        core_labels,
        class_sample_weights(core_labels, spec.weight_mode),
        x_tuning=x_tuning,
        y_tuning=tuning_labels,
        n_estimators=n_estimators,
        num_classes=len(classes),
    )
    return (
        FittedCandidate(spec, preprocessor, model, device, classes),
        _best_tree_count(model),
    )


def predict_candidate(
    fitted: FittedCandidate, rows: pd.DataFrame, temperature: float
) -> np.ndarray:
    features = (*fitted.spec.numeric_features, *fitted.spec.categorical_features)
    transformed = fitted.preprocessor.transform(rows[list(features)])
    raw = fitted.model.predict_proba(transformed)
    probabilities = np.zeros((len(rows), len(PITCH_GROUPS)), dtype=float)
    probabilities[:, fitted.class_indices] = raw
    return apply_temperature(probabilities, temperature)


def find_spec(name: str) -> CandidateSpec:
    return next(spec for spec in candidate_specs() if spec.name == name)


def class_sample_weights(labels: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    exponents = {"none": 0.0, "light": 0.25, "sqrt": 0.5}
    if mode not in exponents:
        raise ValueError(f"unknown class-weight mode: {mode}")
    exponent = exponents[mode]
    if exponent == 0:
        return np.ones(len(y), dtype="float32")
    counts = np.bincount(y)
    present = counts > 0
    class_weights = np.ones(len(counts), dtype=float)
    imbalance = len(y) / (int(present.sum()) * counts[present])
    class_weights[present] = imbalance**exponent
    weights = np.clip(class_weights[y], 0.5, 3.0)
    weights /= weights.mean()
    return weights.astype("float32")


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = validate_probability_matrix(probabilities)
    logits = np.log(np.clip(values, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    adjusted = np.exp(logits)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def fit_temperature(actual: Sequence[int], probabilities: np.ndarray) -> float:
    y = np.asarray(actual, dtype=int)
    values = validate_probability_matrix(probabilities)

    def objective(temperature: float) -> float:
        adjusted = apply_temperature(values, temperature)
        return float(-np.log(np.clip(adjusted[np.arange(len(y)), y], 1e-12, 1)).mean())

    result = minimize_scalar(objective, bounds=(0.5, 3.0), method="bounded")
    candidate = float(result.x)
    return candidate if objective(candidate) < objective(1.0) else 1.0


def _passes_balance_gates(
    reference: dict[str, object], candidate: dict[str, object]
) -> bool:
    return (
        float(candidate["logLoss"]) < float(reference["logLoss"])
        and float(candidate["macroF1"]) > float(reference["macroF1"])
        and float(candidate["accuracy"]) >= float(reference["accuracy"]) - 0.005
        and not candidate["zeroRecallClasses"]
        and float(candidate["majorityPredictionGap"]) <= 0.20
    )


def select_candidate(
    reference: dict[str, object],
    candidates: Sequence[dict[str, object]],
) -> dict[str, object] | None:
    eligible = [
        candidate
        for candidate in candidates
        if _passes_balance_gates(reference, candidate["metrics"])
    ]
    if not eligible:
        return None
    best_loss = min(float(candidate["metrics"]["logLoss"]) for candidate in eligible)
    tied = [
        candidate
        for candidate in eligible
        if float(candidate["metrics"]["logLoss"]) <= best_loss + 0.005
    ]
    return max(tied, key=lambda candidate: float(candidate["metrics"]["macroF1"]))
