"""Consistent training interfaces for RF, RBF-SVM, and XGBoost."""

from __future__ import annotations

import copy
import time
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


SUPPORTED_MODELS = ("random_forest", "svm", "xgboost")


@dataclass
class ModelArtifact:
    """Serializable preprocessing and estimator pair."""

    name: str
    preprocessor: Pipeline
    estimator: BaseEstimator
    feature_columns: list[str]
    selected_parameters: dict[str, Any]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(features[self.feature_columns])
        return np.asarray(self.estimator.predict(transformed), dtype=int)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray | None:
        if not hasattr(self.estimator, "predict_proba"):
            return None
        transformed = self.preprocessor.transform(features[self.feature_columns])
        return np.asarray(self.estimator.predict_proba(transformed))


@dataclass
class TrainingResult:
    """A trained model plus timing and optional tuning diagnostics."""

    artifact: ModelArtifact
    training_seconds: float
    tuning_trials: pd.DataFrame


def _make_preprocessor(model_name: str) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if model_name == "svm":
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _make_estimator(
    model_name: str,
    parameters: Mapping[str, Any],
    *,
    random_seed: int,
    class_count: int,
) -> BaseEstimator:
    params = dict(parameters)
    if model_name == "random_forest":
        params.setdefault("random_state", random_seed)
        return RandomForestClassifier(**params)
    if model_name == "svm":
        params.pop("large_sample_warning", None)
        params.setdefault("random_state", random_seed)
        return SVC(**params)
    if model_name == "xgboost":
        params.setdefault("random_state", random_seed)
        params.setdefault("objective", "multi:softprob")
        params.setdefault("num_class", class_count)
        params.setdefault("eval_metric", "mlogloss")
        params.setdefault("tree_method", "hist")
        return XGBClassifier(**params)
    raise ValueError(
        f"Unsupported model {model_name!r}; choose from {SUPPORTED_MODELS}."
    )


def _fit_once(
    model_name: str,
    parameters: Mapping[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    *,
    random_seed: int,
    feature_columns: Sequence[str],
    sample_weight_train: np.ndarray | None = None,
) -> ModelArtifact:
    if model_name == "svm":
        warning_threshold = int(parameters.get("large_sample_warning", 50000))
        if len(X_train) > warning_threshold:
            warnings.warn(
                f"RBF-SVM received {len(X_train)} training windows, above the "
                f"configured warning threshold {warning_threshold}. The model "
                "was not silently replaced.",
                RuntimeWarning,
                stacklevel=2,
            )

    classes = np.unique(y_train)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError(
            "Training labels must be contiguous zero-based IDs for XGBoost "
            f"compatibility; found {classes.tolist()}."
        )

    preprocessor = _make_preprocessor(model_name)
    transformed_train = preprocessor.fit_transform(X_train)
    transformed_validation = preprocessor.transform(X_validation)
    estimator = _make_estimator(
        model_name,
        parameters,
        random_seed=random_seed,
        class_count=len(classes),
    )

    if model_name == "xgboost":
        sample_weight = (
            sample_weight_train
            if sample_weight_train is not None
            else compute_sample_weight("balanced", y_train)
        )
        estimator.fit(
            transformed_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(transformed_validation, y_validation)],
            verbose=False,
        )
    else:
        estimator.fit(
            transformed_train,
            y_train,
            sample_weight=sample_weight_train,
        )

    return ModelArtifact(
        name=model_name,
        preprocessor=preprocessor,
        estimator=estimator,
        feature_columns=list(feature_columns),
        selected_parameters=dict(parameters),
    )


def _merge_parameters(
    baseline: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(baseline))
    merged.update(dict(override))
    return merged


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: Sequence[int],
    X_validation: pd.DataFrame,
    y_validation: Sequence[int],
    *,
    feature_columns: Sequence[str],
    model_parameters: Mapping[str, Any],
    random_seed: int,
    tune: bool = False,
    tuning_grid: Mapping[str, Sequence[Any]] | None = None,
    sample_weight_train: Sequence[float] | None = None,
    sample_weight_validation: Sequence[float] | None = None,
    selection_healthy_far_target: float | None = None,
    selection_healthy_far_penalty: float = 0.0,
) -> TrainingResult:
    """Train one model, optionally selecting a small grid on validation Macro-F1."""

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model {model_name!r}; choose from {SUPPORTED_MODELS}."
        )

    y_train_array = np.asarray(y_train, dtype=int)
    y_validation_array = np.asarray(y_validation, dtype=int)
    train_weight_array = (
        None
        if sample_weight_train is None
        else np.asarray(sample_weight_train, dtype=float)
    )
    validation_weight_array = (
        None
        if sample_weight_validation is None
        else np.asarray(sample_weight_validation, dtype=float)
    )
    if train_weight_array is not None and len(train_weight_array) != len(
        y_train_array
    ):
        raise ValueError("sample_weight_train length must match y_train.")
    if validation_weight_array is not None and len(
        validation_weight_array
    ) != len(y_validation_array):
        raise ValueError(
            "sample_weight_validation length must match y_validation."
        )
    start = time.perf_counter()
    trial_rows: list[dict[str, Any]] = []

    candidate_overrides: list[dict[str, Any]]
    if tune and tuning_grid:
        candidate_overrides = list(ParameterGrid(dict(tuning_grid)))
    else:
        candidate_overrides = [{}]

    best_artifact: ModelArtifact | None = None
    best_score = -np.inf
    for trial_index, override in enumerate(candidate_overrides, start=1):
        parameters = _merge_parameters(model_parameters, override)
        artifact = _fit_once(
            model_name,
            parameters,
            X_train,
            y_train_array,
            X_validation,
            y_validation_array,
            random_seed=random_seed,
            feature_columns=feature_columns,
            sample_weight_train=train_weight_array,
        )
        prediction = artifact.predict(X_validation)
        macro_f1 = float(
            f1_score(
                y_validation_array,
                prediction,
                average="macro",
                zero_division=0,
                sample_weight=validation_weight_array,
            )
        )
        healthy = y_validation_array == 0
        if np.any(healthy):
            healthy_weights = (
                None
                if validation_weight_array is None
                else validation_weight_array[healthy]
            )
            healthy_far = float(
                np.average(
                    prediction[healthy] != 0,
                    weights=healthy_weights,
                )
            )
        else:
            healthy_far = float("nan")
        score = macro_f1
        if (
            selection_healthy_far_target is not None
            and np.isfinite(healthy_far)
        ):
            score -= selection_healthy_far_penalty * max(
                0.0,
                healthy_far - selection_healthy_far_target,
            )
        trial_rows.append(
            {
                "Model": model_name,
                "Trial": trial_index,
                "ValidationMacroF1": macro_f1,
                "ValidationHealthyFAR": healthy_far,
                "SelectionScore": score,
                "Parameters": repr(parameters),
            }
        )
        if score > best_score:
            best_score = score
            best_artifact = artifact

    if best_artifact is None:
        raise RuntimeError(f"No {model_name} training candidate completed.")

    return TrainingResult(
        artifact=best_artifact,
        training_seconds=time.perf_counter() - start,
        tuning_trials=pd.DataFrame(trial_rows),
    )
