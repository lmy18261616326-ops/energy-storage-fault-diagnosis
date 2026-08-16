"""Evaluate and package the v06 direct switch-resistance specialist."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = PROJECT_ROOT / "ML"
SIM_ROOT = (
    PROJECT_ROOT
    / "simulink"
    / "experiments"
    / "sensor_bias"
    / "scripts"
    / "dataset_output_v16"
)
PILOT_ROOT = SIM_ROOT / "switch_observability_pilot"
VALIDATION_ROOT = SIM_ROOT / "switch_observability_validation"
OUTPUT_ROOT = ML_ROOT / "results" / "switch_observability_specialist"
MODEL_ROOT = ML_ROOT / "models" / "switch_observability_specialist_v16"
FROZEN_RON_THRESHOLD_OHM = 0.0105

sys.path.insert(0, str(ML_ROOT / "src"))
from energy_fault_ml.numpy_cnn import NumpyConv1DClassifier  # noqa: E402
from energy_fault_ml.switch_resistance import SwitchResistanceSpecialist  # noqa: E402

FEATURES = [
    "ActiveRonMedian",
    "ActiveRonMean",
    "ActiveRonMin",
    "ActiveRonMax",
    "ActiveConductionMedian",
    "ActiveConductionMean",
    "OtherRonMedian",
    "ModeCommand",
]


def load_split(root: Path) -> pd.DataFrame:
    specs = [
        ("mode1_health", "S1", "S2", 0),
        ("mode1_s1_high_r", "S1", "S2", 1),
        ("mode2_health", "S2", "S1", 0),
        ("mode2_s2_high_r", "S2", "S1", 1),
    ]
    parts: list[pd.DataFrame] = []
    for folder, active, other, target in specs:
        path = root / folder / "combined" / "feature_dataset.csv"
        frame = pd.read_csv(path)
        frame = frame.loc[
            (frame["WindowStart"] >= 0.4)
            & (frame["IsTrainingEligible"] == 1)
        ].copy()
        frame["ActiveRonMedian"] = frame[f"{active}_ron_estimateMedian"]
        frame["ActiveRonMean"] = frame[f"{active}_ron_estimateMean"]
        frame["ActiveRonMin"] = frame[f"{active}_ron_estimateMin"]
        frame["ActiveRonMax"] = frame[f"{active}_ron_estimateMax"]
        frame["ActiveConductionMedian"] = frame[
            f"{active}_conduction_ratioMedian"
        ]
        frame["ActiveConductionMean"] = frame[
            f"{active}_conduction_ratioMean"
        ]
        frame["OtherRonMedian"] = frame[f"{other}_ron_estimateMedian"]
        frame["Target"] = target
        frame["Scenario"] = folder
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    if result[FEATURES].isna().any().any():
        missing = result[FEATURES].isna().sum()
        raise ValueError(f"Specialist features contain NaN:\n{missing}")
    return result


def model_factories() -> dict[str, object]:
    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=816),
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=240,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=816,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=240,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=816,
            n_jobs=-1,
        ),
        "knn": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=9, weights="distance"),
        ),
        "mlp": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                alpha=1e-3,
                max_iter=500,
                early_stopping=True,
                random_state=816,
            ),
        ),
        "1d_cnn": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            NumpyConv1DClassifier(
                n_filters=8,
                kernel_size=3,
                stride=1,
                pool_segments=2,
                max_epochs=80,
                batch_size=128,
                n_iter_no_change=10,
                random_state=816,
            ),
        ),
    }


def metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    prediction = (score >= 0.5).astype(int)
    healthy = y_true == 0
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "macro_f1": f1_score(y_true, prediction, average="macro"),
        "fault_recall": recall_score(y_true, prediction, pos_label=1),
        "healthy_false_alarm_rate": float(prediction[healthy].mean()),
    }


def aggregate_runs(frame: pd.DataFrame, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    work = frame[["RunID", "Target"]].copy()
    work["Score"] = score
    grouped = work.groupby("RunID", sort=False).agg(
        Target=("Target", "first"), Score=("Score", "median")
    )
    return grouped["Target"].to_numpy(int), grouped["Score"].to_numpy(float)


def evaluate_models(pilot: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    x = pilot[FEATURES].to_numpy(float)
    y = pilot["Target"].to_numpy(int)
    groups = pilot["RunID"].astype(str).to_numpy()
    xv = validation[FEATURES].to_numpy(float)
    yv = validation["Target"].to_numpy(int)
    folds = GroupKFold(n_splits=4)
    rows: list[dict[str, object]] = []

    physics_oof = (
        pilot["ActiveRonMedian"].to_numpy(float) >= FROZEN_RON_THRESHOLD_OHM
    ).astype(float)
    physics_validation = (
        validation["ActiveRonMedian"].to_numpy(float)
        >= FROZEN_RON_THRESHOLD_OHM
    ).astype(float)
    for split, frame, score in [
        ("pilot_frozen_threshold", pilot, physics_oof),
        ("independent_validation", validation, physics_validation),
    ]:
        yy = frame["Target"].to_numpy(int)
        window = metrics(yy, score)
        run_y, run_score = aggregate_runs(frame, score)
        run = metrics(run_y, run_score)
        rows.append(
            {"model": "physics_threshold", "split": split, **window,
             **{f"run_{key}": value for key, value in run.items()},
             "train_seconds": 0.0}
        )

    for name, estimator in model_factories().items():
        oof = np.zeros(len(pilot), dtype=float)
        start = time.perf_counter()
        for train_idx, test_idx in folds.split(x, y, groups):
            estimator.fit(x[train_idx], y[train_idx])
            oof[test_idx] = estimator.predict_proba(x[test_idx])[:, 1]
        elapsed = time.perf_counter() - start
        window = metrics(y, oof)
        run_y, run_score = aggregate_runs(pilot, oof)
        run = metrics(run_y, run_score)
        rows.append(
            {"model": name, "split": "pilot_group_oof", **window,
             **{f"run_{key}": value for key, value in run.items()},
             "train_seconds": elapsed}
        )
        estimator.fit(x, y)
        score = estimator.predict_proba(xv)[:, 1]
        window = metrics(yv, score)
        run_y, run_score = aggregate_runs(validation, score)
        run = metrics(run_y, run_score)
        rows.append(
            {"model": name, "split": "independent_validation", **window,
             **{f"run_{key}": value for key, value in run.items()},
             "train_seconds": elapsed}
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    pilot = load_split(PILOT_ROOT)
    validation = load_split(VALIDATION_ROOT)
    results = evaluate_models(pilot, validation)
    results.to_csv(OUTPUT_ROOT / "model_comparison.csv", index=False)

    healthy_max = float(pilot.loc[pilot.Target == 0, "ActiveRonMedian"].max())
    fault_min = float(pilot.loc[pilot.Target == 1, "ActiveRonMedian"].min())
    validation_health_max = float(
        validation.loc[validation.Target == 0, "ActiveRonMedian"].max()
    )
    validation_fault_min = float(
        validation.loc[validation.Target == 1, "ActiveRonMedian"].min()
    )
    summary = {
        "model": "direct_switch_resistance_threshold",
        "threshold_ohm": FROZEN_RON_THRESHOLD_OHM,
        "pilot_runs": int(pilot.RunID.nunique()),
        "pilot_windows": len(pilot),
        "pilot_healthy_max_ohm": healthy_max,
        "pilot_fault_min_ohm": fault_min,
        "validation_runs": int(validation.RunID.nunique()),
        "validation_windows": len(validation),
        "validation_healthy_max_ohm": validation_health_max,
        "validation_fault_min_ohm": validation_fault_min,
        "required_model": "main_model_fd_v06_switchobservability",
        "required_measurements": [
            "S1 device current and voltage at 1 us",
            "S2 device current and voltage at 1 us",
        ],
        "decision": "high_resistance if active-switch median |V/I| >= threshold",
    }
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    specialist = SwitchResistanceSpecialist(
        threshold_ohm=FROZEN_RON_THRESHOLD_OHM,
        required_model="main_model_fd_v06_switchobservability",
    )
    joblib.dump(
        {"model": specialist, "summary": summary, "feature_names": FEATURES},
        MODEL_ROOT / "switch_resistance_specialist.joblib",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
