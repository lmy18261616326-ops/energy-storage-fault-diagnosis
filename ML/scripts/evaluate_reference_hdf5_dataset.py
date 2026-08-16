#!/usr/bin/env python3
"""Evaluate the synthetic reference HDF5 against frozen and retrained models.

The script deliberately excludes generator truth channels from all learned
features.  It reports three distinct questions:

1. Can the frozen five-class active-scope artifact transfer without tuning?
2. How well do common models perform when retrained with grouped OOF splits?
3. Can pre-fault-only data predict the future label (a synthetic leakage audit)?
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = ROOT / "ML"
SRC = ML_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_fault_ml.event_features import (  # noqa: E402
    add_physics_normalized_features,
    build_event_dataset,
)
from energy_fault_ml.numpy_cnn import NumpyConv1DClassifier  # noqa: E402


DATA_ROOT = ROOT / "output" / "reference_experiment_dataset_2026-08-04"
H5_PATH = DATA_ROOT / "bidirectional_dcdc_reference_raw.h5"
MANIFEST_PATH = DATA_ROOT / "run_manifest.csv"
MODEL_PATH = ML_ROOT / "models" / "final_active_scope_v14" / "active_scope_fault_model.joblib"
OUTPUT = ML_ROOT / "results" / "reference_hdf5_model_evaluation_v18"

SEED = 20260804
LABEL_NAMES = {
    0: "healthy",
    1: "vbus_bias",
    2: "il_bias",
    3: "S1_open",
    4: "S2_open",
    5: "high_resistance",
}
GROUP_TO_LABEL = {value: key for key, value in LABEL_NAMES.items()}

CONTROLLER_RAW = (
    "vbus_ref_V",
    "vbus_meas_V",
    "vbat_meas_V",
    "il_ref_A",
    "il_meas_A",
    "ibat_meas_A",
    "load_current_A",
    "source_current_A",
    "soc_pct",
    "duty_cmd",
    "s1_gate_cmd",
    "s2_gate_cmd",
)

DEVICE_RAW = (
    "s1_device_voltage_V",
    "s1_device_current_A",
    "s2_device_voltage_V",
    "s2_device_current_A",
)

EXCLUDED_TRUTH_CHANNELS = (
    "vbus_true_V",
    "vbat_true_V",
    "il_true_A",
    "ibat_true_A",
    "s1_gate_actual",
    "s2_gate_actual",
    "s1_equiv_active_current_A",
    "s2_equiv_active_current_A",
    "s1_on_resistance_ohm",
    "s2_on_resistance_ohm",
    "vbus_bias_actual_V",
    "il_bias_actual_A",
    "fault_trigger",
    "fault_active",
    "fault_effective",
    "fault_code",
)


def decode_string_matrix(dataset: h5py.Dataset) -> list[str]:
    values = dataset[()]
    if values.ndim == 3:
        values = values[:, :, 0]
    return [bytes(row).rstrip(b"\x00 ").decode("utf-8") for row in values]


def signal_map(dataset: h5py.Dataset, names: list[str]) -> dict[str, np.ndarray]:
    values = np.asarray(dataset[()], dtype=np.float64)
    if values.shape[0] == len(names):
        return {name: values[index] for index, name in enumerate(names)}
    if values.shape[1] == len(names):
        return {name: values[:, index] for index, name in enumerate(names)}
    raise ValueError(f"Unexpected HDF5 signal matrix shape: {values.shape}")


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def safe_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def window_series_features(
    channels: dict[str, np.ndarray],
    sample_count: int,
    *,
    window_samples: int = 1000,
    step_samples: int = 500,
) -> dict[str, float]:
    """Create phase-robust run features from fixed sliding windows."""

    starts = list(range(0, sample_count - window_samples + 1, step_samples))
    window_stats = ("mean", "std", "rms", "min", "max")
    event_stats = ("mean", "std", "median", "min", "max", "first", "last", "q10", "q90", "delta")
    features: dict[str, float] = {}
    for channel_name, values in channels.items():
        per_window = {name: [] for name in window_stats}
        for start in starts:
            part = values[start : start + window_samples]
            finite = part[np.isfinite(part)]
            if finite.size == 0:
                finite = np.asarray([0.0])
            per_window["mean"].append(float(np.mean(finite)))
            per_window["std"].append(safe_std(finite))
            per_window["rms"].append(rms(finite))
            per_window["min"].append(float(np.min(finite)))
            per_window["max"].append(float(np.max(finite)))
        for window_stat, raw_values in per_window.items():
            series = np.asarray(raw_values, dtype=float)
            values_by_stat = {
                "mean": float(np.mean(series)),
                "std": safe_std(series),
                "median": float(np.median(series)),
                "min": float(np.min(series)),
                "max": float(np.max(series)),
                "first": float(series[0]),
                "last": float(series[-1]),
                "q10": float(np.quantile(series, 0.10)),
                "q90": float(np.quantile(series, 0.90)),
                "delta": float(series[-1] - series[0]),
            }
            for event_stat in event_stats:
                features[f"{channel_name}__{window_stat}__{event_stat}"] = values_by_stat[event_stat]
    return features


def derived_channels(signals: dict[str, np.ndarray], include_devices: bool) -> dict[str, np.ndarray]:
    vbus = signals["vbus_meas_V"]
    il = signals["il_meas_A"]
    ibat = signals["ibat_meas_A"]
    result = {name: signals[name] for name in CONTROLLER_RAW}
    result.update(
        {
            "vbus_error_V": signals["vbus_ref_V"] - vbus,
            "il_error_A": signals["il_ref_A"] - il,
            "current_pair_residual_A": il - ibat,
            "pbat_meas_W": signals["vbat_meas_V"] * ibat,
            "pbus_source_W": vbus * signals["source_current_A"],
            "pload_meas_W": vbus * signals["load_current_A"],
            "gate_duty_difference": signals["s1_gate_cmd"] - signals["s2_gate_cmd"],
        }
    )
    if include_devices:
        result.update({name: signals[name] for name in DEVICE_RAW})
        for switch in ("s1", "s2"):
            voltage = np.abs(signals[f"{switch}_device_voltage_V"])
            current = np.abs(signals[f"{switch}_device_current_A"])
            estimate = np.full_like(current, np.nan, dtype=float)
            valid = current >= 0.5
            estimate[valid] = voltage[valid] / current[valid]
            result[f"{switch}_ron_estimate_ohm"] = estimate
    return result


def downsample_sequence(
    signals: dict[str, np.ndarray],
    names: tuple[str, ...],
    sample_count: int,
    *,
    block: int = 100,
) -> np.ndarray:
    usable = sample_count - (sample_count % block)
    channels = []
    for name in names:
        values = signals[name][:usable].reshape(-1, block)
        channels.append(values.mean(axis=1))
    return np.concatenate(channels).astype(np.float64)


def coarse_fault_name(row: pd.Series) -> str:
    group = row["FaultGroup"]
    subtype = row["FaultSubtype"]
    if group == "healthy":
        return "healthy"
    if group == "vbus_bias":
        return "vbus_sensor_bias"
    if group == "il_bias":
        return "inductor_current_sensor_bias"
    if group in {"S1_open", "S2_open"}:
        switch = group[:2]
        suffix = "intermittent" if subtype == "intermittent_open" else subtype
        return f"switch_{switch}_{suffix}"
    if group == "high_resistance":
        switch = str(row["FaultLocation"])
        return f"switch_{switch}_high_resistance"
    raise ValueError(group)


def old_window_features(signals: dict[str, np.ndarray], start: int, stop: int, dt: float) -> dict[str, float]:
    sl = slice(start, stop)
    il = signals["il_meas_A"][sl]
    ibat = signals["ibat_meas_A"][sl]
    vbus = signals["vbus_meas_V"][sl]
    vbat = signals["vbat_meas_V"][sl]
    iload = signals["load_current_A"][sl]
    iref = signals["il_ref_A"][sl]
    vref = signals["vbus_ref_V"][sl]
    duty = signals["duty_cmd"][sl]
    current_error = iref - il
    voltage_error = vref - vbus
    current_pair = il - ibat
    pbat = vbat * ibat
    psource = vbus * signals["source_current_A"][sl]
    pload = vbus * iload
    residual = psource + pbat - pload

    def mean_std_rms(prefix: str, values: np.ndarray, target: dict[str, float]) -> None:
        target[f"{prefix}Mean"] = float(np.mean(values))
        target[f"{prefix}Std"] = safe_std(values)
        target[f"{prefix}RMS"] = rms(values)

    row: dict[str, float] = {}
    mean_std_rms("IL_meas", il, row)
    mean_std_rms("Ibat_meas", ibat, row)
    mean_std_rms("Vbus_meas", vbus, row)
    row["Vbat_measMean"] = float(np.mean(vbat))
    row["Vbat_measStd"] = safe_std(vbat)
    mean_std_rms("Iload_meas", iload, row)
    row["Iload_measDiffRMS"] = rms(np.diff(iload))
    row["SOC_estMean"] = float(np.mean(signals["soc_pct"][sl]))
    row["SOC_estSlope"] = float(np.polyfit(np.arange(stop - start) * dt, signals["soc_pct"][sl], 1)[0])
    row["IrefMean"] = float(np.mean(iref))
    row["VbusRefMean"] = float(np.mean(vref))
    mean_std_rms("CurrentError", current_error, row)
    mean_std_rms("VoltageError", voltage_error, row)
    row["DutyRawMean"] = float(np.mean(duty))
    row["DutyRawStd"] = safe_std(duty)
    row["DutyAppliedMean"] = row["DutyRawMean"]
    row["DutyAppliedStd"] = row["DutyRawStd"]
    row["PIiOutMean"] = np.nan
    row["PIiOutStd"] = np.nan
    row["PIvOutMean"] = np.nan
    row["PIvOutStd"] = np.nan
    row["Psource_measMean"] = float(np.mean(psource))
    row["Psource_measStd"] = safe_std(psource)
    row["Pload_measMean"] = float(np.mean(pload))
    row["Pload_measStd"] = safe_std(pload)
    row["Pstored_measMean"] = float(np.mean(psource - pload))
    row["PowerBalanceResidualMean"] = float(np.mean(residual))
    row["PowerBalanceResidualRMS"] = rms(residual)
    row["CurrentPairResidualMean"] = float(np.mean(current_pair))
    row["CurrentPairResidualStd"] = safe_std(current_pair)
    row["CurrentPairResidualRMS"] = rms(current_pair)
    row["CurrentPairResidualMAE"] = float(np.mean(np.abs(current_pair)))
    row["BalancedPowerResidualMean"] = float(np.mean(residual))
    row["BalancedPowerResidualStd"] = safe_std(residual)
    row["BalancedPowerResidualRMS"] = rms(residual)
    row["PbatMean"] = float(np.mean(pbat))
    row["PbatStd"] = safe_std(pbat)
    row["PbusMean"] = float(np.mean(psource))
    row["PbusStd"] = safe_std(psource)
    row["SatRatio"] = float(np.mean((duty <= 0.0800001) | (duty >= 0.9199999)))
    row["EnableRatio"] = 1.0
    row["S1GateDutyRatio"] = float(np.mean(signals["s1_gate_cmd"][sl]))
    row["S2GateDutyRatio"] = float(np.mean(signals["s2_gate_cmd"][sl]))
    row["DutyLimitResidualMean"] = 0.0
    row["DutyLimitResidualRMS"] = 0.0
    row["DutySatRatio"] = row["SatRatio"]
    return row


def frozen_adapter_windows(signals: dict[str, np.ndarray], manifest_row: pd.Series, dt: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    window_samples = int(round(0.010 / dt))
    step_samples = int(round(0.005 / dt))
    for window_id, start in enumerate(range(0, len(signals["vbus_meas_V"]) - window_samples + 1, step_samples)):
        values: dict[str, object] = {
            "RunID": manifest_row["RunID"],
            "OperatingPointID": manifest_row["OperatingPointID"],
            "WindowID": window_id,
            "WindowStart": start * dt,
            "FaultName": manifest_row["FaultName"],
            "ModeCommand": manifest_row["ModeCommand"],
            "SOCInit": manifest_row["SOCInit_pct"],
            "IrefLevel": manifest_row["Iref_A"],
            "VbusRefSetting": manifest_row["VbusRef_V"],
            "Rload": manifest_row["VbusRef_V"] ** 2 / manifest_row["Pload_W"],
            "Pload": manifest_row["Pload_W"],
            "Rbat": 0.50 * (1 - 0.003 * (manifest_row["Temperature_C"] - 25)),
            "Cbus": np.nan,
            "CbusESR": np.nan,
        }
        values.update(old_window_features(signals, start, start + window_samples, dt))
        rows.append(values)
    return rows


def ron_window_score(signals: dict[str, np.ndarray], start: int, stop: int) -> float:
    scores = []
    for switch in ("s1", "s2"):
        voltage = np.abs(signals[f"{switch}_device_voltage_V"][start:stop])
        current = np.abs(signals[f"{switch}_device_current_A"][start:stop])
        valid = current >= 0.5
        scores.append(float(np.median(voltage[valid] / current[valid])) if np.any(valid) else np.nan)
    return float(np.nanmax(scores))


def make_classical_model(name: str, feature_count: int, seed: int) -> Pipeline:
    prep: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if name in {"logistic_regression", "knn", "mlp"}:
        prep.append(("variance", VarianceThreshold()))
        k = min(128 if name != "knn" else 64, feature_count)
        prep.extend([("select", SelectKBest(f_classif, k=k)), ("scale", StandardScaler())])
    if name == "logistic_regression":
        model = LogisticRegression(C=0.25, class_weight="balanced", max_iter=4000, random_state=seed)
    elif name == "random_forest":
        model = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    elif name == "extra_trees":
        model = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=seed)
    elif name == "xgboost":
        model = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, min_child_weight=2, subsample=0.85, colsample_bytree=0.70, reg_lambda=5.0, objective="multi:softprob", num_class=6, eval_metric="mlogloss", n_jobs=8, random_state=seed)
    elif name == "knn":
        model = KNeighborsClassifier(n_neighbors=7, weights="distance", metric="manhattan", n_jobs=-1)
    elif name == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(96, 48), alpha=1e-3, batch_size=32, learning_rate_init=1e-3, max_iter=500, early_stopping=True, validation_fraction=0.15, n_iter_no_change=25, random_state=seed)
    else:
        raise ValueError(name)
    return Pipeline([*prep, ("model", model)])


def make_cnn(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "model",
                NumpyConv1DClassifier(
                    n_filters=16,
                    kernel_size=9,
                    stride=5,
                    pool_segments=16,
                    max_epochs=180,
                    batch_size=32,
                    n_iter_no_change=20,
                    random_state=seed,
                ),
            ),
        ]
    )


def exact_interval(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    if trials == 0:
        return np.nan, np.nan
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return low, high


def bootstrap_macro_f1(labels: np.ndarray, prediction: np.ndarray, groups: np.ndarray, seed: int) -> tuple[float, float]:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(1500):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        values.append(f1_score(labels[indices], prediction[indices], labels=range(6), average="macro", zero_division=0))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def grouped_oof(
    frame: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    *,
    models: tuple[str, ...],
    audit_type: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    labels = frame["Label"].to_numpy(dtype=int)
    groups = frame["OperatingPointID"].to_numpy(dtype=str)
    splitter = GroupKFold(n_splits=6)
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for representation, matrix in matrices.items():
        for model_index, model_name in enumerate(models):
            oof = np.full(len(frame), -1, dtype=int)
            probabilities = np.full((len(frame), 6), np.nan)
            train_seconds = 0.0
            for fold, (train_idx, test_idx) in enumerate(splitter.split(matrix, labels, groups), start=1):
                train_classes = set(labels[train_idx])
                test_classes = set(labels[test_idx])
                if train_classes != set(range(6)) or test_classes != set(range(6)):
                    raise ValueError(f"Fold {fold} class coverage mismatch: train={train_classes}, test={test_classes}")
                model = make_cnn(SEED + fold + model_index * 100) if model_name == "cnn_1d" else make_classical_model(model_name, matrix.shape[1], SEED + fold + model_index * 100)
                weights = compute_sample_weight("balanced", labels[train_idx])
                fit_params = inspect.signature(model.named_steps["model"].fit).parameters
                start = time.perf_counter()
                if "sample_weight" in fit_params:
                    model.fit(matrix[train_idx], labels[train_idx], model__sample_weight=weights)
                else:
                    model.fit(matrix[train_idx], labels[train_idx])
                train_seconds += time.perf_counter() - start
                probability = model.predict_proba(matrix[test_idx])
                learned = np.asarray(model.named_steps["model"].classes_, dtype=int)
                aligned = np.zeros((len(test_idx), 6), dtype=float)
                aligned[:, learned] = probability
                prediction = np.argmax(aligned, axis=1)
                oof[test_idx] = prediction
                probabilities[test_idx] = aligned
                fold_rows.append(
                    {
                        "audit_type": audit_type,
                        "representation": representation,
                        "model": model_name,
                        "fold": fold,
                        "test_runs": len(test_idx),
                        "test_operating_points": "|".join(sorted(np.unique(groups[test_idx]))),
                        "macro_f1": f1_score(labels[test_idx], prediction, labels=range(6), average="macro", zero_division=0),
                        "accuracy": accuracy_score(labels[test_idx], prediction),
                        "healthy_far": float(np.mean(prediction[labels[test_idx] == 0] != 0)),
                    }
                )
            if np.any(oof < 0) or np.isnan(probabilities).any():
                raise ValueError("OOF predictions are incomplete")
            correct = int(np.sum(oof == labels))
            accuracy_low, accuracy_high = exact_interval(correct, len(labels))
            macro_low, macro_high = bootstrap_macro_f1(labels, oof, groups, SEED + model_index)
            healthy = labels == 0
            far_count = int(np.sum(oof[healthy] != 0))
            far_low, far_high = exact_interval(far_count, int(np.sum(healthy)))
            summary_rows.append(
                {
                    "audit_type": audit_type,
                    "representation": representation,
                    "model": model_name,
                    "runs": len(labels),
                    "operating_points": len(np.unique(groups)),
                    "feature_count": matrix.shape[1],
                    "accuracy": correct / len(labels),
                    "accuracy_ci95_low": accuracy_low,
                    "accuracy_ci95_high": accuracy_high,
                    "balanced_accuracy": balanced_accuracy_score(labels, oof),
                    "macro_f1": f1_score(labels, oof, labels=range(6), average="macro", zero_division=0),
                    "macro_f1_cluster_ci95_low": macro_low,
                    "macro_f1_cluster_ci95_high": macro_high,
                    "healthy_far": far_count / int(np.sum(healthy)),
                    "healthy_far_ci95_low": far_low,
                    "healthy_far_ci95_high": far_high,
                    "high_resistance_recall": float(np.mean(oof[labels == 5] == 5)),
                    "training_seconds_total": train_seconds,
                }
            )
            for index, row in frame.reset_index(drop=True).iterrows():
                item = {
                    "audit_type": audit_type,
                    "representation": representation,
                    "model": model_name,
                    "RunID": row["RunID"],
                    "OperatingPointID": row["OperatingPointID"],
                    "true_label": int(labels[index]),
                    "predicted_label": int(oof[index]),
                    "correct": int(oof[index] == labels[index]),
                }
                for label in range(6):
                    item[f"probability_{label}"] = probabilities[index, label]
                prediction_rows.append(item)
    return summary_rows, fold_rows, prediction_rows


def metric_bundle(labels: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(labels, prediction, labels=range(5), zero_division=0)
    healthy = labels == 0
    return {
        "runs": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, labels=range(5), average="macro", zero_division=0)),
        "healthy_far": float(np.mean(prediction[healthy] != 0)),
        "per_class": {
            LABEL_NAMES[label]: {
                "precision": float(precision[label]),
                "recall": float(recall[label]),
                "f1": float(f1[label]),
                "support": int(support[label]),
            }
            for label in range(5)
        },
        "confusion_matrix": confusion_matrix(labels, prediction, labels=range(5)).tolist(),
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest["RunID"] = manifest["RunID"].astype(str)
    manifest["OperatingPointID"] = (
        manifest["Mode"].astype(str)
        + "_I"
        + manifest["CurrentLevel_A"].astype(str)
        + "_T"
        + manifest["Temperature_C"].astype(str)
    )
    manifest["Label"] = manifest["FaultGroup"].map(GROUP_TO_LABEL).astype(int)
    manifest["ModeCommand"] = np.where(manifest["Mode"].eq("discharge"), 1, 2)
    manifest["FaultName"] = manifest.apply(coarse_fault_name, axis=1)

    controller_features: list[dict[str, float]] = []
    device_features: list[dict[str, float]] = []
    controller_pre_features: list[dict[str, float]] = []
    controller_sequences: list[np.ndarray] = []
    device_sequences: list[np.ndarray] = []
    controller_pre_sequences: list[np.ndarray] = []
    adapter_rows: list[dict[str, object]] = []
    highr_rows: list[dict[str, object]] = []
    nonfinite_cells = 0

    with h5py.File(H5_PATH, "r") as h5:
        names = decode_string_matrix(h5["meta/signal_names"])
        time_axis = np.asarray(h5["time_s"][()]).reshape(-1).astype(float)
        dt = float(np.median(np.diff(time_axis)))
        h5_runs = set(h5["runs"].keys())
        if h5_runs != set(manifest["RunID"]):
            raise ValueError("Manifest and HDF5 RunID sets differ")
        if set(EXCLUDED_TRUTH_CHANNELS).difference(names):
            raise ValueError("Expected truth-channel inventory differs from HDF5")
        for _, row in manifest.iterrows():
            signals = signal_map(h5[f"runs/{row['RunID']}/signals"], names)
            nonfinite_cells += sum(int(np.sum(~np.isfinite(values))) for values in signals.values())
            controller = derived_channels(signals, include_devices=False)
            device = derived_channels(signals, include_devices=True)
            controller_features.append(window_series_features(controller, len(time_axis)))
            device_features.append(window_series_features(device, len(time_axis)))
            pre_count = int(round(0.006 / dt))
            controller_pre = {name: values[:pre_count] for name, values in controller.items()}
            controller_pre_features.append(window_series_features(controller_pre, pre_count))
            controller_sequences.append(downsample_sequence(signals, CONTROLLER_RAW, len(time_axis)))
            device_sequences.append(downsample_sequence(signals, CONTROLLER_RAW + DEVICE_RAW, len(time_axis)))
            controller_pre_sequences.append(downsample_sequence(signals, CONTROLLER_RAW, pre_count))
            adapter_rows.extend(frozen_adapter_windows(signals, row, dt))

            if row["FaultGroup"] in {"healthy", "high_resistance"}:
                window_samples = int(round(0.010 / dt))
                step_samples = int(round(0.005 / dt))
                starts = list(range(0, len(time_axis) - window_samples + 1, step_samples))
                scores = [ron_window_score(signals, start, start + window_samples) for start in starts]
                positives = np.asarray(scores) >= 0.0105
                pairs = np.flatnonzero(positives[:-1] & positives[1:])
                detected = bool(len(pairs))
                confirmation_s = (starts[int(pairs[0]) + 1] + window_samples) * dt if detected else np.nan
                latency_ms = (
                    (confirmation_s - float(row["FaultStart_s"])) * 1000
                    if detected and row["FaultGroup"] == "high_resistance"
                    else np.nan
                )
                highr_rows.append(
                    {
                        "RunID": row["RunID"],
                        "FaultGroup": row["FaultGroup"],
                        "FaultLocation": row["FaultLocation"],
                        "Mode": row["Mode"],
                        "Temperature_C": row["Temperature_C"],
                        "CommandedResistance_mOhm": row["CommandedResistance_mOhm"],
                        "ActualResistance_mOhm": row["ActualResistance_mOhm"],
                        "score_window_1_mOhm": scores[0] * 1000,
                        "score_window_2_mOhm": scores[1] * 1000,
                        "score_window_3_mOhm": scores[2] * 1000,
                        "detected": int(detected),
                        "latency_ms": latency_ms,
                    }
                )

        root_origin = str(np.asarray(h5.attrs["data_origin"]).reshape(-1)[0])
        matrix_layout_declared = str(np.asarray(h5.attrs["matrix_layout"]).reshape(-1)[0])

    index_columns = ["RunID", "OperatingPointID", "Label", "FaultGroup", "FaultSubtype", "Mode", "Temperature_C", "CurrentLevel_A"]
    controller_frame = pd.concat([manifest[index_columns].reset_index(drop=True), pd.DataFrame(controller_features)], axis=1)
    device_frame = pd.concat([manifest[index_columns].reset_index(drop=True), pd.DataFrame(device_features)], axis=1)
    controller_pre_frame = pd.concat([manifest[index_columns].reset_index(drop=True), pd.DataFrame(controller_pre_features)], axis=1)
    controller_frame.to_csv(OUTPUT / "controller_engineered_features.csv", index=False)
    device_frame.to_csv(OUTPUT / "device_augmented_engineered_features.csv", index=False)

    matrices_full = {
        "controller_observable": controller_frame.drop(columns=index_columns).to_numpy(dtype=float),
        "device_augmented": device_frame.drop(columns=index_columns).to_numpy(dtype=float),
    }
    models = ("logistic_regression", "random_forest", "extra_trees", "xgboost", "knn", "mlp")
    summary_rows, fold_rows, prediction_rows = grouped_oof(manifest, matrices_full, models=models, audit_type="full_run")
    cnn_summary, cnn_folds, cnn_predictions = grouped_oof(
        manifest,
        {
            "controller_raw_sequence": np.vstack(controller_sequences),
            "device_raw_sequence": np.vstack(device_sequences),
        },
        models=("cnn_1d",),
        audit_type="full_run",
    )
    summary_rows.extend(cnn_summary)
    fold_rows.extend(cnn_folds)
    prediction_rows.extend(cnn_predictions)

    pre_summary, pre_folds, pre_predictions = grouped_oof(
        manifest,
        {"controller_pre_fault_0_6ms": controller_pre_frame.drop(columns=index_columns).to_numpy(dtype=float)},
        models=("logistic_regression", "extra_trees"),
        audit_type="pre_fault_negative_control",
    )
    pre_cnn_summary, pre_cnn_folds, pre_cnn_predictions = grouped_oof(
        manifest,
        {"controller_pre_fault_raw_sequence": np.vstack(controller_pre_sequences)},
        models=("cnn_1d",),
        audit_type="pre_fault_negative_control",
    )
    summary_rows.extend(pre_summary + pre_cnn_summary)
    fold_rows.extend(pre_folds + pre_cnn_folds)
    prediction_rows.extend(pre_predictions + pre_cnn_predictions)

    summary = pd.DataFrame(summary_rows)
    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary.to_csv(OUTPUT / "model_comparison.csv", index=False)
    folds.to_csv(OUTPUT / "fold_metrics.csv", index=False)
    predictions.to_csv(OUTPUT / "oof_predictions.csv", index=False)

    adapter_frame, adapter_features = build_event_dataset(pd.DataFrame(adapter_rows), include_context=False)
    frozen_model = joblib.load(MODEL_PATH)
    missing_model_features = sorted(set(frozen_model.feature_names).difference(adapter_frame.columns))
    if missing_model_features:
        raise ValueError(f"Frozen adapter missing features: {missing_model_features[:10]}")
    frozen_status = frozen_model.predict_with_status(adapter_frame, adapter_frame["ModeCommand"])
    frozen = adapter_frame[["RunID", "OperatingPointID", "FaultName", "FaultMechanism", "WindowFaultID", "ModeCommand"]].join(frozen_status)
    frozen.to_csv(OUTPUT / "frozen_model_predictions.csv", index=False)
    five_class = frozen[~frozen["FaultMechanism"].eq("high_resistance")].copy()
    full_metrics = metric_bundle(five_class["WindowFaultID"].to_numpy(dtype=int), five_class["PredictedClassID"].to_numpy(dtype=int))
    active = ~(
        (five_class["WindowFaultID"].eq(3) & five_class["ModeCommand"].ne(1))
        | (five_class["WindowFaultID"].eq(4) & five_class["ModeCommand"].ne(2))
    )
    active_part = five_class[active]
    active_metrics = metric_bundle(active_part["WindowFaultID"].to_numpy(dtype=int), active_part["PredictedClassID"].to_numpy(dtype=int))
    frozen_metrics = {
        "feature_adapter_coverage": {
            "expected_features": len(frozen_model.feature_names),
            "available_features": len(adapter_features),
            "all_nan_adapter_features": int(adapter_frame[list(frozen_model.feature_names)].isna().all(axis=0).sum()),
            "all_nan_feature_names": adapter_frame[list(frozen_model.feature_names)].columns[adapter_frame[list(frozen_model.feature_names)].isna().all(axis=0)].tolist(),
        },
        "full_five_class_including_inactive_switch_states": full_metrics,
        "declared_active_observable_scope": active_metrics,
        "primary_verifier_agreement": float(frozen["ModelsAgree"].mean()),
    }
    (OUTPUT / "frozen_model_metrics.json").write_text(json.dumps(frozen_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    highr = pd.DataFrame(highr_rows)
    highr.to_csv(OUTPUT / "high_r_threshold_predictions.csv", index=False)
    highr_fault = highr[highr["FaultGroup"].eq("high_resistance")]
    highr_summary_rows = []
    for severity, part in highr_fault.groupby("CommandedResistance_mOhm", sort=True):
        successes = int(part["detected"].sum())
        low, high = exact_interval(successes, len(part))
        highr_summary_rows.append(
            {
                "CommandedResistance_mOhm": severity,
                "runs": len(part),
                "detection_rate": successes / len(part),
                "detection_ci95_low": low,
                "detection_ci95_high": high,
                "median_latency_ms": float(part["latency_ms"].median()) if successes else np.nan,
                "p95_latency_ms": float(part["latency_ms"].quantile(0.95)) if successes else np.nan,
            }
        )
    healthy = highr[highr["FaultGroup"].eq("healthy")]
    guaranteed = highr_fault[highr_fault["CommandedResistance_mOhm"].ge(12)]
    highr_summary = pd.DataFrame(highr_summary_rows)
    highr_summary.to_csv(OUTPUT / "high_r_threshold_summary.csv", index=False)
    highr_metrics = {
        "frozen_threshold_mOhm": 10.5,
        "healthy_runs": len(healthy),
        "healthy_false_alarm_rate": float(healthy["detected"].mean()),
        "all_labeled_high_resistance_recall": float(highr_fault["detected"].mean()),
        "guaranteed_scope": "CommandedResistance_mOhm >= 12",
        "guaranteed_scope_runs": len(guaranteed),
        "guaranteed_scope_recall": float(guaranteed["detected"].mean()),
    }
    (OUTPUT / "high_r_threshold_metrics.json").write_text(json.dumps(highr_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    full_plot = summary[summary["audit_type"].eq("full_run")].copy()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    labels = [f"{row.model}\n{row.representation}" for row in full_plot.itertuples()]
    colors = ["#2f6f9f" if "device" not in row.representation else "#d97941" for row in full_plot.itertuples()]
    ax.bar(np.arange(len(full_plot)), full_plot["macro_f1"], color=colors)
    ax.set_xticks(np.arange(len(full_plot)), labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("6-class Group OOF Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "model_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(highr_summary["CommandedResistance_mOhm"], highr_summary["detection_rate"], marker="o")
    ax.axvline(10.5, color="#b44", linestyle="--", label="Frozen threshold 10.5 mΩ")
    ax.set_xlabel("Commanded high resistance (mΩ)")
    ax.set_ylabel("Run detection rate")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "high_r_detection_by_severity.png", dpi=180)
    plt.close(fig)

    integrity = {
        "dataset_origin_attribute": root_origin,
        "manifest_runs": len(manifest),
        "hdf5_runs": len(manifest),
        "samples_per_run": len(time_axis),
        "signal_count": len(names),
        "sample_interval_s": dt,
        "time_strictly_increasing": bool(np.all(np.diff(time_axis) > 0)),
        "nonfinite_signal_cells": nonfinite_cells,
        "matrix_layout_attribute": matrix_layout_declared,
        "python_observed_matrix_shape": [len(names), len(time_axis)],
        "truth_channels_excluded_from_models": list(EXCLUDED_TRUTH_CHANNELS),
        "class_counts": {LABEL_NAMES[int(key)]: int(value) for key, value in Counter(manifest["Label"]).items()},
        "operating_points": int(manifest["OperatingPointID"].nunique()),
        "random_seed_ranges_by_fault_group": {
            group: [int(part["RandomSeed"].min()), int(part["RandomSeed"].max())]
            for group, part in manifest.groupby("FaultGroup")
        },
    }
    (OUTPUT / "data_integrity.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")

    best_full = full_plot.sort_values("macro_f1", ascending=False).iloc[0]
    best_prefault = summary[summary["audit_type"].eq("pre_fault_negative_control")].sort_values("macro_f1", ascending=False).iloc[0]
    report = f"""# 合成参考HDF5模型迁移初测

## 结论

- 数据完整：{len(manifest)} Run、{len(time_axis):,} 点/Run、{len(names)} 通道、{manifest['OperatingPointID'].nunique()} 个工况；HDF5信号非有限值计数为 {nonfinite_cells}。
- 数据来源属性明确写为 `{root_origin}`，因此只能作为合成参考域，不能替代Simulink动态仿真或实测。
- 冻结旧主模型在其声明的主动可观测范围上：Macro-F1={active_metrics['macro_f1']:.4f}、Accuracy={active_metrics['accuracy']:.4f}、健康FAR={active_metrics['healthy_far']:.4f}。
- 新域6类分组OOF最佳为 `{best_full['model']}` / `{best_full['representation']}`：Macro-F1={best_full['macro_f1']:.4f}，Accuracy={best_full['accuracy']:.4f}，高阻召回={best_full['high_resistance_recall']:.4f}。
- 故障前0–6 ms负对照最佳为 `{best_prefault['model']}`：Macro-F1={best_prefault['macro_f1']:.4f}。若显著高于随机水平1/6，说明波形生成中的随机种子/相位已携带未来标签信息，完整波形成绩必须降级解释。
- 冻结10.5 mΩ物理阈值：全部标注高阻召回={highr_metrics['all_labeled_high_resistance_recall']:.4f}；在既定保证范围≥12 mΩ的召回={highr_metrics['guaranteed_scope_recall']:.4f}；健康FAR={highr_metrics['healthy_false_alarm_rate']:.4f}。

## 评估口径

- 训练模型严格排除真实值、实际门极、实际导通电阻、实际偏置、故障触发/活动/有效标志和FaultCode。
- OOF按 `Mode × CurrentLevel × Temperature` 的12个工况做6折GroupKFold，同一工况不会同时进入训练与测试。
- `controller_observable`只使用控制指令和常规测量；`device_augmented`额外加入S1/S2器件压降与电流。
- 1D-CNN使用0.1 ms降采样的原始时序；其他模型使用不依赖故障起点的固定滑窗统计。
- 冻结旧模型通过兼容转换补齐原660维事件特征，无法从新数据获得的PI内部量由旧模型训练中位数插补，未重新训练或回调。

## 主要风险

不同故障族的RandomSeed使用互不重叠的区间，且生成器把RandomSeed写入850 Hz波形相位。负对照就是为了量化这一捷径。下一版数据应让健康/各故障共享配对随机种子，或使噪声/相位种子与标签独立，再生成一套完全冻结盲测。
"""
    (OUTPUT / "evaluation_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Outputs: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
