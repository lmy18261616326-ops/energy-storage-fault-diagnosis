#!/usr/bin/env python3
"""Robustly optimize controller-observable models on the reference HDF5 domain.

The optimization deliberately treats the data as synthetic and provisional.
Seven prespecified iterations are compared with grouped OOF predictions.  An
outer 6-fold / inner 5-fold group scheme selects candidates without using the
outer fold labels.  A pre-fault negative control remains a release gate.
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = ROOT / "ML"
SCRIPT_DIR = ML_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_reference_hdf5_dataset import (  # noqa: E402
    CONTROLLER_RAW,
    GROUP_TO_LABEL,
    decode_string_matrix,
    derived_channels,
    safe_std,
    signal_map,
)


DATA_ROOT = ROOT / "output" / "reference_experiment_dataset_2026-08-04"
H5_PATH = DATA_ROOT / "bidirectional_dcdc_reference_raw.h5"
MANIFEST_PATH = DATA_ROOT / "run_manifest.csv"
BASELINE_FEATURES = ML_ROOT / "results" / "reference_hdf5_model_evaluation_v18" / "controller_engineered_features.csv"
OUTPUT = ML_ROOT / "results" / "reference_hdf5_model_optimization_v19"
MODEL_OUTPUT = ML_ROOT / "models" / "reference_hdf5_optimized_v19"
SEED = 20260806
LABELS = tuple(range(6))


@dataclass(frozen=True)
class Candidate:
    iteration: int
    name: str
    representation: str
    rationale: str


CANDIDATES = (
    Candidate(1, "baseline_extra_trees_full", "baseline_full", "Reproduce the existing 950-feature Extra Trees baseline."),
    Candidate(2, "extra_trees_delta", "delta_all", "Use phase-robust pre/post/delta statistics."),
    Candidate(3, "extra_trees_physics_compact", "delta_compact", "Reduce redundancy to physically interpretable residual channels."),
    Candidate(4, "random_forest_delta", "delta_all", "Test a more conservative bagged-tree boundary."),
    Candidate(5, "hist_gradient_delta", "delta_all", "Test regularized histogram boosting on the compact Run table."),
    Candidate(6, "xgboost_delta", "delta_all", "Test explicitly regularized boosted trees."),
    Candidate(7, "logistic_delta", "delta_all", "Test whether a low-capacity linear boundary is sufficient."),
)

COMPACT_CHANNELS = {
    "vbus_meas_V",
    "il_meas_A",
    "ibat_meas_A",
    "load_current_A",
    "source_current_A",
    "duty_cmd",
    "vbus_error_V",
    "il_error_A",
    "current_pair_residual_A",
    "pbat_meas_W",
    "pbus_source_W",
    "pload_meas_W",
    "gate_duty_difference",
}


def exact_interval(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    if trials == 0:
        return np.nan, np.nan
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return low, high


def summary_stats(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        finite = np.asarray([0.0])
    return {
        "mean": float(np.mean(finite)),
        "std": safe_std(finite),
        "rms": float(np.sqrt(np.mean(np.square(finite)))),
        "median": float(np.median(finite)),
        "q10": float(np.quantile(finite, 0.10)),
        "q90": float(np.quantile(finite, 0.90)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def differential_features(
    channels: dict[str, np.ndarray],
    dt: float,
    *,
    pre: tuple[float, float],
    post: tuple[float, float],
) -> dict[str, float]:
    pre_slice = slice(int(round(pre[0] / dt)), int(round(pre[1] / dt)))
    post_slice = slice(int(round(post[0] / dt)), int(round(post[1] / dt)))
    row: dict[str, float] = {}
    for channel, values in channels.items():
        before = summary_stats(values[pre_slice])
        after = summary_stats(values[post_slice])
        scale = max(abs(before["median"]), before["std"], 1e-6)
        for stat in before:
            row[f"{channel}__pre__{stat}"] = before[stat]
            row[f"{channel}__post__{stat}"] = after[stat]
            row[f"{channel}__delta__{stat}"] = after[stat] - before[stat]
            row[f"{channel}__relative__{stat}"] = (after[stat] - before[stat]) / scale
    return row


def make_model(candidate: Candidate, feature_count: int, seed: int) -> Pipeline:
    common: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if candidate.name == "logistic_delta":
        common.extend(
            [
                ("variance", VarianceThreshold()),
                ("select", SelectKBest(f_classif, k=min(160, feature_count))),
                ("scale", StandardScaler()),
            ]
        )
        model = LogisticRegression(C=0.15, class_weight="balanced", max_iter=5000, random_state=seed)
    elif candidate.name in {"baseline_extra_trees_full", "extra_trees_delta"}:
        model = ExtraTreesClassifier(
            n_estimators=450,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    elif candidate.name == "extra_trees_physics_compact":
        model = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=1,
            max_features=0.45,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    elif candidate.name == "random_forest_delta":
        model = RandomForestClassifier(
            n_estimators=450,
            min_samples_leaf=2,
            max_features=0.45,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    elif candidate.name == "hist_gradient_delta":
        model = HistGradientBoostingClassifier(
            learning_rate=0.055,
            max_iter=260,
            max_leaf_nodes=15,
            min_samples_leaf=8,
            l2_regularization=2.0,
            class_weight="balanced",
            random_state=seed,
        )
    elif candidate.name == "xgboost_delta":
        model = XGBClassifier(
            n_estimators=360,
            max_depth=3,
            learning_rate=0.035,
            min_child_weight=3,
            subsample=0.80,
            colsample_bytree=0.65,
            reg_alpha=0.15,
            reg_lambda=8.0,
            objective="multi:softprob",
            num_class=6,
            eval_metric="mlogloss",
            n_jobs=8,
            random_state=seed,
        )
    else:
        raise ValueError(candidate.name)
    return Pipeline([*common, ("model", model)])


def fit_model(model: Pipeline, matrix: np.ndarray, labels: np.ndarray) -> None:
    weights = compute_sample_weight("balanced", labels)
    params = inspect.signature(model.named_steps["model"].fit).parameters
    if "sample_weight" in params:
        model.fit(matrix, labels, model__sample_weight=weights)
    else:
        model.fit(matrix, labels)


def predict_aligned(model: Pipeline, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = model.predict_proba(matrix)
    classes = np.asarray(model.named_steps["model"].classes_, dtype=int)
    probabilities = np.zeros((len(matrix), len(LABELS)), dtype=float)
    probabilities[:, classes] = raw
    return np.argmax(probabilities, axis=1), probabilities


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    healthy = labels == 0
    high_r = labels == 5
    macro = float(f1_score(labels, prediction, labels=LABELS, average="macro", zero_division=0))
    far = float(np.mean(prediction[healthy] != 0))
    high_r_recall = float(np.mean(prediction[high_r] == 5))
    objective = macro - 0.50 * max(0.0, far - 0.05) - 0.30 * max(0.0, 0.90 - high_r_recall)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": macro,
        "healthy_far": far,
        "high_resistance_recall": high_r_recall,
        "selection_objective": objective,
    }


def grouped_candidate_oof(
    candidate: Candidate,
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], float]:
    prediction = np.full(len(labels), -1, dtype=int)
    probabilities = np.full((len(labels), len(LABELS)), np.nan)
    rows: list[dict[str, object]] = []
    training_seconds = 0.0
    splitter = GroupKFold(n_splits=6)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(matrix, labels, groups), start=1):
        model = make_model(candidate, matrix.shape[1], SEED + candidate.iteration * 100 + fold)
        start = time.perf_counter()
        fit_model(model, matrix[train_idx], labels[train_idx])
        training_seconds += time.perf_counter() - start
        fold_prediction, fold_probabilities = predict_aligned(model, matrix[test_idx])
        prediction[test_idx] = fold_prediction
        probabilities[test_idx] = fold_probabilities
        rows.append(
            {
                "iteration": candidate.iteration,
                "candidate": candidate.name,
                "fold": fold,
                "test_runs": len(test_idx),
                "test_groups": "|".join(sorted(np.unique(groups[test_idx]))),
                **metrics(labels[test_idx], fold_prediction),
            }
        )
    return prediction, probabilities, rows, training_seconds


def inner_score(
    candidate: Candidate,
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    outer_fold: int,
) -> float:
    prediction = np.full(len(labels), -1, dtype=int)
    splitter = GroupKFold(n_splits=5)
    for inner_fold, (train_idx, test_idx) in enumerate(splitter.split(matrix, labels, groups), start=1):
        model = make_model(candidate, matrix.shape[1], SEED + outer_fold * 1000 + candidate.iteration * 50 + inner_fold)
        fit_model(model, matrix[train_idx], labels[train_idx])
        prediction[test_idx], _ = predict_aligned(model, matrix[test_idx])
    return metrics(labels, prediction)["selection_objective"]


def negative_control_oof(matrix: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    candidate = Candidate(0, "extra_trees_physics_compact", "pre_fault_compact", "Leakage audit only")
    prediction = np.full(len(labels), -1, dtype=int)
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=6).split(matrix, labels, groups), start=1):
        model = make_model(candidate, matrix.shape[1], SEED + 9000 + fold)
        fit_model(model, matrix[train_idx], labels[train_idx])
        prediction[test_idx], _ = predict_aligned(model, matrix[test_idx])
    return prediction, metrics(labels, prediction)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
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
    labels = manifest["Label"].to_numpy(dtype=int)
    groups = manifest["OperatingPointID"].to_numpy(dtype=str)

    baseline = pd.read_csv(BASELINE_FEATURES)
    if baseline["RunID"].astype(str).tolist() != manifest["RunID"].tolist():
        raise ValueError("Baseline feature order does not match manifest")
    baseline_index = ["RunID", "OperatingPointID", "Label", "FaultGroup", "FaultSubtype", "Mode", "Temperature_C", "CurrentLevel_A"]
    baseline_names = [column for column in baseline.columns if column not in baseline_index]
    baseline_matrix = baseline[baseline_names].to_numpy(dtype=float)

    delta_rows: list[dict[str, float]] = []
    pre_rows: list[dict[str, float]] = []
    nonfinite_signal_cells = 0
    with h5py.File(H5_PATH, "r") as h5:
        names = decode_string_matrix(h5["meta/signal_names"])
        time_axis = np.asarray(h5["time_s"]).reshape(-1).astype(float)
        dt = float(np.median(np.diff(time_axis)))
        if set(h5["runs"].keys()) != set(manifest["RunID"]):
            raise ValueError("HDF5 and manifest RunID sets differ")
        for run_id in manifest["RunID"]:
            signals = signal_map(h5[f"runs/{run_id}/signals"], names)
            nonfinite_signal_cells += sum(int(np.sum(~np.isfinite(values))) for values in signals.values())
            channels = derived_channels(signals, include_devices=False)
            delta_rows.append(differential_features(channels, dt, pre=(0.000, 0.006), post=(0.012, 0.020)))
            pre_rows.append(differential_features(channels, dt, pre=(0.000, 0.003), post=(0.003, 0.006)))
        origin = str(np.asarray(h5.attrs["data_origin"]).reshape(-1)[0])

    delta_frame = pd.DataFrame(delta_rows)
    pre_frame = pd.DataFrame(pre_rows)
    delta_names = list(delta_frame.columns)
    compact_names = [name for name in delta_names if name.split("__", 1)[0] in COMPACT_CHANNELS]
    pre_compact_names = [name for name in pre_frame.columns if name.split("__", 1)[0] in COMPACT_CHANNELS]
    matrices = {
        "baseline_full": baseline_matrix,
        "delta_all": delta_frame[delta_names].to_numpy(dtype=float),
        "delta_compact": delta_frame[compact_names].to_numpy(dtype=float),
    }
    feature_names = {
        "baseline_full": baseline_names,
        "delta_all": delta_names,
        "delta_compact": compact_names,
    }
    delta_export = manifest[["RunID", "OperatingPointID", "Label", "FaultGroup", "Mode", "Temperature_C", "CurrentLevel_A"]].join(delta_frame)
    delta_export.to_csv(OUTPUT / "differential_features.csv", index=False)

    seed_sets = {group: set(part["RandomSeed"].astype(int)) for group, part in manifest.groupby("FaultGroup")}
    seed_overlaps = []
    seed_groups = sorted(seed_sets)
    for index, first in enumerate(seed_groups):
        for second in seed_groups[index + 1 :]:
            overlap = seed_sets[first].intersection(seed_sets[second])
            if overlap:
                seed_overlaps.append({"first": first, "second": second, "count": len(overlap)})
    data_quality = {
        "dataset_origin": origin,
        "runs": len(manifest),
        "run_id_unique": bool(manifest["RunID"].is_unique),
        "operating_points": int(manifest["OperatingPointID"].nunique()),
        "groups_with_all_six_classes": int(manifest.groupby("OperatingPointID")["FaultGroup"].nunique().eq(6).sum()),
        "nonfinite_signal_cells": nonfinite_signal_cells,
        "required_manifest_nulls": {
            column: int(manifest[column].isna().sum())
            for column in ["RunID", "FaultGroup", "Mode", "CurrentLevel_A", "Temperature_C", "RandomSeed"]
        },
        "random_seed_ranges_by_fault_group": {
            group: [int(part["RandomSeed"].min()), int(part["RandomSeed"].max())]
            for group, part in manifest.groupby("FaultGroup")
        },
        "random_seed_overlap_pairs": seed_overlaps,
        "seed_label_shortcut_risk": "high: all fault families use disjoint seed ranges",
    }
    (OUTPUT / "data_quality.json").write_text(json.dumps(data_quality, ensure_ascii=False, indent=2), encoding="utf-8")

    iteration_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    candidate_probabilities: dict[str, np.ndarray] = {}
    for candidate in CANDIDATES:
        matrix = matrices[candidate.representation]
        prediction, probabilities, candidate_folds, train_seconds = grouped_candidate_oof(candidate, matrix, labels, groups)
        result = metrics(labels, prediction)
        success_low, success_high = exact_interval(int(np.sum(prediction == labels)), len(labels))
        result.update(
            {
                "iteration": candidate.iteration,
                "candidate": candidate.name,
                "representation": candidate.representation,
                "feature_count": matrix.shape[1],
                "training_seconds_total": train_seconds,
                "accuracy_ci95_low": success_low,
                "accuracy_ci95_high": success_high,
                "rationale": candidate.rationale,
            }
        )
        iteration_rows.append(result)
        fold_rows.extend(candidate_folds)
        candidate_probabilities[candidate.name] = probabilities
        for index, row in manifest.iterrows():
            prediction_rows.append(
                {
                    "iteration": candidate.iteration,
                    "candidate": candidate.name,
                    "RunID": row["RunID"],
                    "OperatingPointID": row["OperatingPointID"],
                    "true_label": int(labels[index]),
                    "predicted_label": int(prediction[index]),
                    "correct": int(prediction[index] == labels[index]),
                }
            )

    iteration_table = pd.DataFrame(iteration_rows).sort_values("iteration")
    iteration_table.to_csv(OUTPUT / "iteration_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT / "iteration_fold_results.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUTPUT / "iteration_oof_predictions.csv", index=False)

    nested_prediction = np.full(len(labels), -1, dtype=int)
    nested_probabilities = np.full((len(labels), len(LABELS)), np.nan)
    selection_rows: list[dict[str, object]] = []
    outer = GroupKFold(n_splits=6)
    for outer_fold, (train_idx, test_idx) in enumerate(outer.split(np.zeros(len(labels)), labels, groups), start=1):
        scores = {}
        for candidate in CANDIDATES:
            matrix = matrices[candidate.representation]
            scores[candidate.name] = inner_score(
                candidate,
                matrix[train_idx],
                labels[train_idx],
                groups[train_idx],
                outer_fold,
            )
        chosen = max(CANDIDATES, key=lambda item: (scores[item.name], -item.iteration))
        matrix = matrices[chosen.representation]
        model = make_model(chosen, matrix.shape[1], SEED + 20000 + outer_fold)
        fit_model(model, matrix[train_idx], labels[train_idx])
        fold_prediction, fold_probabilities = predict_aligned(model, matrix[test_idx])
        nested_prediction[test_idx] = fold_prediction
        nested_probabilities[test_idx] = fold_probabilities
        selection_rows.append(
            {
                "outer_fold": outer_fold,
                "chosen_candidate": chosen.name,
                "chosen_representation": chosen.representation,
                "inner_objective": scores[chosen.name],
                "test_groups": "|".join(sorted(np.unique(groups[test_idx]))),
                **{f"inner_{name}": value for name, value in scores.items()},
                **{f"outer_{key}": value for key, value in metrics(labels[test_idx], fold_prediction).items()},
            }
        )
    if np.any(nested_prediction < 0) or np.isnan(nested_probabilities).any():
        raise ValueError("Nested OOF predictions are incomplete")
    nested_metrics = metrics(labels, nested_prediction)
    nested_accuracy_low, nested_accuracy_high = exact_interval(int(np.sum(nested_prediction == labels)), len(labels))
    nested_metrics.update({"accuracy_ci95_low": nested_accuracy_low, "accuracy_ci95_high": nested_accuracy_high})
    pd.DataFrame(selection_rows).to_csv(OUTPUT / "nested_selection_by_fold.csv", index=False)
    nested_output = manifest[["RunID", "OperatingPointID", "FaultGroup"]].copy()
    nested_output["true_label"] = labels
    nested_output["predicted_label"] = nested_prediction
    nested_output["correct"] = (labels == nested_prediction).astype(int)
    for label in LABELS:
        nested_output[f"probability_{label}"] = nested_probabilities[:, label]
    nested_output.to_csv(OUTPUT / "nested_oof_predictions.csv", index=False)

    pre_matrix = pre_frame[pre_compact_names].to_numpy(dtype=float)
    pre_prediction, pre_metrics = negative_control_oof(pre_matrix, labels, groups)
    negative_control = {
        "scope": "0-6 ms before any injected fault",
        "representation": "two half-window differences on compact controller channels",
        **pre_metrics,
        "random_chance_macro_f1": 1 / 6,
    }
    (OUTPUT / "negative_control.json").write_text(json.dumps(negative_control, ensure_ascii=False, indent=2), encoding="utf-8")

    best_row = iteration_table.sort_values(["selection_objective", "iteration"], ascending=[False, True]).iloc[0]
    best_candidate = next(candidate for candidate in CANDIDATES if candidate.name == best_row["candidate"])
    best_matrix = matrices[best_candidate.representation]
    final_model = make_model(best_candidate, best_matrix.shape[1], SEED + 30000)
    fit_model(final_model, best_matrix, labels)
    status = "provisional_synthetic_only"
    artifact = {
        "model": final_model,
        "candidate": best_candidate.name,
        "representation": best_candidate.representation,
        "feature_names": feature_names[best_candidate.representation],
        "label_names": {
            0: "healthy",
            1: "vbus_bias",
            2: "il_bias",
            3: "S1_open",
            4: "S2_open",
            5: "high_resistance",
        },
        "qualification_status": status,
        "required_scope": "synthetic reference HDF5 domain; controller-observable channels",
        "validation": {
            "fixed_candidate_group_oof": best_row.to_dict(),
            "nested_selection_group_oof": nested_metrics,
            "pre_fault_negative_control": negative_control,
        },
        "known_risk": "Fault-family RandomSeed ranges are disjoint; regenerate a label-independent frozen blind set before promotion.",
    }
    artifact_path = MODEL_OUTPUT / "optimized_controller_fault_model_provisional.joblib"
    joblib.dump(artifact, artifact_path)

    baseline_result = iteration_table.loc[iteration_table["iteration"].eq(1)].iloc[0]
    improvement = float(best_row["macro_f1"] - baseline_result["macro_f1"])
    qualifies_research_gate = bool(
        nested_metrics["macro_f1"] >= 0.90
        and nested_metrics["healthy_far"] <= 0.05
        and nested_metrics["high_resistance_recall"] >= 0.90
    )
    release_blocked = bool(pre_metrics["macro_f1"] > 0.22 or data_quality["seed_label_shortcut_risk"].startswith("high"))
    manifest_payload = {
        "created_date": "2026-08-06",
        "iterations_completed": len(CANDIDATES),
        "best_fixed_candidate": best_candidate.name,
        "baseline_macro_f1": float(baseline_result["macro_f1"]),
        "best_macro_f1": float(best_row["macro_f1"]),
        "macro_f1_improvement": improvement,
        "nested_metrics": nested_metrics,
        "pre_fault_negative_control": negative_control,
        "research_gate_passed": qualifies_research_gate,
        "promotion_blocked": release_blocked,
        "promotion_block_reason": "Synthetic label-correlated seed ranges require a newly generated frozen blind set.",
        "artifact": str(artifact_path),
    }
    (MODEL_OUTPUT / "manifest.json").write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "optimization_summary.json").write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(iteration_table))
    ax.bar(x, iteration_table["macro_f1"], color="#2f6f9f", label="Macro-F1")
    ax.plot(x, iteration_table["high_resistance_recall"], color="#d97941", marker="o", label="High-R recall")
    ax.plot(x, iteration_table["healthy_far"], color="#b23a48", marker="s", label="Healthy FAR")
    ax.set_xticks(x, [f"I{value}" for value in iteration_table["iteration"]])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Prespecified optimization iteration")
    ax.set_ylabel("Grouped OOF metric")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "iteration_comparison.png", dpi=180)
    plt.close(fig)

    report_lines = [
        "# HDF5机器学习模型七轮优化报告",
        "",
        "## 结论",
        "",
        f"- 已完成 {len(CANDIDATES)} 轮预先定义的模型/特征迭代；外层6折按工况评估，内层5折仅用于候选选择。",
        f"- 原Extra Trees基线 Macro-F1={baseline_result['macro_f1']:.4f}、健康FAR={baseline_result['healthy_far']:.4f}、高阻召回={baseline_result['high_resistance_recall']:.4f}。",
        f"- 最佳固定候选为 `{best_candidate.name}`：Macro-F1={best_row['macro_f1']:.4f}、健康FAR={best_row['healthy_far']:.4f}、高阻召回={best_row['high_resistance_recall']:.4f}，相对基线变化={improvement:+.4f}。",
        f"- 嵌套选择OOF：Macro-F1={nested_metrics['macro_f1']:.4f}、Accuracy={nested_metrics['accuracy']:.4f}、健康FAR={nested_metrics['healthy_far']:.4f}、高阻召回={nested_metrics['high_resistance_recall']:.4f}。",
        f"- 故障前负对照 Macro-F1={pre_metrics['macro_f1']:.4f}（六分类随机水平0.1667）。",
        f"- 研究推进门槛={'通过' if qualifies_research_gate else '未通过'}；模型晋级为外部验证/部署版本={'阻止' if release_blocked else '允许'}。",
        "",
        "## 七轮迭代",
        "",
        "|轮次|候选|表示|特征数|Macro-F1|健康FAR|高阻召回|",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in iteration_table.itertuples():
        report_lines.append(
            f"|I{row.iteration}|{row.candidate}|{row.representation}|{row.feature_count}|{row.macro_f1:.4f}|{row.healthy_far:.4f}|{row.high_resistance_recall:.4f}|"
        )
    report_lines.extend(
        [
            "",
            "## 仍然存在的问题",
            "",
            "1. 数据是合成工程参考数据，不是实测或HIL数据。",
            "2. 六个故障族使用互不重叠的RandomSeed区间，且种子影响波形相位；即使差分特征降低相位敏感性，也不能证明捷径已消失。",
            "3. 只有12个工况、162个Run；模型分数对少量分组仍可能敏感。",
            "4. 本轮产物只能标记为`provisional_synthetic_only`，必须用标签独立/配对随机种子重新生成全新冻结盲测后才能晋级。",
            "",
            "## 复现文件",
            "",
            f"- 训练脚本：`{Path(__file__)}`",
            f"- 结果目录：`{OUTPUT}`",
            f"- 暂定模型：`{artifact_path}`",
        ]
    )
    (OUTPUT / "optimization_report_cn.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines[:9]))
    print(f"Outputs: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
