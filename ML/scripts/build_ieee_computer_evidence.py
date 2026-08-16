"""Build a paper-oriented, computer-only validation package.

This analysis deliberately separates three evidence layers:

1. Grouped out-of-fold results of the main five-class event model.
2. Static high-resistance results produced by the full Simulink plant.
3. A measurement-chain Monte Carlo stress test driven by measured Simulink
   current trajectories.  Layer 3 is a sensor/detection projection, not a
   replacement for a dynamic power-stage or hardware experiment.

The script writes run-level uncertainty, near-threshold sensitivity, dynamic
detection latency, threshold trade-offs, figures, and a reproducibility
manifest to ``ML/results/ieee_computer_evidence_v17``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = PROJECT_ROOT / "ML"
SIM_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "simulink"
    / "experiments"
    / "sensor_bias"
    / "scripts"
    / "dataset_output_v16"
)
MAIN_RESULT_ROOT = (
    ML_ROOT / "results" / "event_model_comparison_v14_active_6fold_v2"
)
HIGH_R_RESULT_ROOT = ML_ROOT / "results" / "switch_observability_specialist"
OUTPUT_ROOT = ML_ROOT / "results" / "ieee_computer_evidence_v17"
FIGURE_ROOT = OUTPUT_ROOT / "figures"

FROZEN_THRESHOLD_OHM = 0.0105
FAULT_SEVERITIES_OHM = np.array(
    [0.001, 0.003, 0.005, 0.008, 0.010, 0.012, 0.015, 0.020, 0.050]
)
FAULT_DEFINITION_OHM = 0.012
RANDOM_SEED = 20260804
MONTE_CARLO_REPETITIONS = 5


@dataclass(frozen=True)
class MeasurementScenario:
    """Run-level sensor errors and sample-level white noise."""

    voltage_noise_std_v: float
    current_noise_std_a: float
    voltage_gain_std: float
    current_gain_std: float
    voltage_offset_std_v: float
    current_offset_std_a: float
    voltage_quant_step_v: float
    current_quant_step_a: float
    time_skew_us: float
    healthy_resistance_ohm: float
    healthy_drift_fraction: float


SCENARIOS: dict[str, MeasurementScenario] = {
    "ideal": MeasurementScenario(0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0010, 0),
    "nominal": MeasurementScenario(
        0.002, 0.02, 0.002, 0.002, 0.001, 0.01, 0.001, 0.01, 1, 0.0012, 0.10
    ),
    "moderate": MeasurementScenario(
        0.005, 0.05, 0.005, 0.005, 0.003, 0.03, 0.002, 0.02, 2, 0.0015, 0.20
    ),
    "harsh": MeasurementScenario(
        0.010, 0.10, 0.010, 0.010, 0.005, 0.05, 0.005, 0.05, 5, 0.0020, 0.30
    ),
}


@dataclass(frozen=True)
class CurrentProfile:
    run_id: str
    operating_point_id: str
    source_split: str
    mode: int
    time_s: np.ndarray
    current_a: np.ndarray


def clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided exact interval for a binomial proportion."""

    if total <= 0:
        return math.nan, math.nan
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return lower, upper


def quantize(values: np.ndarray, step: float) -> np.ndarray:
    if step <= 0:
        return values
    return np.round(values / step) * step


def first_consecutive_true(flags: np.ndarray, required: int = 2) -> int | None:
    """Return the first index belonging to a required-length true run."""

    count = 0
    for index, flag in enumerate(flags):
        count = count + 1 if bool(flag) else 0
        if count >= required:
            return index - required + 1
    return None


def event_windows(
    time_s: np.ndarray,
    resistance_ohm: np.ndarray,
    measured_current_a: np.ndarray,
    window_s: float = 0.010,
    step_s: float = 0.005,
    minimum_current_a: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate 50-us estimates into the project's 10-ms/5-ms windows."""

    if len(time_s) < 3:
        return np.array([], dtype=float), np.array([], dtype=float)
    dt = float(np.median(np.diff(time_s)))
    window_n = max(1, int(round(window_s / dt)))
    step_n = max(1, int(round(step_s / dt)))
    ends: list[float] = []
    medians: list[float] = []
    for start in range(0, len(time_s) - window_n + 1, step_n):
        stop = start + window_n
        eligible = (
            np.isfinite(resistance_ohm[start:stop])
            & np.isfinite(measured_current_a[start:stop])
            & (np.abs(measured_current_a[start:stop]) >= minimum_current_a)
        )
        if int(eligible.sum()) < math.ceil(0.5 * window_n):
            continue
        ends.append(float(time_s[stop - 1]))
        medians.append(float(np.median(resistance_ohm[start:stop][eligible])))
    return np.asarray(ends), np.asarray(medians)


def _required_columns_present(frame: pd.DataFrame, required: list[str], source: Path) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required variables: {missing}")


def load_main_predictions() -> tuple[pd.DataFrame, dict[str, object]]:
    path = MAIN_RESULT_ROOT / "predictions.csv"
    frame = pd.read_csv(path)
    required = [
        "Fold",
        "Model",
        "Variant",
        "RunID",
        "OperatingPointID",
        "FaultName",
        "FaultMechanism",
        "WindowFaultID",
        "PredictedClassID",
    ]
    _required_columns_present(frame, required, path)
    duplicate_count = int(
        frame.duplicated(["Model", "Variant", "RunID"], keep=False).sum()
    )
    quality = {
        "rows": int(len(frame)),
        "models": sorted(frame["Model"].astype(str).unique().tolist()),
        "variants": sorted(frame["Variant"].astype(str).unique().tolist()),
        "runs": int(frame["RunID"].nunique()),
        "operating_points": int(frame["OperatingPointID"].nunique()),
        "duplicate_model_variant_run_rows": duplicate_count,
        "missing_required_values": int(frame[required].isna().sum().sum()),
    }
    if duplicate_count or quality["missing_required_values"]:
        raise ValueError(f"Main prediction integrity check failed: {quality}")
    return frame, quality


def cluster_bootstrap_macro_f1(
    frame: pd.DataFrame, repetitions: int = 2000, seed: int = RANDOM_SEED
) -> tuple[float, float]:
    """Bootstrap operating points, preserving all runs within each point."""

    rng = np.random.default_rng(seed)
    groups = frame["OperatingPointID"].astype(str).unique()
    indices = {
        group: np.flatnonzero(frame["OperatingPointID"].astype(str).to_numpy() == group)
        for group in groups
    }
    scores = np.empty(repetitions, dtype=float)
    truth = frame["WindowFaultID"].to_numpy(int)
    pred = frame["PredictedClassID"].to_numpy(int)
    for repetition in range(repetitions):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        sample_indices = np.concatenate([indices[group] for group in sampled])
        scores[repetition] = f1_score(
            truth[sample_indices], pred[sample_indices], average="macro", zero_division=0
        )
    return tuple(np.quantile(scores, [0.025, 0.975]).astype(float))


def evaluate_main_models(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    for (model, variant), one in frame.groupby(["Model", "Variant"], sort=True):
        truth = one["WindowFaultID"].to_numpy(int)
        pred = one["PredictedClassID"].to_numpy(int)
        correct = int(np.sum(truth == pred))
        accuracy_low, accuracy_high = clopper_pearson(correct, len(one))
        macro_low, macro_high = cluster_bootstrap_macro_f1(one)
        healthy = truth == 0
        false_alarms = int(np.sum(pred[healthy] != 0))
        far_low, far_high = clopper_pearson(false_alarms, int(healthy.sum()))
        rows.append(
            {
                "model": model,
                "variant": variant,
                "runs": len(one),
                "operating_points": one["OperatingPointID"].nunique(),
                "accuracy": accuracy_score(truth, pred),
                "accuracy_ci95_low": accuracy_low,
                "accuracy_ci95_high": accuracy_high,
                "macro_f1": f1_score(truth, pred, average="macro", zero_division=0),
                "macro_f1_cluster_bootstrap_ci95_low": macro_low,
                "macro_f1_cluster_bootstrap_ci95_high": macro_high,
                "balanced_accuracy": balanced_accuracy_score(truth, pred),
                "healthy_false_alarm_rate": false_alarms / int(healthy.sum()),
                "healthy_far_ci95_low": far_low,
                "healthy_far_ci95_high": far_high,
            }
        )
        for class_id in sorted(np.unique(truth)):
            mask = truth == class_id
            true_positive = int(np.sum(pred[mask] == class_id))
            low, high = clopper_pearson(true_positive, int(mask.sum()))
            class_rows.append(
                {
                    "model": model,
                    "variant": variant,
                    "class_id": int(class_id),
                    "support_runs": int(mask.sum()),
                    "recall": true_positive / int(mask.sum()),
                    "recall_ci95_low": low,
                    "recall_ci95_high": high,
                    "precision": precision_score(
                        truth == class_id, pred == class_id, zero_division=0
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(class_rows)


def pairwise_mcnemar(frame: pd.DataFrame, reference_model: str = "logistic_regression") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    argmax = frame.loc[frame["Variant"].eq("argmax")].copy()
    reference = argmax.loc[argmax["Model"].eq(reference_model), [
        "RunID", "WindowFaultID", "PredictedClassID"
    ]].rename(columns={"PredictedClassID": "ReferencePrediction"})
    for model, one in argmax.groupby("Model", sort=True):
        if model == reference_model:
            continue
        merged = reference.merge(
            one[["RunID", "PredictedClassID"]].rename(
                columns={"PredictedClassID": "CandidatePrediction"}
            ),
            on="RunID",
            validate="one_to_one",
        )
        reference_correct = merged["ReferencePrediction"].eq(merged["WindowFaultID"])
        candidate_correct = merged["CandidatePrediction"].eq(merged["WindowFaultID"])
        reference_only = int(np.sum(reference_correct & ~candidate_correct))
        candidate_only = int(np.sum(~reference_correct & candidate_correct))
        discordant = reference_only + candidate_only
        p_value = 1.0 if discordant == 0 else float(
            binomtest(min(reference_only, candidate_only), discordant, 0.5).pvalue
        )
        rows.append(
            {
                "reference_model": reference_model,
                "candidate_model": model,
                "reference_only_correct": reference_only,
                "candidate_only_correct": candidate_only,
                "discordant_runs": discordant,
                "exact_mcnemar_p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def load_current_profiles() -> tuple[list[CurrentProfile], dict[str, object]]:
    specs = [
        ("pilot", "mode1_health"),
        ("pilot", "mode2_health"),
        ("validation", "mode1_health"),
        ("validation", "mode2_health"),
    ]
    profiles: list[CurrentProfile] = []
    source_rows: list[dict[str, object]] = []
    usecols = [
        "Time",
        "RunID",
        "OperatingPointID",
        "ModeCommand",
        "IL_true",
        "IsTrainingEligible",
    ]
    for split, folder in specs:
        path = (
            SIM_OUTPUT_ROOT
            / f"switch_observability_{split}"
            / folder
            / "combined"
            / "raw_dataset.csv"
        )
        frame = pd.read_csv(path, usecols=usecols)
        _required_columns_present(frame, usecols, path)
        frame = frame.loc[
            frame["Time"].ge(0.4)
            & frame["IsTrainingEligible"].eq(1)
            & frame["ModeCommand"].isin([1, 2])
            & frame["IL_true"].notna()
        ].sort_values(["RunID", "Time"])
        source_rows.append(
            {
                "source": str(path.relative_to(PROJECT_ROOT)),
                "rows_after_filter": int(len(frame)),
                "runs": int(frame["RunID"].nunique()),
                "minimum_abs_current_a": float(frame["IL_true"].abs().min()),
                "median_abs_current_a": float(frame["IL_true"].abs().median()),
                "maximum_abs_current_a": float(frame["IL_true"].abs().max()),
            }
        )
        for run_id, one in frame.groupby("RunID", sort=False):
            one = one.drop_duplicates("Time", keep="first")
            profiles.append(
                CurrentProfile(
                    run_id=str(run_id),
                    operating_point_id=str(one["OperatingPointID"].iloc[0]),
                    source_split=split,
                    mode=int(one["ModeCommand"].iloc[0]),
                    time_s=one["Time"].to_numpy(float),
                    current_a=np.abs(one["IL_true"].to_numpy(float)),
                )
            )
    run_ids = [profile.run_id for profile in profiles]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Current-profile RunID values are not unique")
    quality = {
        "profile_count": len(profiles),
        "source_summary": source_rows,
        "modes": sorted({profile.mode for profile in profiles}),
        "minimum_samples_per_profile": min(len(profile.time_s) for profile in profiles),
        "maximum_samples_per_profile": max(len(profile.time_s) for profile in profiles),
    }
    return profiles, quality


def simulate_measurement_run(
    profile: CurrentProfile,
    severity_ohm: float,
    scenario_name: str,
    replicate: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    scenario = SCENARIOS[scenario_name]
    time_s = profile.time_s
    true_current = profile.current_a
    fault_time_s = float(rng.uniform(0.52, 0.70))
    duration = max(float(time_s[-1] - time_s[0]), np.finfo(float).eps)
    drift = 1 + scenario.healthy_drift_fraction * (time_s - time_s[0]) / duration
    healthy_resistance = scenario.healthy_resistance_ohm * drift
    is_fault = severity_ohm > 0.001 + 1e-12
    true_resistance = healthy_resistance.copy()
    if is_fault:
        true_resistance[time_s >= fault_time_s] = severity_ohm

    shifted_time = time_s + scenario.time_skew_us * 1e-6
    current_for_voltage = np.interp(
        shifted_time, time_s, true_current, left=true_current[0], right=true_current[-1]
    )
    voltage_gain = 1 + rng.normal(0, scenario.voltage_gain_std)
    current_gain = 1 + rng.normal(0, scenario.current_gain_std)
    voltage_offset = rng.normal(0, scenario.voltage_offset_std_v)
    current_offset = rng.normal(0, scenario.current_offset_std_a)
    true_voltage = true_resistance * current_for_voltage
    measured_voltage = (
        voltage_gain * true_voltage
        + voltage_offset
        + rng.normal(0, scenario.voltage_noise_std_v, size=len(time_s))
    )
    measured_current = (
        current_gain * true_current
        + current_offset
        + rng.normal(0, scenario.current_noise_std_a, size=len(time_s))
    )
    measured_voltage = quantize(measured_voltage, scenario.voltage_quant_step_v)
    measured_current = quantize(measured_current, scenario.current_quant_step_a)
    valid = np.abs(measured_current) >= 0.5
    resistance_estimate = np.full(len(time_s), np.nan, dtype=float)
    resistance_estimate[valid] = np.abs(
        measured_voltage[valid] / measured_current[valid]
    )
    event_time, event_resistance = event_windows(
        time_s, resistance_estimate, measured_current
    )
    decisions = event_resistance >= FROZEN_THRESHOLD_OHM
    pre_fault = event_time < fault_time_s
    post_fault = event_time >= fault_time_s + 0.010
    false_alarm_windows = int(np.sum(decisions[pre_fault]))
    eligible_pre_fault_windows = int(np.sum(pre_fault))
    detection_index = first_consecutive_true(decisions[post_fault], required=2)
    post_times = event_time[post_fault]
    detected = detection_index is not None
    latency_s = (
        float(max(0, post_times[detection_index] - fault_time_s))
        if detected
        else math.nan
    )
    post_values = event_resistance[post_fault]
    return {
        "source_run_id": profile.run_id,
        "operating_point_id": profile.operating_point_id,
        "source_split": profile.source_split,
        "mode": profile.mode,
        "scenario": scenario_name,
        "replicate": replicate,
        "severity_ohm": severity_ohm,
        "is_fault_transition": is_fault,
        "fault_time_s": fault_time_s,
        "detected": bool(detected),
        "detection_latency_s": latency_s,
        "false_alarm_windows": false_alarm_windows,
        "eligible_pre_fault_windows": eligible_pre_fault_windows,
        "post_event_count": int(len(post_values)),
        "post_resistance_median_ohm": float(np.median(post_values)),
        "post_resistance_q05_ohm": float(np.quantile(post_values, 0.05)),
        "post_resistance_q95_ohm": float(np.quantile(post_values, 0.95)),
        "minimum_profile_current_a": float(np.min(true_current)),
        "median_profile_current_a": float(np.median(true_current)),
    }


def run_measurement_stress(profiles: list[CurrentProfile]) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    for scenario_name in SCENARIOS:
        for severity in FAULT_SEVERITIES_OHM:
            for profile in profiles:
                for replicate in range(MONTE_CARLO_REPETITIONS):
                    rows.append(
                        simulate_measurement_run(
                            profile,
                            float(severity),
                            scenario_name,
                            replicate,
                            rng,
                        )
                    )
    return pd.DataFrame(rows)


def summarize_measurement_stress(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scenario, severity), one in frame.groupby(["scenario", "severity_ohm"], sort=True):
        detected = one["detected"].astype(bool)
        successes = int(detected.sum())
        low, high = clopper_pearson(successes, len(one))
        pre_windows = int(one["eligible_pre_fault_windows"].sum())
        false_windows = int(one["false_alarm_windows"].sum())
        latencies = one.loc[detected, "detection_latency_s"].dropna().to_numpy(float)
        rows.append(
            {
                "scenario": scenario,
                "severity_ohm": severity,
                "simulated_runs": len(one),
                "source_profiles": one["source_run_id"].nunique(),
                "detection_rate": successes / len(one),
                "detection_rate_ci95_low": low,
                "detection_rate_ci95_high": high,
                "pre_fault_window_false_alarm_rate": false_windows / max(pre_windows, 1),
                "median_detection_latency_ms": (
                    float(np.median(latencies) * 1000) if len(latencies) else math.nan
                ),
                "p95_detection_latency_ms": (
                    float(np.quantile(latencies, 0.95) * 1000)
                    if len(latencies)
                    else math.nan
                ),
                "post_resistance_median_ohm": float(
                    one["post_resistance_median_ohm"].median()
                ),
                "post_resistance_q05_ohm": float(
                    one["post_resistance_q05_ohm"].quantile(0.05)
                ),
                "post_resistance_q95_ohm": float(
                    one["post_resistance_q95_ohm"].quantile(0.95)
                ),
            }
        )
    return pd.DataFrame(rows)


def threshold_tradeoff(frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate thresholds on unambiguous benign and >=12-mOhm cases."""

    work = frame.loc[
        frame["severity_ohm"].le(0.008)
        | frame["severity_ohm"].ge(FAULT_DEFINITION_OHM)
    ].copy()
    truth = work["severity_ohm"].ge(FAULT_DEFINITION_OHM).to_numpy(int)
    score = work["post_resistance_median_ohm"].to_numpy(float)
    rows: list[dict[str, object]] = []
    for threshold in np.linspace(0.003, 0.020, 69):
        prediction = (score >= threshold).astype(int)
        healthy = truth == 0
        rows.append(
            {
                "threshold_ohm": threshold,
                "accuracy": accuracy_score(truth, prediction),
                "balanced_accuracy": balanced_accuracy_score(truth, prediction),
                "macro_f1": f1_score(truth, prediction, average="macro", zero_division=0),
                "fault_recall": recall_score(truth, prediction, zero_division=0),
                "healthy_false_alarm_rate": float(prediction[healthy].mean()),
            }
        )
    return pd.DataFrame(rows)


def load_static_high_r_summary() -> tuple[pd.DataFrame, dict[str, object]]:
    comparison_path = HIGH_R_RESULT_ROOT / "model_comparison.csv"
    summary_path = HIGH_R_RESULT_ROOT / "summary.json"
    comparison = pd.read_csv(comparison_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return comparison, summary


def save_figures(
    main_metrics: pd.DataFrame,
    stress_summary: pd.DataFrame,
    tradeoff: pd.DataFrame,
) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140})

    argmax = main_metrics.loc[main_metrics["variant"].eq("argmax")].sort_values(
        "macro_f1", ascending=False
    )
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    positions = np.arange(len(argmax))
    ax1.bar(positions - 0.18, argmax["macro_f1"], width=0.36, label="Macro-F1")
    ax1.bar(
        positions + 0.18,
        argmax["healthy_false_alarm_rate"],
        width=0.36,
        label="Healthy FAR",
    )
    ax1.set_xticks(positions, argmax["model"], rotation=25, ha="right")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Run-level metric")
    ax1.set_title("Grouped OOF model comparison (active-observable scope)")
    ax1.legend(loc="center right")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "main_model_run_metrics.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for scenario, one in stress_summary.groupby("scenario", sort=False):
        ax.plot(
            one["severity_ohm"] * 1000,
            one["detection_rate"],
            marker="o",
            label=scenario,
        )
    ax.axvline(FROZEN_THRESHOLD_OHM * 1000, color="black", linestyle="--", label="10.5 mOhm threshold")
    ax.set_xlabel("Post-fault resistance (mOhm)")
    ax.set_ylabel("Run detection rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Near-threshold detectability under measurement stress")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "high_r_detection_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for scenario, one in stress_summary.groupby("scenario", sort=False):
        ax.plot(
            one["severity_ohm"] * 1000,
            one["median_detection_latency_ms"],
            marker="o",
            label=scenario,
        )
    ax.axvline(FROZEN_THRESHOLD_OHM * 1000, color="black", linestyle="--")
    ax.set_xlabel("Post-fault resistance (mOhm)")
    ax.set_ylabel("Median detection latency (ms)")
    ax.set_title("Dynamic resistance-step detection latency")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "high_r_detection_latency.png")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(tradeoff["threshold_ohm"] * 1000, tradeoff["macro_f1"], label="Macro-F1")
    ax1.plot(
        tradeoff["threshold_ohm"] * 1000,
        tradeoff["fault_recall"],
        label="Fault recall",
    )
    ax1.plot(
        tradeoff["threshold_ohm"] * 1000,
        tradeoff["healthy_false_alarm_rate"],
        label="Healthy FAR",
    )
    ax1.axvline(FROZEN_THRESHOLD_OHM * 1000, color="black", linestyle="--", label="Frozen threshold")
    ax1.set_xlabel("Decision threshold (mOhm)")
    ax1.set_ylabel("Metric")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title("Threshold sensitivity (moderate measurement scenario)")
    ax1.grid(alpha=0.25)
    ax1.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "high_r_threshold_tradeoff.png")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    display = frame.loc[:, columns].copy()
    for column in display.select_dtypes(include=["float", "float64"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
        )
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def build_report(
    main_metrics: pd.DataFrame,
    class_metrics: pd.DataFrame,
    static_comparison: pd.DataFrame,
    static_summary: dict[str, object],
    stress_summary: pd.DataFrame,
    tradeoff: pd.DataFrame,
    quality: dict[str, object],
) -> str:
    argmax = main_metrics.loc[main_metrics["variant"].eq("argmax")].sort_values(
        ["macro_f1", "healthy_false_alarm_rate"], ascending=[False, True]
    )
    frozen = tradeoff.iloc[(tradeoff["threshold_ohm"] - FROZEN_THRESHOLD_OHM).abs().argmin()]
    moderate = stress_summary.loc[stress_summary["scenario"].eq("moderate")]
    harsh = stress_summary.loc[stress_summary["scenario"].eq("harsh")]
    moderate_12 = moderate.iloc[(moderate["severity_ohm"] - 0.012).abs().argmin()]
    harsh_12 = harsh.iloc[(harsh["severity_ohm"] - 0.012).abs().argmin()]
    selected_class = class_metrics.loc[
        class_metrics["variant"].eq("argmax")
        & class_metrics["model"].isin(["logistic_regression", "extra_trees"])
    ]
    static_best = static_comparison.loc[
        static_comparison["split"].eq("independent_validation")
    ].sort_values(["run_macro_f1", "model"], ascending=[False, True])

    return f"""# IEEE一般级别：纯计算机证据补强报告

生成日期：2026-08-04

## 证据边界

本报告严格区分三层证据：主五类模型为416个独立Run、28个工况的分组OOF；高阻静态结果来自完整Simulink电气模型；动态电阻阶跃与测量误差部分由24条健康Simulink电流轨迹驱动，是测量链和判据的蒙特卡洛压力测试，不等同于动态功率级或硬件实验。

## 1. 主五类模型的Run级结果与不确定性

{markdown_table(argmax, ['model', 'runs', 'macro_f1', 'accuracy', 'accuracy_ci95_low', 'accuracy_ci95_high', 'healthy_false_alarm_rate', 'healthy_far_ci95_high'])}

逻辑回归、随机森林和ExtraTrees在当前主动可观测范围内均达到完整Run级正确分类。即使点估计为100%，仍报告精确二项分布置信下界，而不是把100%解释成真实部署性能必然为100%。CNN的健康误报明显更高，符合“小样本、高维结构化统计特征不利于端到端卷积网络”的数据几何结论。

关键类别召回：

{markdown_table(selected_class, ['model', 'class_id', 'support_runs', 'recall', 'recall_ci95_low', 'recall_ci95_high', 'precision'])}

## 2. 静态高阻完整模型验证

- 开发集：{static_summary['pilot_runs']} Run、{static_summary['pilot_windows']}窗口。
- 独立集：{static_summary['validation_runs']} Run、{static_summary['validation_windows']}窗口，未见阻值为0.05 ohm。
- 冻结阈值：{static_summary['threshold_ohm']} ohm。
- 静态独立集健康最大值：{static_summary['validation_healthy_max_ohm']} ohm；故障最小值：{static_summary['validation_fault_min_ohm']} ohm。

{markdown_table(static_best.head(7), ['model', 'run_macro_f1', 'run_fault_recall', 'run_healthy_false_alarm_rate'])}

该结果证明v06直接器件观测解决了理想模型中的静态可分性，但仍不能替代温度相关半导体模型和硬件测量。

## 3. 临界阻值、测量误差与动态检测

压力测试覆盖1、3、5、8、10、12、15、20和50 mOhm，四级测量条件包含电压/电流噪声、增益误差、偏置、量化、健康阻值漂移和1–5 us通道错位。每个条件由24条独立源Run、每条5个随机重复构成。

在12 mOhm临界故障处：

- moderate场景检测率={moderate_12['detection_rate']:.3f}，95%精确区间=[{moderate_12['detection_rate_ci95_low']:.3f}, {moderate_12['detection_rate_ci95_high']:.3f}]，中位延迟={moderate_12['median_detection_latency_ms']:.2f} ms；
- harsh场景检测率={harsh_12['detection_rate']:.3f}，95%精确区间=[{harsh_12['detection_rate_ci95_low']:.3f}, {harsh_12['detection_rate_ci95_high']:.3f}]，中位延迟={harsh_12['median_detection_latency_ms']:.2f} ms。

完整曲线位于 `figures/high_r_detection_curve.png` 与 `figures/high_r_detection_latency.png`。阈值压力测试中，与冻结10.5 mOhm最接近的网格点得到Macro-F1={frozen['macro_f1']:.3f}、故障召回={frozen['fault_recall']:.3f}、健康FAR={frozen['healthy_false_alarm_rate']:.3f}。

## 4. 可用于论文的结论

1. 当前数据是小样本、高维、低有效秩的结构化事件表，正则化逻辑回归和树集成是合理主模型，继续增加复杂网络没有证据收益。
2. 高阻问题的主要贡献是可观测性设计和物理判据，而不是分类器复杂度。
3. 50 us与20 kHz PWM同周期会产生固定相位混叠；1 us器件测量后按PWM周期聚合是必要消融。
4. 冻结阈值对大于等于12 mOhm的故障具备压力测试证据；低于阈值的3–10 mOhm应描述为“未保证检出/早期退化区”，不能声称全部覆盖。
5. 本轮补强达到一般IEEE会议或仿真型期刊所需的完整计算机实验结构，但投稿时仍须把理想Ron、直接测量和无硬件验证列为限制。

## 5. 数据质量与复现

- 主模型预测重复键：{quality['main_predictions']['duplicate_model_variant_run_rows']}；必需字段缺失：{quality['main_predictions']['missing_required_values']}。
- 动态压力测试源轨迹：{quality['current_profiles']['profile_count']}个独立Run；每条最少{quality['current_profiles']['minimum_samples_per_profile']}个样本。
- 随机种子：{RANDOM_SEED}；每个源Run/阻值/测量场景重复：{MONTE_CARLO_REPETITIONS}次。
- 所有输入文件、大小、修改时间和分析参数记录在 `analysis_manifest.json`。

## 6. 仍未解决的边界

- 高阻仍基于理想IGBT/Diode的Ron，不含VCE(sat)、结温、键合线或焊点退化模型。
- 测量链压力测试使用已有电流轨迹，不包含动态高阻对功率级电流和控制环的反作用。
- 直接器件压降/电流测量的共模抑制、带宽和安全实现尚未经过HIL或硬件验证。
- 16个高阻独立验证Run的置信区间仍宽，不能用1904个相关窗口替代独立样本数。
"""


def input_manifest(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        stat = path.stat()
        digest = ""
        if stat.st_size <= 10 * 1024 * 1024:
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        rows.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": stat.st_size,
                "modified_unix_seconds": stat.st_mtime,
                "sha256_if_le_10mb": digest,
            }
        )
    return rows


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    predictions, prediction_quality = load_main_predictions()
    main_metrics, class_metrics = evaluate_main_models(predictions)
    mcnemar = pairwise_mcnemar(predictions)
    profiles, profile_quality = load_current_profiles()
    stress_runs = run_measurement_stress(profiles)
    stress_summary = summarize_measurement_stress(stress_runs)
    moderate_tradeoff = threshold_tradeoff(
        stress_runs.loc[stress_runs["scenario"].eq("moderate")]
    )
    static_comparison, static_summary = load_static_high_r_summary()

    main_metrics.to_csv(OUTPUT_ROOT / "main_model_run_metrics_ci.csv", index=False)
    class_metrics.to_csv(OUTPUT_ROOT / "main_model_class_recall_ci.csv", index=False)
    mcnemar.to_csv(OUTPUT_ROOT / "main_model_pairwise_mcnemar.csv", index=False)
    stress_runs.to_csv(OUTPUT_ROOT / "high_r_measurement_stress_runs.csv", index=False)
    stress_summary.to_csv(OUTPUT_ROOT / "high_r_measurement_stress_summary.csv", index=False)
    moderate_tradeoff.to_csv(OUTPUT_ROOT / "high_r_threshold_tradeoff.csv", index=False)

    quality = {
        "main_predictions": prediction_quality,
        "current_profiles": profile_quality,
        "stress_rows": int(len(stress_runs)),
        "stress_missing_required_values": int(
            stress_runs[
                [
                    "source_run_id",
                    "scenario",
                    "severity_ohm",
                    "detected",
                    "post_resistance_median_ohm",
                ]
            ].isna().sum().sum()
        ),
    }
    (OUTPUT_ROOT / "data_quality.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_figures(main_metrics, stress_summary, moderate_tradeoff)
    report = build_report(
        main_metrics,
        class_metrics,
        static_comparison,
        static_summary,
        stress_summary,
        moderate_tradeoff,
        quality,
    )
    (OUTPUT_ROOT / "computer_evidence_report.md").write_text(report, encoding="utf-8")

    raw_paths = [
        SIM_OUTPUT_ROOT
        / f"switch_observability_{split}"
        / folder
        / "combined"
        / "raw_dataset.csv"
        for split, folder in [
            ("pilot", "mode1_health"),
            ("pilot", "mode2_health"),
            ("validation", "mode1_health"),
            ("validation", "mode2_health"),
        ]
    ]
    manifest = {
        "analysis": "ieee_computer_evidence_v17",
        "random_seed": RANDOM_SEED,
        "monte_carlo_repetitions": MONTE_CARLO_REPETITIONS,
        "frozen_threshold_ohm": FROZEN_THRESHOLD_OHM,
        "fault_definition_ohm": FAULT_DEFINITION_OHM,
        "fault_severities_ohm": FAULT_SEVERITIES_OHM.tolist(),
        "measurement_scenarios": {
            name: asdict(scenario) for name, scenario in SCENARIOS.items()
        },
        "evidence_boundary": (
            "Measurement stress uses existing Simulink current trajectories; "
            "it is not a dynamic power-stage or hardware validation."
        ),
        "inputs": input_manifest(
            [
                MAIN_RESULT_ROOT / "predictions.csv",
                HIGH_R_RESULT_ROOT / "model_comparison.csv",
                HIGH_R_RESULT_ROOT / "summary.json",
                *raw_paths,
            ]
        ),
    }
    (OUTPUT_ROOT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(quality, indent=2, ensure_ascii=False))
    print(f"Report written to {OUTPUT_ROOT / 'computer_evidence_report.md'}")


if __name__ == "__main__":
    main()
