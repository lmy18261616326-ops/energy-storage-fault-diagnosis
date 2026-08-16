#!/usr/bin/env python3
"""Generate publication figures from the verified Simulink CSV exports."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ML" / "reports" / "ieee_paper_simulink_enhanced_2026-08-04"
FIG = OUT / "figures"
SMOKE = (
    ROOT
    / "simulink"
    / "experiments"
    / "sensor_bias"
    / "scripts"
    / "dataset_output_v16"
    / "switch_observability_smoke"
)
VALID = SMOKE.parent / "switch_observability_validation"

BLUE = "#1F5A91"
ORANGE = "#E07A2D"
GREEN = "#2E8B57"
RED = "#C7473A"
PURPLE = "#7654A3"
GRAY = "#5D6B78"
GRID = "#D6DEE5"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.2,
            "legend.fontsize": 8.0,
            "axes.edgecolor": "#526575",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_smoke(case: str) -> pd.DataFrame:
    path = SMOKE / case / "combined" / "raw_dataset.csv"
    cols = [
        "Time",
        "Iref",
        "IL_meas",
        "Ibat_meas",
        "Vbus_meas",
        "Vbat_meas",
        "Iload_meas",
        "DutyApplied",
        "ConverterEnable",
        "S1_device_current",
        "S1_device_voltage",
        "S2_device_current",
        "S2_device_voltage",
        "S1_ron_estimate",
        "S2_ron_estimate",
        "Pload_meas",
        "PowerBalanceResidual",
    ]
    return pd.read_csv(path, usecols=cols)


def block(ax, xy, wh, title, detail, color, edge=BLUE, fontsize=9.0):
    x, y = xy
    w, h = wh
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        facecolor=color,
        edgecolor=edge,
        linewidth=1.25,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", weight="bold", color="#17324D", fontsize=fontsize)
    ax.text(x + w / 2, y + h * 0.30, detail, ha="center", va="center", color="#42586A", fontsize=fontsize - 1.1)
    return p


def arrow(ax, start, end, color=BLUE, text=None, rad=0.0, style="-|>", lw=1.25):
    a = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)
    if text:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.018, text, ha="center", va="bottom", fontsize=7.7, color=color)


def architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 6.6), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, "Simulink implementation architecture", ha="center", va="center", fontsize=14, weight="bold", color="#17324D")
    ax.text(0.5, 0.925, "Bidirectional power stage, cascaded control, fault injection, protection, and diagnostic observability", ha="center", va="center", fontsize=9.5, color=GRAY)

    # Power path
    block(ax, (0.035, 0.62), (0.13, 0.17), "DC bus", "Cdc = 2.2 mF\nESR = 1 mOhm", "#E8F1F8")
    block(ax, (0.21, 0.62), (0.15, 0.17), "Half bridge", "S1 / S2 IGBT-diode\nRon = 1 mOhm", "#FDE9DE", edge=ORANGE)
    block(ax, (0.405, 0.62), (0.13, 0.17), "Inductor", "L = 15 mH\nrL = 1 Ohm", "#E9F4EA", edge=GREEN)
    block(ax, (0.58, 0.62), (0.14, 0.17), "Battery", "Li-ion, 200 V\n9.8 Ah + Rbat", "#FFF4D9", edge="#B8881B")
    block(ax, (0.765, 0.62), (0.19, 0.17), "Load and source", "R load + stepped P load\nVdc reference = 400 V", "#EEF0F4", edge=GRAY)
    arrow(ax, (0.165, 0.735), (0.21, 0.735), ORANGE, "charge / buck")
    arrow(ax, (0.36, 0.735), (0.405, 0.735), ORANGE)
    arrow(ax, (0.535, 0.735), (0.58, 0.735), ORANGE)
    arrow(ax, (0.72, 0.735), (0.765, 0.735), ORANGE)
    arrow(ax, (0.765, 0.665), (0.72, 0.665), GREEN, "discharge / boost")
    arrow(ax, (0.58, 0.665), (0.535, 0.665), GREEN)
    arrow(ax, (0.405, 0.665), (0.36, 0.665), GREEN)
    arrow(ax, (0.21, 0.665), (0.165, 0.665), GREEN)

    # Control and supervision layer, with actual subsystem names.
    block(ax, (0.035, 0.30), (0.15, 0.17), "Mode_Manager", "charge/discharge logic\ncurrent-sign selection", "#E8F1F8")
    block(ax, (0.225, 0.30), (0.16, 0.17), "Voltage + current PI", "feedforward duty\nlimits and anti-windup", "#E5F2ED", edge=GREEN)
    block(ax, (0.425, 0.30), (0.14, 0.17), "PWM controller", "dead time +\ncomplementary gates", "#FFF4D9", edge="#B8881B")
    block(ax, (0.60, 0.30), (0.16, 0.17), "Fault injection", "sensor bias + S1/S2\nopen/partial/intermittent/Ron", "#FDE9DE", edge=RED)
    block(ax, (0.80, 0.30), (0.16, 0.17), "Power stage", "gate commands and\nswitched network", "#EEF0F4", edge=GRAY)
    for x0, x1 in [(0.185, 0.225), (0.385, 0.425), (0.565, 0.60), (0.76, 0.80)]:
        arrow(ax, (x0, 0.385), (x1, 0.385), BLUE)

    # Measurement and diagnostic layer.
    block(ax, (0.09, 0.055), (0.19, 0.14), "Synchronized sensing", "Vdc, Vbat, iL, ibat, iload\nS1/S2 device voltage and current", "#E8F1F8")
    block(ax, (0.405, 0.055), (0.19, 0.14), "Fault_Diag_Manager", "residuals, event windows,\nphysics-guided Ron estimate", "#EEE8F6", edge=PURPLE)
    block(ax, (0.72, 0.055), (0.19, 0.14), "Energy_Protection_Manager", "power/energy residual, SOC and\ncharge-discharge permission", "#FDE9DE", edge=RED)
    arrow(ax, (0.185, 0.30), (0.185, 0.195), GRAY, "mode + references")
    arrow(ax, (0.88, 0.30), (0.28, 0.125), GRAY, "measured electrical states", rad=-0.12)
    arrow(ax, (0.28, 0.125), (0.405, 0.125), PURPLE)
    arrow(ax, (0.595, 0.125), (0.72, 0.125), RED)
    arrow(ax, (0.815, 0.195), (0.69, 0.30), RED, "trip / permission", rad=-0.08)
    arrow(ax, (0.50, 0.195), (0.305, 0.30), PURPLE, "diagnostic residuals", rad=0.08)

    ax.text(0.98, 0.015, "Verified model: main_model_fd_v06_switchobservability.slx | powergui discrete, Ts = 1 us", ha="right", va="bottom", fontsize=7.7, color=GRAY)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "simulink_implementation_architecture.png", bbox_inches="tight")
    plt.close(fig)


def steady_mean(df: pd.DataFrame, col: str, a: float, b: float) -> float:
    w = df.loc[(df.Time >= a) & (df.Time <= b), col]
    return float(w.mean())


def healthy_waveform_figure(m1: pd.DataFrame, m2: pd.DataFrame) -> dict:
    fig, ax = plt.subplots(2, 2, figsize=(11.7, 7.0), dpi=220, sharex=False)

    for df, color, label in [(m1, ORANGE, "Mode 1: charge/buck"), (m2, GREEN, "Mode 2: discharge/boost")]:
        w = df.Time <= 0.16
        ax[0, 0].plot(df.loc[w, "Time"], df.loc[w, "IL_meas"], color=color, lw=1.35, label=label)
        ax[0, 0].plot(df.loc[w, "Time"], df.loc[w, "Iref"], color=color, lw=0.9, ls="--", alpha=0.75)
    ax[0, 0].axvline(0.006, color=GRAY, ls=":", lw=1.0)
    ax[0, 0].text(0.008, 0.92, "converter enabled", transform=ax[0, 0].get_xaxis_transform(), color=GRAY, fontsize=7.7)
    ax[0, 0].set(title="(a) Bidirectional current start-up", xlabel="Time (s)", ylabel="Inductor current (A)")
    ax[0, 0].legend(loc="center right")

    ax[0, 1].plot(m1.Time, m1.Vbus_meas, color=ORANGE, lw=1.05, label="Mode 1")
    ax[0, 1].plot(m2.Time, m2.Vbus_meas, color=GREEN, lw=1.05, label="Mode 2")
    ax[0, 1].axvline(0.35, color=RED, ls="--", lw=1.0, label="load step")
    ax[0, 1].set(title="(b) DC-bus response over the full run", xlabel="Time (s)", ylabel="DC-bus voltage (V)")
    ax[0, 1].legend(loc="best")

    for df, color, label in [(m1, ORANGE, "Mode 1"), (m2, GREEN, "Mode 2")]:
        w = (df.Time >= 0.28) & (df.Time <= 0.50)
        ax[1, 0].plot(df.loc[w, "Time"], df.loc[w, "Iload_meas"], color=color, lw=1.15, label=label)
    ax[1, 0].axvline(0.35, color=RED, ls="--", lw=1.0)
    ax[1, 0].set(title="(c) Applied load-current step", xlabel="Time (s)", ylabel="Load current (A)")
    ax[1, 0].legend(loc="best")

    for df, color, label in [(m1, ORANGE, "Mode 1"), (m2, GREEN, "Mode 2")]:
        w = (df.Time >= 0.28) & (df.Time <= 0.50)
        ax[1, 1].plot(df.loc[w, "Time"], df.loc[w, "DutyApplied"], color=color, lw=1.15, label=label)
    ax[1, 1].axvline(0.35, color=RED, ls="--", lw=1.0)
    ax[1, 1].set(title="(d) Duty-ratio compensation", xlabel="Time (s)", ylabel="Applied duty ratio")
    ax[1, 1].legend(loc="best")

    for a in ax.ravel():
        a.grid(True)
    fig.suptitle("Healthy bidirectional operation and load-step response", fontsize=13, weight="bold", color="#17324D")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(FIG / "healthy_bidirectional_waveforms.png", bbox_inches="tight")
    plt.close(fig)

    metrics = {}
    for key, df in [("mode1", m1), ("mode2", m2)]:
        pre = steady_mean(df, "Vbus_meas", 0.30, 0.34)
        post = steady_mean(df, "Vbus_meas", 0.90, 0.95)
        metrics[key] = {
            "pre_step_vbus_mean_v": pre,
            "post_step_vbus_mean_v": post,
            "load_step_vbus_change_v": post - pre,
            "load_step_vbus_change_pct": 100.0 * (post - pre) / pre,
            "steady_il_mean_a": steady_mean(df, "IL_meas", 0.90, 0.95),
            "steady_il_std_a": float(df.loc[(df.Time >= 0.90) & (df.Time <= 0.95), "IL_meas"].std()),
            "post_step_duty_mean": steady_mean(df, "DutyApplied", 0.90, 0.95),
        }
    return metrics


def fit_device(df: pd.DataFrame, switch: str) -> tuple[np.ndarray, np.ndarray, float]:
    i = df[f"{switch}_device_current"].to_numpy(float)
    v = df[f"{switch}_device_voltage"].to_numpy(float)
    mask = (df.Time.to_numpy() >= 0.20) & (np.abs(i) >= 1.0) & (np.abs(v) <= 5.0)
    x = np.abs(i[mask])
    y = np.abs(v[mask])
    slope = float(np.polyfit(x, y, 1)[0]) if x.size >= 2 else float("nan")
    return x, y, slope


def validation_resistance() -> pd.DataFrame:
    rows = []
    specs = [
        ("mode1_health", "S1", "healthy"),
        ("mode1_s1_high_r", "S1", "50 mOhm"),
        ("mode2_health", "S2", "healthy"),
        ("mode2_s2_high_r", "S2", "50 mOhm"),
    ]
    for folder, switch, status in specs:
        p = VALID / folder / "combined" / "feature_dataset.csv"
        d = pd.read_csv(p, usecols=["RunID", f"{switch}_ron_estimateMedian"])
        per_run = d.groupby("RunID", as_index=False)[f"{switch}_ron_estimateMedian"].median()
        for _, row in per_run.iterrows():
            rows.append({"RunID": row.RunID, "switch": switch, "status": status, "ron": float(row.iloc[1])})
    return pd.DataFrame(rows)


def high_r_figure(m1h: pd.DataFrame, m1f: pd.DataFrame, m2h: pd.DataFrame, m2f: pd.DataFrame) -> dict:
    fig, ax = plt.subplots(2, 2, figsize=(11.7, 7.2), dpi=220)

    w = (m1h.Time >= 0.25) & (m1h.Time <= 0.30)
    ax[0, 0].plot(m1h.loc[w, "Time"], m1h.loc[w, "IL_meas"], color=BLUE, lw=1.0, label="S1 healthy")
    ax[0, 0].plot(m1f.loc[w, "Time"], m1f.loc[w, "IL_meas"], color=RED, lw=1.0, alpha=0.85, label="S1 high-R (50 mOhm)")
    ax[0, 0].set(title="(a) Weak change in system-level current", xlabel="Time (s)", ylabel="Inductor current (A)")
    ax[0, 0].legend(loc="best")

    slopes = {}
    for df, switch, status, color in [
        (m1h, "S1", "S1 healthy", BLUE),
        (m1f, "S1", "S1 high-R", RED),
    ]:
        x, y, slope = fit_device(df, switch)
        slopes[status] = slope
        take = np.linspace(0, len(x) - 1, min(len(x), 550)).astype(int)
        ax[0, 1].scatter(x[take], y[take] * 1e3, s=8, alpha=0.40, color=color, label=f"{status}: {1e3*slope:.1f} mOhm")
    ax[0, 1].set(title="(b) S1 on-state voltage-current evidence", xlabel="|Device current| (A)", ylabel="|On-state voltage| (mV)")
    ax[0, 1].legend(loc="best")

    w = (m2h.Time >= 0.25) & (m2h.Time <= 0.30)
    for df, status, color in [
        (m2h, "S2 healthy", GREEN),
        (m2f, "S2 high-R", PURPLE),
    ]:
        ron = float(df.loc[w, "S2_ron_estimate"].median())
        slopes[status] = ron
        ax[1, 0].plot(df.loc[w, "Time"], 1e3 * df.loc[w, "S2_ron_estimate"], color=color, lw=1.1, label=f"{status}: {1e3*ron:.1f} mOhm")
    ax[1, 0].axhline(10.5, color=ORANGE, ls="--", lw=1.1, label="10.5 mOhm threshold")
    ax[1, 0].set(title="(c) S2 bin-wise physics estimate", xlabel="Time (s)", ylabel="Estimated on-resistance (mOhm)")
    ax[1, 0].legend(loc="best")

    val = validation_resistance()
    xj = {("S1", "healthy"): 0, ("S1", "50 mOhm"): 1, ("S2", "healthy"): 2, ("S2", "50 mOhm"): 3}
    colors = {"healthy": BLUE, "50 mOhm": RED}
    for _, r in val.iterrows():
        x = xj[(r.switch, r.status)]
        jitter = ((sum(ord(c) for c in r.RunID) % 17) - 8) / 120.0
        ax[1, 1].scatter(x + jitter, r.ron * 1e3, color=colors[r.status], s=28, edgecolor="white", linewidth=0.45, zorder=3)
    ax[1, 1].axhline(10.5, color=ORANGE, ls="--", lw=1.25, label="decision threshold = 10.5 mOhm")
    ax[1, 1].set_xticks(range(4), ["S1\nhealthy", "S1\n50 mOhm", "S2\nhealthy", "S2\n50 mOhm"])
    ax[1, 1].set_ylim(-2, 56)
    ax[1, 1].set(title="(d) Independent unseen-level validation", ylabel="Run-median resistance estimate (mOhm)")
    ax[1, 1].legend(loc="center right")

    fig.suptitle("Observability redesign for high-resistance switch degradation", fontsize=13, weight="bold", color="#17324D")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(FIG / "switch_high_r_observability.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "fitted_resistance_ohm": slopes,
        "validation_runs": int(val.RunID.nunique()),
        "validation_fault_runs": int((val.status == "50 mOhm").sum()),
        "validation_healthy_runs": int((val.status == "healthy").sum()),
        "threshold_ohm": 0.0105,
        "validation_fault_detected": int(((val.status == "50 mOhm") & (val.ron > 0.0105)).sum()),
        "validation_false_alarms": int(((val.status == "healthy") & (val.ron > 0.0105)).sum()),
    }


def copy_secondary_figures() -> None:
    sources = {
        ROOT / "ML" / "results" / "model_selection_notebook" / "pca_geometry.png": "pca_geometry.png",
        ROOT / "ML" / "results" / "ieee_computer_evidence_v17" / "figures" / "main_model_run_metrics.png": "main_model_run_metrics.png",
        ROOT / "ML" / "results" / "ieee_computer_evidence_v17" / "figures" / "high_r_detection_curve.png": "high_r_detection_curve.png",
    }
    for source, name in sources.items():
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, FIG / name)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    set_style()
    architecture_figure()
    m1h = read_smoke("mode1_health")
    m1f = read_smoke("mode1_s1_high_r")
    m2h = read_smoke("mode2_health")
    m2f = read_smoke("mode2_s2_high_r")
    metrics = {
        "solver_fixed_step_s": 1e-6,
        "export_sample_step_s": float(m1h.Time.diff().median()),
        "healthy_operation": healthy_waveform_figure(m1h, m2h),
        "high_resistance_observability": high_r_figure(m1h, m1f, m2h, m2f),
    }
    copy_secondary_figures()
    (OUT / "simulation_waveform_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
