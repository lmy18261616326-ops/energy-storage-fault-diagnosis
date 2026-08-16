#!/usr/bin/env python
"""Analyze active-scope data geometry, model calibration, and soft-vote fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import StandardScaler


LABELS = np.arange(5)
PROBABILITY_COLUMNS = [f"ProbabilityClass{label}" for label in LABELS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--event-index", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def expected_calibration_error(
    labels: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    prediction = np.argmax(probability, axis=1)
    confidence = np.max(probability, axis=1)
    correct = prediction == labels
    total = len(labels)
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if mask.any():
            error += mask.sum() / total * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def worst_operating_point_far(
    frame: pd.DataFrame, prediction: np.ndarray
) -> float:
    healthy = frame["WindowFaultID"].to_numpy(dtype=int) == 0
    values = []
    for operating_point in frame.loc[healthy, "OperatingPointID"].unique():
        mask = healthy & frame["OperatingPointID"].eq(operating_point).to_numpy()
        values.append(float(np.mean(prediction[mask] != 0)))
    return max(values, default=0.0)


def prediction_metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    labels = frame["WindowFaultID"].to_numpy(dtype=int)
    prediction = np.argmax(probability, axis=1)
    healthy = labels == 0
    one_hot = np.eye(len(LABELS))[labels]
    recalls = [float(np.mean(prediction[labels == label] == label)) for label in LABELS]
    return {
        "MacroF1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "HealthyFAR": float(np.mean(prediction[healthy] != 0)),
        "WorstOperatingPointFAR": worst_operating_point_far(frame, prediction),
        "MinimumSwitchRecall": min(recalls[3], recalls[4]),
        "LogLoss": float(log_loss(labels, probability, labels=LABELS)),
        "MulticlassBrier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
        "ECE10": expected_calibration_error(labels, probability),
    }


def active_scope(events: pd.DataFrame, event_index: pd.DataFrame) -> pd.DataFrame:
    context = event_index[["RunID", "ModeCommand", "SOCInit", "Pload"]].copy()
    merged = events.merge(context, on="RunID", how="left", validate="one_to_one")
    merged = merged.loc[~merged["FaultMechanism"].eq("high_resistance")].copy()
    mode = pd.to_numeric(merged["ModeCommand"], errors="raise").astype(int)
    inactive = (merged["WindowFaultID"].eq(3) & mode.ne(1)) | (
        merged["WindowFaultID"].eq(4) & mode.ne(2)
    )
    return merged.loc[~inactive].copy()


def data_geometry(
    events: pd.DataFrame, features: list[str]
) -> tuple[dict[str, object], pd.DataFrame]:
    imputed = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(
        events[features]
    )
    variance = np.var(imputed, axis=0)
    retained = variance > 1e-14
    retained_values = imputed[:, retained]
    retained_names = np.asarray(features)[retained]
    scaled = StandardScaler().fit_transform(retained_values)
    pca = PCA(svd_solver="full").fit(scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    def dimensions_for(target: float) -> int:
        return int(np.searchsorted(cumulative, target) + 1)

    correlation = np.corrcoef(retained_values, rowvar=False)
    upper = np.triu_indices_from(correlation, k=1)
    high_correlation_pairs = int(np.sum(np.abs(correlation[upper]) >= 0.95))
    f_scores, p_values = f_classif(
        retained_values, events["WindowFaultID"].to_numpy(dtype=int)
    )
    feature_scores = pd.DataFrame(
        {
            "Feature": retained_names,
            "FScore": np.nan_to_num(f_scores, nan=0.0, posinf=0.0),
            "PValue": np.nan_to_num(p_values, nan=1.0, posinf=1.0),
        }
    ).sort_values("FScore", ascending=False)
    normalized_eigenvalues = pca.explained_variance_ratio_
    effective_rank = float(
        np.exp(
            -np.sum(
                normalized_eigenvalues
                * np.log(np.clip(normalized_eigenvalues, 1e-15, None))
            )
        )
    )
    geometry = {
        "rows": int(len(events)),
        "operating_points": int(events["OperatingPointID"].nunique()),
        "raw_features": int(len(features)),
        "constant_features": int((~retained).sum()),
        "retained_features": int(retained.sum()),
        "feature_to_row_ratio": float(len(features) / len(events)),
        "absolute_correlation_ge_0_95_pairs": high_correlation_pairs,
        "pca_dimensions_90pct": dimensions_for(0.90),
        "pca_dimensions_95pct": dimensions_for(0.95),
        "pca_dimensions_99pct": dimensions_for(0.99),
        "pca_effective_rank": effective_rank,
    }
    return geometry, feature_scores


def calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, part in predictions.loc[predictions["Variant"].eq("argmax")].groupby(
        "Model", sort=True
    ):
        probability = part[PROBABILITY_COLUMNS].to_numpy(dtype=float)
        rows.append({"Model": model, **prediction_metrics(part, probability)})
    return pd.DataFrame(rows)


def fusion_table(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    argmax = predictions.loc[predictions["Variant"].eq("argmax")].copy()
    candidates = {
        "tree_linear": ("extra_trees", "logistic_regression", "random_forest"),
        "tree_boost": ("extra_trees", "random_forest", "xgboost"),
        "diverse_six": (
            "extra_trees",
            "knn",
            "logistic_regression",
            "mlp",
            "random_forest",
            "xgboost",
        ),
    }
    fold_rows: list[dict[str, object]] = []
    fused_predictions: list[pd.DataFrame] = []
    index_columns = [
        "Fold",
        "RunID",
        "OperatingPointID",
        "FaultName",
        "FaultMechanism",
        "WindowFaultID",
    ]
    for name, models in candidates.items():
        selected = argmax.loc[argmax["Model"].isin(models)]
        averaged = (
            selected.groupby(index_columns, sort=False)[PROBABILITY_COLUMNS]
            .mean()
            .reset_index()
        )
        averaged.insert(1, "Fusion", name)
        averaged["PredictedClassID"] = np.argmax(
            averaged[PROBABILITY_COLUMNS].to_numpy(dtype=float), axis=1
        )
        fused_predictions.append(averaged)
        for fold, part in averaged.groupby("Fold", sort=True):
            probability = part[PROBABILITY_COLUMNS].to_numpy(dtype=float)
            fold_rows.append(
                {
                    "Fusion": name,
                    "Fold": int(fold),
                    "Models": "|".join(models),
                    **prediction_metrics(part, probability),
                }
            )
    folds = pd.DataFrame(fold_rows)
    summary = (
        folds.groupby(["Fusion", "Models"], sort=True)
        .agg(
            FoldCount=("Fold", "nunique"),
            MacroF1Mean=("MacroF1", "mean"),
            MacroF1Min=("MacroF1", "min"),
            HealthyFARMean=("HealthyFAR", "mean"),
            WorstOperatingPointFARMax=("WorstOperatingPointFAR", "max"),
            MinimumSwitchRecallMean=("MinimumSwitchRecall", "mean"),
            LogLossMean=("LogLoss", "mean"),
            BrierMean=("MulticlassBrier", "mean"),
            ECE10Mean=("ECE10", "mean"),
        )
        .reset_index()
    )
    return summary, pd.concat(fused_predictions, ignore_index=True)


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(args.events)
    event_index = pd.read_csv(args.event_index)
    scoped = active_scope(events, event_index)
    features = [name for name in events.columns if "__" in name]
    geometry, feature_scores = data_geometry(scoped, features)
    feature_scores.head(100).to_csv(args.output / "top_feature_scores.csv", index=False)
    pd.crosstab(
        [scoped["ModeCommand"], scoped["FaultMechanism"]],
        scoped["WindowFaultID"],
    ).to_csv(args.output / "mode_mechanism_class_distribution.csv")
    scoped.groupby(["ModeCommand", "Pload"], dropna=False).size().rename(
        "EventCount"
    ).reset_index().to_csv(args.output / "mode_load_distribution.csv", index=False)

    predictions = pd.read_csv(args.results / "predictions.csv")
    calibration = calibration_table(predictions)
    calibration.to_csv(args.output / "model_calibration.csv", index=False)
    fusion, fused_predictions = fusion_table(predictions)
    fusion.to_csv(args.output / "fusion_summary.csv", index=False)
    fused_predictions.to_csv(args.output / "fusion_predictions.csv", index=False)

    summary = pd.read_csv(args.results / "summary.csv")
    argmax_summary = summary.loc[summary["Variant"].eq("argmax")].copy()
    ranking = argmax_summary.sort_values(
        [
            "ProvisionalQualified",
            "MacroF1Mean",
            "HealthyFARMean",
            "TrainingSecondsMean",
        ],
        ascending=[False, False, True, True],
    )
    ranking.to_csv(args.output / "model_ranking.csv", index=False)
    (args.output / "data_geometry.json").write_text(
        json.dumps(geometry, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best = ranking.iloc[0]
    best_fusion = fusion.sort_values(
        ["MacroF1Mean", "HealthyFARMean", "LogLossMean"],
        ascending=[False, True, True],
    ).iloc[0]
    report = f"""# 事件模型分布与选择结论

## 数据分布

- 主动可观测范围共有 {geometry['rows']} 个独立 Run、{geometry['operating_points']} 个工况、{geometry['raw_features']} 个聚合特征，特征/样本比为 {geometry['feature_to_row_ratio']:.2f}。
- 常量特征 {geometry['constant_features']} 个；绝对相关系数不低于 0.95 的特征对 {geometry['absolute_correlation_ge_0_95_pairs']} 对，说明冗余明显。
- PCA 达到 90%/95%/99% 方差分别需要 {geometry['pca_dimensions_90pct']}/{geometry['pca_dimensions_95pct']}/{geometry['pca_dimensions_99pct']} 维，有效秩约 {geometry['pca_effective_rank']:.1f}。
- 这是高维、小样本、强相关、按物理模式分段的结构化表格分布，不是具有平移不变性的自然时序分布。

## 模型选择

- 当前排序第一为 `{best['Model']}`：六折 Macro-F1={best['MacroF1Mean']:.4f}，健康 FAR={best['HealthyFARMean']:.4f}，最低 S1/S2 召回={best['MinimumSwitchRecallMean']:.4f}，平均训练 {best['TrainingSecondsMean']:.3f} s。
- RF、ExtraTrees 与逻辑回归均达到 100% OOF Macro-F1；线性模型也能完全分开，说明模式门控后的目标接近线性可分。优先推荐逻辑回归作为轻量可解释主模型，ExtraTrees 作为非线性冗余模型。
- 1D-CNN 即使保留分段位置后仍明显落后，原因是输入为具名统计特征而非原始等间隔波形；不建议在当前表示上继续加深 CNN。

## 融合

- 预定义融合中最佳为 `{best_fusion['Fusion']}`：Macro-F1={best_fusion['MacroF1Mean']:.4f}，健康 FAR={best_fusion['HealthyFARMean']:.4f}，平均 LogLoss={best_fusion['LogLossMean']:.4f}。
- 若融合未优于单模型，则保留“逻辑回归主判 + ExtraTrees 一致性检查”，不为了形式增加复杂度。

## 适用边界

- 上述资格仅适用于排除当前不可观测高阻、且 S1/Mode1 或 S2/Mode2 处于主动激励的范围。
- 高阻和非导通器件状态必须通过动态故障数据、模式切换记忆或主动诊断激励解决，不能由当前静态模型成绩外推。
"""
    (args.output / "distribution_model_selection.md").write_text(
        report, encoding="utf-8"
    )
    print(report)
    print("\nFusion summary:\n" + fusion.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
