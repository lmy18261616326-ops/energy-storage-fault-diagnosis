"""Dataset loading and leakage-safe grouped splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitBundle:
    """Train/validation/test tables and their group assignments."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    group_assignments: pd.DataFrame


@dataclass(frozen=True)
class GroupCrossValidationFold:
    """One leakage-safe outer test fold with separate validation groups."""

    fold: int
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    validation_groups: tuple[str, ...]
    test_groups: tuple[str, ...]


def load_feature_dataset(
    path: str | Path,
    *,
    label_column: str = "WindowFaultID",
    eligibility_column: str = "IsTrainingEligible",
    expected_labels: Sequence[int] = (0, 1, 2, 3, 4),
) -> pd.DataFrame:
    """Load the MATLAB-generated feature CSV and retain eligible windows."""

    data_path = Path(path).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Feature dataset does not exist: {data_path}. "
            "Run collect_fault_dataset.m in MATLAB first or pass --data."
        )

    frame = pd.read_csv(data_path)
    required = {label_column, eligibility_column, "RunID", "OperatingPointID"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    eligible = pd.to_numeric(frame[eligibility_column], errors="coerce").fillna(0)
    frame = frame.loc[eligible.ne(0)].copy()
    if frame.empty:
        raise ValueError("No IsTrainingEligible != 0 windows remain.")

    numeric_labels = pd.to_numeric(frame[label_column], errors="coerce")
    if numeric_labels.isna().any():
        raise ValueError(f"{label_column} contains missing or non-numeric labels.")
    if not np.allclose(numeric_labels, np.round(numeric_labels)):
        raise ValueError(f"{label_column} must contain integer class IDs.")
    frame[label_column] = numeric_labels.astype(int)

    present = set(frame[label_column].unique().tolist())
    expected = set(int(value) for value in expected_labels)
    missing_labels = sorted(expected.difference(present))
    unexpected_labels = sorted(present.difference(expected))
    if missing_labels or unexpected_labels:
        raise ValueError(
            "Label coverage mismatch after eligibility filtering: "
            f"missing={missing_labels}, unexpected={unexpected_labels}."
        )

    frame["RunID"] = frame["RunID"].astype(str)
    frame["OperatingPointID"] = frame["OperatingPointID"].astype(str)
    return frame.reset_index(drop=True)


def _allocate_counts(group_count: int, ratios: Sequence[float]) -> tuple[int, int, int]:
    ratio_array = np.asarray(ratios, dtype=float)
    if ratio_array.shape != (3,) or np.any(ratio_array < 0):
        raise ValueError("split_ratios must contain three non-negative values.")
    if not np.isclose(ratio_array.sum(), 1.0):
        raise ValueError("split_ratios must sum to 1.")
    if group_count < 3:
        raise ValueError("At least three independent groups are required.")

    raw = group_count * ratio_array
    counts = np.floor(raw).astype(int)
    remainder = group_count - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    for index in order[:remainder]:
        counts[index] += 1

    for target in range(3):
        if counts[target] == 0:
            donor = int(np.argmax(counts))
            if counts[donor] <= 1:
                raise ValueError("Unable to allocate non-empty grouped splits.")
            counts[donor] -= 1
            counts[target] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def _labels_in(frame: pd.DataFrame, label_column: str) -> set[int]:
    return set(pd.to_numeric(frame[label_column]).astype(int).unique().tolist())


def assert_no_group_leakage(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    fields: Iterable[str] = ("RunID", "OperatingPointID"),
) -> None:
    """Raise when any identifier appears in more than one split."""

    for field in fields:
        sets = [
            set(part[field].astype(str).unique().tolist())
            for part in (train, validation, test)
        ]
        overlap = (sets[0] & sets[1]) | (sets[0] & sets[2]) | (sets[1] & sets[2])
        if overlap:
            preview = sorted(overlap)[:10]
            raise ValueError(f"Detected {field} leakage across splits: {preview}")


def constrained_group_split(
    frame: pd.DataFrame,
    *,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
    random_seed: int = 240727,
    group_column: str = "OperatingPointID",
    run_column: str = "RunID",
    label_column: str = "WindowFaultID",
    expected_labels: Sequence[int] = (0, 1, 2, 3, 4),
    max_attempts: int = 2000,
) -> SplitBundle:
    """Split whole operating-point groups while requiring class coverage."""

    required = {group_column, run_column, label_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Grouped split is missing columns: {missing}")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")

    working = frame.copy()
    working[group_column] = working[group_column].astype(str)
    working[run_column] = working[run_column].astype(str)

    run_group_counts = working.groupby(run_column, dropna=False)[group_column].nunique()
    invalid_runs = run_group_counts[run_group_counts.ne(1)]
    if not invalid_runs.empty:
        raise ValueError(
            "Each RunID must belong to exactly one operating point; invalid RunID(s): "
            f"{invalid_runs.index.astype(str).tolist()[:10]}"
        )

    expected = set(int(value) for value in expected_labels)
    groups_per_label = (
        working[[group_column, label_column]]
        .drop_duplicates()
        .groupby(label_column)[group_column]
        .nunique()
    )
    insufficient = {
        int(label): int(groups_per_label.get(label, 0))
        for label in expected
        if int(groups_per_label.get(label, 0)) < 3
    }
    if insufficient:
        raise ValueError(
            "Every class must occur in at least three independent groups to cover "
            f"train/validation/test; counts={insufficient}."
        )

    groups = np.asarray(sorted(working[group_column].unique().tolist()), dtype=object)
    train_count, validation_count, test_count = _allocate_counts(len(groups), ratios)

    for attempt in range(max_attempts):
        rng = np.random.default_rng(random_seed + attempt)
        shuffled = groups.copy()
        rng.shuffle(shuffled)
        train_groups = set(shuffled[:train_count].tolist())
        validation_groups = set(
            shuffled[train_count : train_count + validation_count].tolist()
        )
        test_groups = set(shuffled[-test_count:].tolist())

        train = working.loc[working[group_column].isin(train_groups)].copy()
        validation = working.loc[
            working[group_column].isin(validation_groups)
        ].copy()
        test = working.loc[working[group_column].isin(test_groups)].copy()

        if all(
            _labels_in(part, label_column) == expected
            for part in (train, validation, test)
        ):
            assert_no_group_leakage(
                train,
                validation,
                test,
                fields=(run_column, group_column),
            )
            assignment = pd.DataFrame(
                {
                    group_column: list(train_groups)
                    + list(validation_groups)
                    + list(test_groups),
                    "Split": ["train"] * len(train_groups)
                    + ["validation"] * len(validation_groups)
                    + ["test"] * len(test_groups),
                }
            ).sort_values(["Split", group_column], kind="stable")
            return SplitBundle(
                train=train.reset_index(drop=True),
                validation=validation.reset_index(drop=True),
                test=test.reset_index(drop=True),
                group_assignments=assignment.reset_index(drop=True),
            )

    raise ValueError(
        f"Unable to find a grouped split with all labels in every set after "
        f"{max_attempts} attempts. Increase independent operating points or "
        "change split ratios."
    )


def constrained_group_kfolds(
    frame: pd.DataFrame,
    *,
    n_splits: int = 6,
    random_seed: int = 240727,
    group_column: str = "OperatingPointID",
    run_column: str = "RunID",
    label_column: str = "WindowFaultID",
    expected_labels: Sequence[int] = (0, 1, 2, 3, 4),
    max_attempts: int = 2000,
) -> list[GroupCrossValidationFold]:
    """Build grouped folds whose test and validation parts cover every class.

    Each operating point is used once in an outer test fold and once in the
    following validation fold. All remaining groups form the training set. A
    constrained search is needed because some operating points can lack an
    eligible fault class even when the full simulation scenario exists.
    """

    required = {group_column, run_column, label_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Grouped cross-validation is missing columns: {missing}")

    working = frame.copy()
    working[group_column] = working[group_column].astype(str)
    working[run_column] = working[run_column].astype(str)
    groups = sorted(working[group_column].unique().tolist())
    if n_splits < 3:
        raise ValueError("Grouped cross-validation requires at least three folds.")
    if len(groups) < n_splits:
        raise ValueError(
            f"Grouped cross-validation has {len(groups)} groups but "
            f"n_splits={n_splits}."
        )
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")

    run_group_counts = working.groupby(run_column, dropna=False)[group_column].nunique()
    invalid_runs = run_group_counts[run_group_counts.ne(1)]
    if not invalid_runs.empty:
        raise ValueError(
            "Each RunID must belong to exactly one operating point; invalid RunID(s): "
            f"{invalid_runs.index.astype(str).tolist()[:10]}"
        )

    expected = set(int(value) for value in expected_labels)
    group_array = np.asarray(groups, dtype=object)
    for attempt in range(max_attempts):
        rng = np.random.default_rng(random_seed + attempt)
        shuffled = group_array.copy()
        rng.shuffle(shuffled)
        partitions = [
            tuple(str(value) for value in part.tolist())
            for part in np.array_split(shuffled, n_splits)
        ]
        if not all(
            _labels_in(
                working.loc[working[group_column].isin(part)],
                label_column,
            )
            == expected
            for part in partitions
        ):
            continue

        folds: list[GroupCrossValidationFold] = []
        for fold_index, test_groups in enumerate(partitions):
            validation_groups = partitions[(fold_index + 1) % n_splits]
            excluded = set(test_groups) | set(validation_groups)
            test = working.loc[
                working[group_column].isin(test_groups)
            ].copy()
            validation = working.loc[
                working[group_column].isin(validation_groups)
            ].copy()
            train = working.loc[
                ~working[group_column].isin(excluded)
            ].copy()
            if _labels_in(train, label_column) != expected:
                break
            assert_no_group_leakage(
                train,
                validation,
                test,
                fields=(run_column, group_column),
            )
            folds.append(
                GroupCrossValidationFold(
                    fold=fold_index + 1,
                    train=train.reset_index(drop=True),
                    validation=validation.reset_index(drop=True),
                    test=test.reset_index(drop=True),
                    validation_groups=validation_groups,
                    test_groups=test_groups,
                )
            )
        if len(folds) == n_splits:
            return folds

    raise ValueError(
        "Unable to construct grouped cross-validation folds with every expected "
        f"label in each test and validation fold after {max_attempts} attempts. "
        "Reduce n_splits or add independent operating points for sparse classes."
    )


def split_class_summary(
    bundle: SplitBundle, label_column: str = "WindowFaultID"
) -> pd.DataFrame:
    """Return window and independent-run counts by split and class."""

    rows: list[pd.DataFrame] = []
    for split_name, part in (
        ("train", bundle.train),
        ("validation", bundle.validation),
        ("test", bundle.test),
    ):
        windows = part.groupby(label_column).size().rename("WindowCount")
        runs = (
            part[["RunID", label_column]]
            .drop_duplicates()
            .groupby(label_column)
            .size()
            .rename("RunIDCount")
        )
        summary = pd.concat([windows, runs], axis=1).reset_index()
        summary.insert(0, "Split", split_name)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def run_split_assignments(bundle: SplitBundle) -> pd.DataFrame:
    """Map every run and operating point to its assigned split."""

    rows: list[pd.DataFrame] = []
    for split_name, part in (
        ("train", bundle.train),
        ("validation", bundle.validation),
        ("test", bundle.test),
    ):
        one = part[["RunID", "OperatingPointID"]].drop_duplicates().copy()
        one["Split"] = split_name
        rows.append(one)
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(["Split", "OperatingPointID", "RunID"], kind="stable")
