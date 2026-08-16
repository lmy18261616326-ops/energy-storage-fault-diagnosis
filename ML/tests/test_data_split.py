from __future__ import annotations

import pytest

from energy_fault_ml.data import constrained_group_kfolds, constrained_group_split
from tests.make_synthetic_dataset import make_synthetic_frame


def test_group_split_is_leakage_free_and_has_every_class() -> None:
    frame = make_synthetic_frame(group_count=15, windows_per_class=2)
    bundle = constrained_group_split(
        frame,
        ratios=(0.60, 0.20, 0.20),
        random_seed=240727,
        expected_labels=(0, 1, 2, 3, 4),
    )

    expected = {0, 1, 2, 3, 4}
    for part in (bundle.train, bundle.validation, bundle.test):
        assert set(part["WindowFaultID"].unique()) == expected

    for field in ("RunID", "OperatingPointID"):
        train_ids = set(bundle.train[field].unique())
        validation_ids = set(bundle.validation[field].unique())
        test_ids = set(bundle.test[field].unique())
        assert train_ids.isdisjoint(validation_ids)
        assert train_ids.isdisjoint(test_ids)
        assert validation_ids.isdisjoint(test_ids)


def test_group_split_rejects_run_assigned_to_two_operating_points() -> None:
    frame = make_synthetic_frame(group_count=6, windows_per_class=1)
    frame.loc[1, "RunID"] = frame.loc[0, "RunID"]
    frame.loc[1, "OperatingPointID"] = "different_operating_point"

    with pytest.raises(ValueError, match="exactly one operating point"):
        constrained_group_split(frame)


def test_group_kfold_uses_each_group_once_without_leakage() -> None:
    frame = make_synthetic_frame(group_count=6, windows_per_class=2)
    folds = constrained_group_kfolds(frame, n_splits=3)

    assert len(folds) == 3
    assert {
        group for fold in folds for group in fold.test_groups
    } == set(frame["OperatingPointID"].unique())
    assert {
        group for fold in folds for group in fold.validation_groups
    } == set(frame["OperatingPointID"].unique())

    expected = {0, 1, 2, 3, 4}
    for fold in folds:
        assert set(fold.test_groups).isdisjoint(fold.validation_groups)
        assert set(fold.train["WindowFaultID"].unique()) == expected
        assert set(fold.validation["WindowFaultID"].unique()) == expected
        assert set(fold.test["WindowFaultID"].unique()) == expected
        train_groups = set(fold.train["OperatingPointID"].unique())
        validation_groups = set(fold.validation["OperatingPointID"].unique())
        test_groups = set(fold.test["OperatingPointID"].unique())
        assert train_groups.isdisjoint(validation_groups)
        assert train_groups.isdisjoint(test_groups)
        assert validation_groups.isdisjoint(test_groups)


def test_group_kfold_pairs_groups_when_some_lack_sparse_classes() -> None:
    frame = make_synthetic_frame(group_count=6, windows_per_class=2)
    sparse_groups = {"op_0000", "op_0003"}
    frame = frame.loc[
        ~(
            frame["OperatingPointID"].isin(sparse_groups)
            & frame["WindowFaultID"].isin({3, 4})
        )
    ].reset_index(drop=True)

    folds = constrained_group_kfolds(
        frame,
        n_splits=3,
        random_seed=240727,
        max_attempts=200,
    )

    expected = {0, 1, 2, 3, 4}
    assert len(folds) == 3
    for fold in folds:
        assert set(fold.validation["WindowFaultID"].unique()) == expected
        assert set(fold.test["WindowFaultID"].unique()) == expected
