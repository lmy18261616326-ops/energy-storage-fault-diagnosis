from __future__ import annotations

from energy_fault_ml.features import (
    remove_unusable_training_features,
    select_feature_columns,
)
from tests.make_synthetic_dataset import make_synthetic_frame


def test_physics_features_exclude_metadata_truth_and_validation_columns() -> None:
    frame = make_synthetic_frame(group_count=6, windows_per_class=1)
    selected = select_feature_columns(
        frame,
        feature_set="physics_enhanced",
        include_mode_command=True,
    )

    assert "ModeCommand" in selected
    assert "IL_measMean" in selected
    assert "CurrentPairResidualRMS" in selected
    assert "BalancedPowerResidualRMS" in selected
    assert "WindowFaultID" not in selected
    assert "ScenarioFaultID" not in selected
    assert "FaultMagnitude" not in selected
    assert "Validation_ILSensorResidualRMS" not in selected
    assert "IL_trueMean" not in selected
    assert "PowerResidualRMS" not in selected


def test_statistical_ablation_and_training_only_constant_removal() -> None:
    frame = make_synthetic_frame(group_count=6, windows_per_class=1)
    selected = select_feature_columns(
        frame,
        feature_set="statistical",
        include_mode_command=False,
    )
    assert "IL_measMean" in selected
    assert "Vbus_measMean" in selected
    assert "CurrentPairResidualRMS" not in selected
    assert "BalancedPowerResidualRMS" not in selected
    assert "ModeCommand" not in selected

    physics = select_feature_columns(
        frame,
        feature_set="physics_enhanced",
        include_mode_command=False,
    )
    kept, removed = remove_unusable_training_features(frame, physics)
    assert "ConstantOnlineFeature" not in kept
    assert removed["ConstantOnlineFeature"] == "constant_in_training"
