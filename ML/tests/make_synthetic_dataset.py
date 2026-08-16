"""Create a small five-class dataset for tests and pipeline smoke runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic_frame(
    *,
    group_count: int = 15,
    windows_per_class: int = 4,
    random_seed: int = 240727,
) -> pd.DataFrame:
    """Return separable windows with realistic metadata and forbidden columns."""

    if group_count < 3:
        raise ValueError("group_count must be at least 3.")
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    for group_index in range(group_count):
        operating_point = f"op_{group_index:04d}"
        mode = group_index % 3
        for label in range(5):
            run_id = f"run_{group_index:04d}_{label}"
            for window_index in range(windows_per_class):
                jitter = rng.normal(0.0, 0.04)
                rows.append(
                    {
                        "RunID": run_id,
                        "OperatingPointID": operating_point,
                        "WindowID": window_index,
                        "WindowStart": 0.005 * window_index,
                        "WindowEnd": 0.005 * window_index + 0.010,
                        "RandomSeed": random_seed + group_index,
                        "SOCInit": 20 + 5 * group_index,
                        "Rload": 200.0,
                        "ScenarioFaultID": label,
                        "WindowFaultID": label,
                        "FaultName": f"class_{label}",
                        "FaultMagnitude": float(label),
                        "FaultStartTime": 0.5,
                        "FaultEndTime": 0.7,
                        "FaultActiveRatio": 1.0 if label else 0.0,
                        "FaultObservableRatio": 1.0 if label else 0.0,
                        "IsTransitionWindow": 0,
                        "IsTrainingEligible": 1,
                        "SampleWeight": 1.0,
                        "ModeCommand": mode,
                        "IL_measMean": 2.5 * label + 0.2 * mode + jitter,
                        "IL_measStd": 0.1 + 0.3 * label + abs(jitter),
                        "Ibat_measRMS": 1.5 * label - 0.1 * mode + jitter,
                        "Vbus_measMean": 400.0 + 4.0 * label + jitter,
                        "Vbat_measStd": 0.02 + 0.15 * label + abs(jitter),
                        "CurrentPairResidualRMS": 0.5 * label + abs(jitter),
                        "CurrentErrorRMSE": 0.8 * label + abs(jitter),
                        "BalancedPowerResidualRMS": 3.0 * label + jitter,
                        "DutySatRatio": 0.05 * label,
                        "PowerResidualRMS": 999.0 * label,
                        "Validation_ILSensorResidualRMS": 100.0 * label,
                        "IL_trueMean": 50.0 * label,
                        "ConstantOnlineFeature": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="Destination CSV path.")
    parser.add_argument("--groups", type=int, default=15)
    parser.add_argument("--windows-per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=240727)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic_frame(
        group_count=args.groups,
        windows_per_class=args.windows_per_class,
        random_seed=args.seed,
    ).to_csv(output, index=False)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
