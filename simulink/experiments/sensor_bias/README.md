# Sensor-bias experiment

Default baseline: `../../models/main_model_fd_v05_energyprotect.slx`

Switch-observability extension:
`../../models/main_model_fd_v06_switchobservability.slx`. v03/v04 are retained
only for historical comparison. The centralized `scripts/signal_config.m`
selects v05 by default; specialist v06 scripts override the model explicitly.

## Sensor fault identifiers

| ID | Fault |
|---:|---|
| 0 | Healthy / allowed load step |
| 2 | DC-bus voltage sensor bias |
| 3 | Inductor-current sensor bias |
| 8 | Battery-voltage sensor bias |
| 9 | Battery-current sensor bias |

Existing non-sensor IDs 1 and 4-7 remain reserved by v03.

## True and measured logs

| True | Measured |
|---|---|
| `log_Vbus_true` | `log_Vbus` |
| `log_Vbat_true` | `log_Vbat` |
| `log_IL_true` | `log_I_L` |
| `log_Ibat_true` | `log_Ibat` |

True signals are validation-only. Do not use true-minus-measured quantities
as online classifier predictors.

## Workflow

1. Inspect `scripts/signal_config.m` and confirm the model, cases, random seed,
   output root, and `overwritePolicy`.
2. Run `result = collect_fault_dataset();` from the `scripts` folder.
3. Inspect the generated `combined/dataset_report.txt` and failed-run tables.
4. Confirm true-minus-measured values are used for injection validation only,
   never as online predictors.
5. Split by `OperatingPointID`/Run and freeze the model before blind-data
   generation.

The legacy v04 smoke scripts remain under `scripts/_legacy` and are not the
current experiment entry point.

## Important interpretation

`CurrentTrackingError = IL_meas - Iref` is a control error, not an
independent sensor residual. `CurrentPairResidual = Ibat_meas + IL_meas`
is useful for detecting disagreement between the two current sensors but
cannot identify which sensor is biased without an observer or independent
power-balance residual.
