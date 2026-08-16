# Simulink experiment workspace

This folder contains only active models, experiment scripts and results, and
reference models. Old development files are preserved outside this folder in
`../archive/simulink_legacy_20260727`.

## Layout

- `models/`
  - `main_model_fd_v05_energyprotect.slx`: reproducible baseline for the
    five-class dataset and energy-protection experiments.
  - `main_model_fd_v06_switchobservability.slx`: switch-device observability
    extension used by high-resistance and specialist experiments.
  - `main_model_fd_v03_faultdiag.slx` and `main_model_fd_v04_faultdiag.slx`:
    historical comparison models, not default entry points.
- `experiments/sensor_bias/`
  - `scripts/`: sensor-bias dataset and classifier scripts.
  - `results/smoke/`: short validation results.
  - `results/full/`: complete experiment and classifier results.
- `experiments/current_performance/`
  - `scripts/`: current-loop performance calculations.
  - `results/`: current-loop analysis tables and workbook.
- `references/`: independent reference and example models.

## Default fault-dataset experiment

From MATLAB:

```matlab
cd(fullfile("simulink", "experiments", "sensor_bias", "scripts"));
cfg = signal_config();
result = collect_fault_dataset();
```

`signal_config` defaults to v05. The switch-observability smoke, pilot, and
validation scripts explicitly override the model to v06. Generated datasets
are written under `dataset_output*` and are intentionally excluded from Git.

From the repository root, `audit_main_model.m` audits v05 by default. Set the
environment variable `ENERGY_STORAGE_MODEL` to a model name (without `.slx`)
to audit another active model.

## Generated files

`slprj/`, `*.slxc`, `*.autosave`, dataset outputs, and inspection sidecars are
generated artifacts. They are excluded from source control and can be
regenerated. MATLAB Project cache and code-generation paths are placed under
the repository-local ignored `work/` directory.
