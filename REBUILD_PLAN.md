# JAPCA Ground-Up Rebuild Repo Plan

## Summary

- Create a new standalone rebuild repo at `/Volumes/NathanSSD/JAPCA`.
- Keep the canonical raw and processed data in `/Volumes/NathanSSD/JIFRESSE` and reference them via configuration.
- Scope Milestone 1 to both variable selection and model bakeoff/tuning.
- Optimize the first benchmark stack for a local 14-inch 2021 MacBook Pro with Apple M1 Pro and 32 GB RAM.

## Milestone 1 implementation

- Scaffold a standalone Python repository with versioned configs, package modules, tests, and gitignored outputs/artifacts.
- Define a canonical variable inventory spanning IMERG history, GFS moisture/dynamics/instability fields, terrain, and aerosol species.
- Build derived feature families for wind speed, moisture flux, upslope flow, shear, and lagged precipitation summaries.
- Lock temporal splits to:
  - leave-one-year-out screening across 2017-2022
  - train on 2017-2021
  - dev/tune on 2022
  - final untouched test on 2023
- Replace one-off aerosol scripts with grouped benchmark runs and parameterized feature groups.
- Benchmark climatology, persistence, hurdle XGBoost, small temporal U-Net, and ConvLSTM-U-Net.
- Score all candidates with weather-specific threshold, spatial, calibration, and amount metrics, then rank them with a composite score.

## Required configs

- `configs/paths.yaml`
- `configs/variables.yaml`
- `configs/feature_groups.yaml`
- `configs/metrics.yaml`
- `configs/models/*.yaml`
- `configs/tuning/*.yaml`

## Required CLI entrypoints

- `python -m japca.data.audit`
- `python -m japca.features.build --feature-set <name>`
- `python -m japca.baselines.run --suite milestone_01`
- `python -m japca.models.run --model <family> --config <file>`
- `python -m japca.tuning.run --model <family> --study <name>`
- `python -m japca.reports.generate --suite milestone_01`

## Acceptance criteria

- Markdown leaderboard and summary report under `outputs/`
- ranked feature-group keep/drop decisions
- selected retained feature set
- ranked model-family comparison on 2022 dev and 2023 final test
- chosen v1 architecture and tuned config with seed-stability reruns
