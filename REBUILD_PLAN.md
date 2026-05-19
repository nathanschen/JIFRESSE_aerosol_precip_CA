# Summary of Changes for 2026/05/18 at 11:26pm

- The repo now targets a reproducible local conda environment named `py312_JIFRESSE`, defined by `environment.yml`, and the package metadata now targets Python `3.12`.
- The feature-build path has been adapted for the local machine by sampling the supervised benchmark timeline instead of forcing the full raw timeline through every baseline/model verify pass.
- Coordinate loading was hardened so benchmark arrays are canonicalized, time-filtered, transposed into consistent dimension order, and cast more conservatively for memory safety.
- The leading retained architecture is currently a dual-stage precipitation path: XGBoost occurrence probability plus persistence (`imerg_t`) for the amount field.
- The best retained dev `composite_score` improved from `-0.30603608859891945` to `-0.08884266647012698` at commit `8dc0108880509cb37a531af09417be3640f9d41f`.
- A blend sweep showed that, on the current sampled benchmark, pure persistence amount outperforms tested persistence/XGBoost amount blends.
- The current neural baselines still trail the retained hybrid path, and `temporal_unet` has shown instability during reruns, so neural stabilization remains an active next step.
- The verify path is mechanically valid but still slower than ideal because `python -m japca.baselines.run --suite milestone_01` dominates runtime.

# JAPCA Ground-Up Rebuild Repo Plan

## Summary

- Create a new standalone rebuild repo at `/Volumes/NathanSSD/JAPCA`.
- Keep the canonical raw and processed data in `/Volumes/NathanSSD/JIFRESSE` and reference them via configuration.
- Scope Milestone 1 to both variable selection and model bakeoff/tuning.
- Optimize the first benchmark stack for a local 14-inch 2021 MacBook Pro with Apple M1 Pro and 32 GB RAM.
- Use the repo-local `py312_JIFRESSE` conda environment as the canonical execution target.

## Milestone 1 implementation

- Scaffold a standalone Python repository with versioned configs, package modules, tests, and gitignored outputs/artifacts.
- Define a canonical variable inventory spanning IMERG history, GFS moisture/dynamics/instability fields, terrain, and aerosol species.
- Build derived feature families for wind speed, moisture flux, upslope flow, shear, and lagged precipitation summaries.
- Use sampled supervised benchmark slices for local-first baseline/model verification on the M1 Pro system, while keeping the benchmark configuration explicit in `configs/paths.yaml`.
- Lock temporal splits to:
  - leave-one-year-out screening across 2017-2022
  - train on 2017-2021
  - dev/tune on 2022
  - final untouched test on 2023
- Replace one-off aerosol scripts with grouped benchmark runs and parameterized feature groups.
- Benchmark climatology, persistence, hurdle XGBoost, small temporal U-Net, and ConvLSTM-U-Net.
- Track a JAPCA-native dual-stage path inspired by legacy JIFRESSE, where occurrence and amount can be handled separately instead of forcing a single monolithic predictor.
- Score all candidates with weather-specific threshold, spatial, calibration, and amount metrics, then rank them with a composite score.

## Required configs

- `configs/paths.yaml`
- `configs/variables.yaml`
- `configs/feature_groups.yaml`
- `configs/metrics.yaml`
- `configs/models/*.yaml`
- `configs/tuning/*.yaml`

## Current system details

- Canonical environment:
  - `environment.yml`
  - conda env name: `py312_JIFRESSE`
  - package Python target: `>=3.12,<3.13`
- Current benchmark runtime adjustments:
  - `configs/paths.yaml` includes local sampling counts for train/dev/test benchmark slices
  - feature building now selects supervised indices explicitly instead of materializing the full aligned time range for every verify run
  - data loading now canonicalizes dimension order and can filter to a supplied time index before interpolation
- Current leading retained baseline family:
  - occurrence: hurdle XGBoost probability
  - amount: persistence amount field from `imerg_t`
  - status: retained best dev candidate so far
- Current retained benchmark state:
  - baseline best dev `composite_score`: `-0.30603608859891945`
  - retained best dev `composite_score`: `-0.08884266647012698`
  - retained commit: `8dc0108880509cb37a531af09417be3640f9d41f`
- Current known weaknesses:
  - `temporal_unet` can become numerically unstable during reruns
  - verify speed is still bottlenecked by baseline execution time
  - the current best dual-stage path is strong empirically, but it is still a practical hybrid baseline rather than the final polished JAPCA architecture

## Current architectural direction

- Provisional leading architecture class: two-stage / dual-model precipitation forecasting
- Current evidence:
  - retained best score comes from separating occurrence probability from amount estimation
  - persistence amount beats tested persistence/XGBoost amount blends on the current benchmark
  - recent precipitation literature supports staged occurrence/amount or cascaded forecasting pipelines
- Near-term research priorities:
  - stabilize `temporal_unet` and `convlstm_unet`
  - reduce verify latency without losing ranking fidelity
  - decide whether the final JAPCA v1 should remain a boosted/persistence hybrid or absorb the two-stage logic into a more native learned architecture

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
- documented retained benchmark environment and reproducible local setup from files in the repo
