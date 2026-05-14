# JAPCA

JAPCA is a ground-up rebuild of the California 6-hour precipitation forecasting workflow that previously lived in JIFRESSE. The repository is designed around two first-order goals:

1. identify which variables matter most for precipitation forecast quality
2. select and tune the strongest local-first model family using weather-specific metrics

## Milestone 1

Milestone 1 implements:

- a canonical dataset manifest that points to the existing JIFRESSE data on `NathanSSD`
- feature building and derived meteorological features on the 140x150 benchmark grid
- blocked temporal splits for feature screening, model development, and final test
- fast tabular baselines and a neural model bakeoff scaffold
- weather-specific metrics, leaderboard generation, and report export

## Repository layout

- `REBUILD_PLAN.md`: canonical milestone plan
- `configs/`: versioned paths, variables, feature groups, metrics, models, tuning configs
- `src/japca/`: package implementation
- `tests/`: unit and smoke coverage
- `outputs/`: generated reports, studies, feature sets, metrics
- `artifacts/`: model checkpoints and temporary artifacts

## Core commands

```bash
python -m japca.data.audit
python -m japca.features.build --feature-set milestone_01_full
python -m japca.baselines.run --suite milestone_01
python -m japca.models.run --model temporal_unet --config configs/models/temporal_unet.yaml
python -m japca.tuning.run --model temporal_unet --study milestone_01_temporal_unet
python -m japca.reports.generate --suite milestone_01
```

## Notes

- JAPCA does not duplicate canonical JIFRESSE raw data.
- The tuning entrypoint requires `optuna`; the code will emit a clear error if it is unavailable at runtime.
