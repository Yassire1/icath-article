# AGENTS.md

## Read First
- Runtime truth lives in `scripts/pipeline_config.py`, `scripts/run_pipeline.py`, and `scripts/step_*.py`.
- Treat `README.md`, `EXECUTION_MAP.md`, `PRE_IMPLEMENTATION_GUIDE.md`, `config/*.yaml`, `scripts/setup.sh`, and `src/experiments/*.py` as partial/historical context. They still mention stale scope like `mimii`, `cross_domain`, CUDA defaults, and larger model IDs.

## Scope
- Active pipeline scope is `cmapss`, `wind_scada`, and `phm_milling`.
- Active transfer outputs live in `results/cross_condition/`, not `results/cross_domain/`.
- Runtime device selection now lives in `scripts/pipeline_config.py`: it uses `ICATH_DEVICE` when set, otherwise auto-selects `cuda` when PyTorch can see a GPU.
- The live model IDs are the lighter runtime ones from `scripts/pipeline_config.py`: `AutonLab/MOMENT-1-small`, `amazon/chronos-t5-tiny`, and `time-series-foundation-models/Lag-Llama`.

## Setup
- Run Python entrypoints from the repo root; the scripts assume root-relative paths and inject the project root into `sys.path`.
- On this repo snapshot, the working interpreter is the repo-local virtualenv at `.venv/bin/python`; plain `python` may be missing from `PATH`.
- `environment.yml` and `requirements.txt` are not sufficient for every wrapper. Steps `3-6` also depend on model packages imported at runtime: `momentfm`, `chronos-forecasting`, and `lag-llama`.
- Model caches are redirected by `scripts/pipeline_config.py` to `/mnt/datasets/icath-cache` when that mount exists, otherwise to repo-local `.cache/`.
- Kaggle auth is auto-loaded from the repo-root `.env` using keys `kaggle_username` and `kaggle_api`; the code can materialize `~/.kaggle/kaggle.json` from those values.
- Treat the root `.env` as secret. `.gitignore` only ignores `.env/` as a directory, not the `.env` file itself.

## Commands
- Full pipeline: `.venv/bin/python scripts/run_pipeline.py`
- Resume from a failed step: `.venv/bin/python scripts/run_pipeline.py --from <step>`
- Focus a rerun: `.venv/bin/python scripts/run_pipeline.py --only <steps...>`
- Fail fast on missing datasets during download: `.venv/bin/python scripts/step_01_download.py --strict`
- Real dependency order: `1 -> 2 -> {3,4,5,6 in any order} -> 7`; step `8` reads per-run JSONs directly and does not require the CSVs from step `7`.
- Step `3` and step `4` only use `cmapss/FD001`, `wind_scada`, and `phm_milling`; step `5` is the only path that needs all four C-MAPSS subsets.

## Caching
- Step `2` writes `processed_data.pt` under `data/processed/...`; PHM Milling also writes `data/processed/phm_milling/cut_level_features.csv`.
- Reruns are heavily cached. If you change preprocessing or experiment logic, delete the specific `processed_data.pt` or result JSONs you need to refresh.
- Cache invalidation is uneven: step `2`, `5`, and `6` skip on file existence only; step `3` checks both `result_version` and `metric_version`; step `4` checks `metric_version` only.
- Progress and logs are tracked in `results/manifest.json` and `results/pipeline.log`.

## Evaluation Quirks
- `PatchTST` is a supervised baseline here. Step `3` trains it on the training split before evaluation; do not describe it as true zero-shot.
- Step `4` now runs all four models in scope: `moment`, `chronos`, `lag_llama`, and `patchtst`. Chronos few-shot is implemented here as a lightweight calibration head on top of zero-shot forecasts, not native Chronos finetuning.
- Despite older docs discussing RUL and anomaly metrics, the active step scripts score step `3` and `4` with forecasting metrics for all datasets, and step `5` only writes `mae` and `rmse`.
- Runtime knobs are env-overridable through `ICATH_*` variables in `scripts/pipeline_config.py`; the checked-in defaults are now full-run oriented rather than smoke-test sized.

## Paper
- Paper entrypoint is `paper/main.tex`; edit `paper/sections/*.tex` for content changes.
- The checked-in paper uses `IEEEtran` plus `IEEEtran` bibliography style. Ignore the MDPI wording in `paper/instructions/submission-guides.md` unless the user explicitly asks to switch templates.
- No paper build script, test suite, CI workflow, or lint/formatter config is checked in. Verification is by running the relevant pipeline step(s) or a manual LaTeX build.
