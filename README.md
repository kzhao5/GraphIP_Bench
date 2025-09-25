# GraphIP_Bench (anonymous)

A minimal, anonymized benchmark scaffolding for graph model integrity and property inference studies. This public repo intentionally omits datasets and generated outputs; it contains only the essential code to run small benchmark examples.

## What this contains
- Minimal source code structure under `src/`
- A starter requirements.txt for runtime
- A strict `.gitignore` to exclude datasets, artifacts, and outputs

## What’s intentionally excluded
- Datasets (place under `data/` or `datasets/` locally)
- Any generated results, logs, or figures

## Quick start
1. Create a fresh environment and install deps:
   - `python -m venv .venv && . .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Put your datasets under `data/` (ignored by git).
3. Run a small smoke test (replace with your entry-point once code is added):
   - `python -m src.example_run --dataset Cora --device cpu`

## Notes
- Keep training logs/results in `outputs/` (ignored by git).
- Extend `src/` with your modules (datasets, models, utils) and a small CLI runner.
