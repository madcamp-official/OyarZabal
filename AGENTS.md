# OyarZabal working guide

## Purpose

OyarZabal builds leakage-safe MLB next-pitch predictions offline and renders
completed games through a static React replay app.

## Important directories

- `ml/oyarzabal`: data, feature, model, metric, and artifact code.
- `ml/tests`: Python regression tests.
- `web/src`: replay UI and state.
- `web/public/data`: generated, reviewable demo artifacts.
- `docs`: product decisions and experiment history.

## Verified commands

Run from the repository root unless noted:

- Python setup: `uv sync --extra dev`
- Python tests: `uv run pytest -q`
- Python lint: `uv run ruff check .`
- Build demo data: `uv run oyarzabal-build-demo --help`
- Web setup: `cd web && npm install`
- Web tests: `cd web && npm test`
- Web build: `cd web && npm run build`

## Constraints

- A prediction may use only information known before the target pitch.
- Lagged and rolling features must exclude the current pitch.
- Do not select showcase games based on model performance.
- Keep raw Statcast, model checkpoints, caches, and raw logs out of Git.
- Do not label a historical showcase as a frozen 2026 holdout result.
- Do not commit or push without explicit user authorization.

## Done

A change is complete when targeted tests, Python lint, web tests, and the web
production build pass; generated probabilities sum to one; and `git diff` plus
`git status` contain no unrelated changes.
