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
- `.codex/setup.sh`: dependency setup for Codex-managed worktrees.
- `.github/workflows/ci.yml`: pull request checks.

## Change map

- `data.py`: resumable Statcast collection.
- `features.py`: leakage-safe feature construction.
- `modeling.py`: Global candidate training and selection.
- `residual.py`: pooled contextual residual training and inference.
- `hybrid.py`: probability blending, eligibility, and exposure gates.
- `training.py`: offline training and model registry generation.
- `pipeline.py`: showcase artifact generation.
- `web/src/types.ts`: consumer-side artifact contract.

`docs/PROJECT_PLAN.md` describes the current design. Treat `docs/DECISIONS.md`
and `docs/EXPERIMENT_LOG.md` as append-only history; newer entries supersede
older decisions when they conflict.

## Verified commands

Run from the repository root unless noted:

- Python setup: `uv sync --extra dev`
- Codex worktree setup: `bash .codex/setup.sh`
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
- The user has given standing authorization to publish completed changes:
  after all required checks pass, commit only the task's files on a feature
  branch, push it, open or update a PR to `main`, and merge the PR.
- Do not merge when checks fail, the PR is conflicted, or unrelated changes
  would be included. Report the blocker instead.

## Done

A change is complete when targeted tests, Python lint, web tests, and the web
production build pass; generated probabilities sum to one; and `git diff` plus
`git status` contain no unrelated changes.
