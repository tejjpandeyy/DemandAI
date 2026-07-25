# Phase 10 — Changes & Verification

Documentation, environment, and deployment-readiness pass. No models were
retrained, no API contract was changed, and no Phase 1–9 source logic was
rewritten.

## Files created

| File | Purpose |
|---|---|
| `README.md` | **Overwrote** the Phase-1 placeholder with the full portfolio README (problem, architecture + Mermaid diagram, feature engineering, model comparison, final metrics, install/run/test, API endpoints, limitations, future work). |
| `docs/API.md` | Endpoint reference for `GET /`, `GET /health`, `POST /predict`, `POST /predict/batch`, with request/response examples that match the exact Pydantic schemas in `backend/app/schemas/predict.py`. |
| `docs/ML_METHODOLOGY.md` | Methodology + results: feature categories, leakage prevention, why chronological splitting, model-comparison table, final metrics, and why the test set was not used for selection. |
| `deploy/render.yaml` | Render Blueprint **template** for the FastAPI backend (not deployed). |
| `deploy/vercel.json` | Vercel config **template** for the Vite frontend (not deployed). |
| `deploy/README.md` | Deployment guide; explicitly states nothing is deployed and no real URLs exist. |
| `backend/.env.example` | Backend-specific env template reflecting variables that actually exist (`API_HOST`, `API_PORT`, `CORS_ALLOW_ORIGINS`). |
| `PHASE10_CHANGES.md` | This file. |

## Files modified

| File | Change | Why it was necessary |
|---|---|---|
| `backend/app/api/predict.py` | Added optional, env-gated CORS middleware (reads `CORS_ALLOW_ORIGINS`). | The spec allows a minimal CORS change if useful. Separate frontend/backend hosting genuinely needs it. The change is **additive and inert by default**: when the env var is unset, no middleware is added and behavior is byte-identical to Phase 8. No endpoint, schema, or response shape changed. |
| `.env.example` (root) | Rewrote stale Phase-1 scaffold content (PostgreSQL, `SAFETY_STOCK_FACTOR`) to reflect the features that actually exist (backend host/port, CORS, `VITE_API_BASE_URL`). | The old template documented a database and inventory engine that were never built; leaving it would mislead reviewers. No secrets; template only. |
| `.gitignore` (root) | Appended rules for generated pipeline outputs (`datasets/analytics/`, `datasets/features/`, `models/*.csv`), test caches, and `Thumbs.db`. Nothing was removed. | Keep generated artifacts out of Git; the spec asks for this. A comment documents how to intentionally commit the model for deployment. |

## What was deliberately NOT changed

- No model retraining; `best_model.joblib` untouched.
- No changes to ML logic, feature engineering, or the prediction service.
- No changes to any endpoint path, request schema, or response schema.
- No frontend source changes (the frontend already supports `VITE_API_BASE_URL`, so no code change was needed for configurable API URLs).
- No authentication, no database, no Docker (all remain future work).

## Commands actually run (in this environment) and their real results

> Environment note: this build environment has **no network access** and
> therefore **cannot install `fastapi` or `xgboost`**, and has **no npm
> registry access** (cannot run `npm install`). Results below report only
> what was actually executed. Commands that could not run are marked as
> NOT RUN — they were not faked.

### Backend tests (executed)

Runner: the five sandbox-runnable suites, executed with an XGBoost API
stand-in (scikit-learn `HistGradientBoostingRegressor`) for the two
model-training modules, since real `xgboost` is not installable here.

```
tests.test_data_preprocessing: 20 passed, 0 failed
tests.test_analytics:          14 passed, 0 failed
tests.test_feature_engineering:20 passed, 0 failed
tests.test_training:           14 passed, 0 failed
tests.test_model_comparison:   17 passed, 0 failed
TOTAL: 85 passed, 0 failed
```

- `tests/test_prediction_api.py` — **NOT RUN** here (requires `fastapi`,
  not installable in this environment). It passed in Phase 8 on the
  developer machine (17/17 per project history). Please re-run it locally
  with `pytest -v` where FastAPI is installed; the CORS change is inert
  by default and is not expected to affect it.

### Backend syntax / contract checks (executed)

- `python -m py_compile app/api/predict.py` → **compiled successfully**
  after the CORS change.
- Verified all four endpoint decorators (`/`, `/health`, `/predict`,
  `/predict/batch`) are unchanged.
- Verified the CORS block is env-gated (`if origins_env:`), so default
  behavior equals Phase 8.

### Frontend (partially executed)

- `npm install` — **NOT RUN** (no npm registry access here).
- `npm test` (Vitest) — **NOT RUN** here. Passed in Phase 9 on the
  developer machine (20/20 per project history). Please re-run locally.
- `npm run build` — **NOT RUN** here (needs installed deps). Please run
  locally to confirm the production build.
- Executed instead: transformed **all 18** frontend `.jsx/.js` files with
  the same esbuild that Vite/Vitest use — all transform cleanly (no syntax
  errors, JSX valid). Verified every relative import resolves to an
  existing file. Confirmed `package.json` defines the `dev`, `build`, and
  `test` scripts the README references.

### Documentation cross-checks (executed)

- API examples verified field-by-field against
  `backend/app/schemas/predict.py` (no invented fields).
- README doc links (`docs/API.md`, `docs/ML_METHODOLOGY.md`) resolve to
  real files.
- Mermaid architecture diagram block present in the README.
- All ML metrics in the docs use the numbers supplied in the Phase 10
  brief (Validation R² 0.7294, Test R² 0.7041, Test MAE 4.9338, Test RMSE
  7.6485) and the Phase 7 comparison table — none were invented.

## Phase 10 corrections (post-review)

The following corrections were applied after review. They touch only
documentation, `.gitignore`, and deployment templates — no API contract,
ML logic, model, frontend design, metrics, or tests were changed.

| # | Change | Files | Verification |
|---|---|---|---|
| 1 | Set README Author to **Tej Pandey**; removed the placeholder instruction. No email/LinkedIn/other contact details were invented. | `README.md` | Visual confirmation of the Author section. |
| 2 | Chose ONE model-deployment strategy: **track the single production artifact** `backend/models/best_model.joblib` via a precise `.gitignore` exception, while all other `.joblib` files remain ignored. Updated Render docs/template and the deploy checklist to match; removed the earlier `git add -f` manual-workaround note. | `.gitignore`, `deploy/README.md`, `deploy/render.yaml` | **Executed** `git check-ignore` in a scratch repo: `best_model.joblib` → TRACKED; `baseline_model.joblib` and other `.joblib` → IGNORED. The artifact is ~6 MB, within Git/GitHub limits. |
| 3 | Added a Windows PowerShell note to the backend Setup and Running-Locally sections: if execution policy blocks `venv\Scripts\activate` or `npm.ps1`, use `cmd.exe` or call the venv Python directly (`venv\Scripts\python.exe -m uvicorn ...`). No recommendation to weaken security settings. | `README.md` | Visual confirmation; commands use the documented direct-Python form. |

### Honesty note on the model artifact

The `.gitignore` exception makes `best_model.joblib` **eligible** to be
committed. Whether it is actually committed and pushed depends on the
developer running `git add backend/models/best_model.joblib && git commit`
in the real repository. This document does **not** claim the model has
been committed or deployed — only that the ignore rules now permit and
document tracking it as the single, reproducible strategy.

## Recommended local verification before merging

```bash
# Backend (with fastapi + xgboost installed)
cd backend && pytest -v

# Frontend
cd ../frontend && npm install && npm test && npm run build
```
