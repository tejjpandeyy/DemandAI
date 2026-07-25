# Deployment Guide

> **Status: NOT deployed.** The files in this folder are configuration
> templates prepared for deployment. No live deployment has been
> performed, and no production URLs exist yet. Do not treat any URL in
> these templates as real.

DemandAI is two independently deployable pieces: a FastAPI backend and a
static React/Vite frontend.

## Backend → Render (or any Python host)

`deploy/render.yaml` is a Render Blueprint template.

1. Copy it to the repo root as `render.yaml` (or point a Blueprint at
   `deploy/render.yaml`).
2. In Render, create a Blueprint from your GitHub repo.
3. Set `CORS_ALLOW_ORIGINS` to your deployed frontend origin.

**Model artifact.** The service needs `backend/models/best_model.joblib`.
This single production artifact **is tracked in Git** (via an explicit
exception in `.gitignore`; it is ~6 MB, well within Git/GitHub limits),
so it is present on a fresh clone and no extra build step is required to
obtain it. All other `.joblib` files (the baseline and any experimental
artifacts) remain ignored.

The start command binds the port Render provides:

```
uvicorn app.api.predict:app --host 0.0.0.0 --port $PORT
```

## Frontend → Vercel (or any static host)

`deploy/vercel.json` is a Vercel config template.

1. Copy it to `frontend/vercel.json`.
2. Import the repo in Vercel with the project root set to `frontend/`.
3. Set the environment variable `VITE_API_BASE_URL` to your deployed
   backend URL (e.g. `https://<your-service>.onrender.com`).
4. Deploy. Vercel runs `npm run build` and serves `dist/`.

## Production configuration checklist

- [ ] `VITE_API_BASE_URL` points at the real backend (never `localhost`).
- [ ] `CORS_ALLOW_ORIGINS` on the backend includes the real frontend origin.
- [ ] The model artifact `backend/models/best_model.joblib` is present
      (it is tracked in Git, so it ships with the repo automatically).
- [ ] No secrets are committed (`.env` files are git-ignored).

## Local development needs none of this

Locally, Vite proxies `/api` to `http://127.0.0.1:8000`, so the frontend
and backend behave as same-origin and no CORS or production env vars are
required. See the root README's "Running Locally" section.
