# DemandAI Frontend (Phase 9)

React + Vite dashboard consuming the DemandAI FastAPI backend.

## Setup

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173 (proxies /api -> http://127.0.0.1:8000)
```

Start the backend first (from `backend/`): `uvicorn app.api.predict:app --reload`

## Test

```bash
npm test           # Vitest + React Testing Library
```

## Configuration

Copy `.env.example` to `.env` to override `VITE_API_BASE_URL` (defaults to `/api`).

## Structure

- `src/services/api.js` — the only module that talks to the backend
- `src/hooks/` — `useHealth` (30s polling), `useHistory` (session history)
- `src/components/` — reusable UI + feature components
- `src/pages/` — Dashboard, Single, Batch, History
- `src/test/` — Vitest suite
