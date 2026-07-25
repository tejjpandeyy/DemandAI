"""FastAPI prediction API for DemandAI (Phase 8).

The model is loaded EXACTLY ONCE, during application startup (lifespan
handler), and shared via app.state + dependency injection -- never
reloaded per request.

Run locally (from the backend/ directory):
    uvicorn app.api.predict:app --reload

Note: this module hosts the app factory for Phase 8; when the full
backend is assembled (Phase 12/13), create_app() moves to app/main.py
and this file keeps only the router.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request

from app.schemas.predict import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    RootResponse,
)
from app.services.prediction_service import (
    DEFAULT_MODEL_PATH,
    ModelNotLoadedError,
    ModelService,
    PredictionError,
)

router = APIRouter()


def get_service(request: Request) -> ModelService:
    """Dependency: the single ModelService created at startup."""
    return request.app.state.model_service


@router.get("/", response_model=RootResponse)
def root() -> dict:
    return {"service": "DemandAI Prediction API", "status": "running"}


@router.get("/health", response_model=HealthResponse)
def health(service: ModelService = Depends(get_service)) -> dict:
    return {"status": "healthy", "model_loaded": service.loaded}


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest,
            service: ModelService = Depends(get_service)) -> dict:
    try:
        return service.predict_single(payload.model_dump())
    except PredictionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest,
                  service: ModelService = Depends(get_service)) -> dict:
    try:
        return service.predict_batch(
            [r.model_dump() for r in payload.requests]
        )
    except PredictionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def create_app(model_path: Path | str = DEFAULT_MODEL_PATH) -> FastAPI:
    """Build the FastAPI app; the model loads once in the lifespan."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model_service = ModelService(model_path).load_model()
        yield
        app.state.model_service = None

    app = FastAPI(
        title="DemandAI Prediction API",
        version="0.8.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
