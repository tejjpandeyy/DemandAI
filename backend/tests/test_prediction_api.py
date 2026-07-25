"""Pytest suite for the Phase 8 prediction API.

A tiny real model bundle is trained once (module scope) and served by a
TestClient built through create_app(). The client is used as a context
manager so the lifespan runs and the model loads exactly once -- which
one test then proves via load_count.

Run (from the backend/ directory):
    pytest tests/test_prediction_api.py -v
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.predict import create_app

BUNDLE_FEATURES = [
    "price", "snap_day", "holiday", "has_named_event",
    "year", "month", "day_of_week", "is_weekend",
    "lag_1", "lag_7", "rolling_mean_7",
]


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory):
    """Train and save a tiny real model bundle for the API to serve."""
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((300, len(BUNDLE_FEATURES))) * 10,
                     columns=BUNDLE_FEATURES)
    y = X["lag_1"] * 2 + X["snap_day"] + rng.normal(0, 0.5, 300)
    model = HistGradientBoostingRegressor(max_iter=30, random_state=0)
    model.fit(X, y)

    path = tmp_path_factory.mktemp("models") / "best_model.joblib"
    joblib.dump({
        "model": model,
        "feature_names": BUNDLE_FEATURES,
        "model_type": "HistGradientBoosting",
        "params": None,
        "metrics": {},
        "target": "sales_quantity",
        "trained_at": "test",
    }, path)
    return path


@pytest.fixture(scope="module")
def app(bundle_path):
    return create_app(model_path=bundle_path)


@pytest.fixture(scope="module")
def client(app):
    # Context manager form runs the lifespan => model loads at startup.
    with TestClient(app) as test_client:
        yield test_client


def valid_payload(**overrides) -> dict:
    payload = {
        "date": "2016-03-01",
        "product_id": "FOODS_3_090",
        "price": 3.48,
        "snap_day": 1,
        "holiday": 0,
        "has_named_event": 0,
        "features": [
            {"name": "lag_1", "value": 12.0},
            {"name": "lag_7", "value": 9.0},
            {"name": "rolling_mean_7", "value": 10.5},
        ],
    }
    payload.update(overrides)
    return payload


# ------------------------------ Basic endpoints ---------------------------

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "service": "DemandAI Prediction API", "status": "running"
    }


def test_health_endpoint_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


# ------------------------------ Single prediction -------------------------

def test_single_prediction_success(client):
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["predicted_sales"], float)
    assert body["product_id"] == "FOODS_3_090"
    assert body["model_type"] == "HistGradientBoosting"


def test_prediction_response_format(client):
    body = client.post("/predict", json=valid_payload()).json()
    assert set(body) == {"product_id", "date", "predicted_sales",
                         "processing_time_ms", "model_type"}
    assert body["processing_time_ms"] >= 0


def test_predicted_sales_never_negative(client):
    # Force inputs that could push a regressor below zero.
    payload = valid_payload(features=[
        {"name": "lag_1", "value": -50.0},
        {"name": "lag_7", "value": -50.0},
    ])
    body = client.post("/predict", json=payload).json()
    assert body["predicted_sales"] >= 0.0


def test_missing_engineered_features_are_allowed_as_nan(client):
    # No features list at all: NaN-tolerant model must still predict.
    response = client.post("/predict", json=valid_payload(features=[]))
    assert response.status_code == 200


def test_deterministic_same_input_same_output(client):
    first = client.post("/predict", json=valid_payload()).json()
    second = client.post("/predict", json=valid_payload()).json()
    assert first["predicted_sales"] == second["predicted_sales"]


# ------------------------------ Validation rejections ---------------------

def test_missing_required_field_rejected(client):
    payload = valid_payload()
    del payload["price"]
    assert client.post("/predict", json=payload).status_code == 422


def test_wrong_type_rejected(client):
    payload = valid_payload(price="not-a-number")
    assert client.post("/predict", json=payload).status_code == 422


def test_negative_price_rejected(client):
    payload = valid_payload(price=-2.5)
    assert client.post("/predict", json=payload).status_code == 422


def test_invalid_date_rejected(client):
    for bad in ("2016-13-45", "March 1st", "20160301"):
        response = client.post("/predict", json=valid_payload(date=bad))
        assert response.status_code == 422, bad


def test_duplicate_feature_names_rejected(client):
    payload = valid_payload(features=[
        {"name": "lag_1", "value": 5.0},
        {"name": "lag_1", "value": 7.0},
    ])
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "duplicate" in response.text.lower()


def test_unknown_feature_name_rejected(client):
    payload = valid_payload(features=[{"name": "not_a_feature",
                                       "value": 1.0}])
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "unknown feature" in response.json()["detail"]


# ------------------------------ Batch -------------------------------------

def test_batch_prediction_success(client):
    batch = {"requests": [valid_payload(product_id=f"P{i}")
                          for i in range(3)]}
    response = client.post("/predict/batch", json=batch)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3
    assert body["predictions"][1]["product_id"] == "P1"
    assert set(body["predictions"][0]) == {"product_id", "date",
                                           "predicted_sales"}


def test_empty_batch_rejected(client):
    response = client.post("/predict/batch", json={"requests": []})
    assert response.status_code == 422


def test_batch_with_one_invalid_item_rejected(client):
    batch = {"requests": [valid_payload(),
                          valid_payload(price=-1.0)]}
    assert client.post("/predict/batch", json=batch).status_code == 422


# ------------------------------ Single loading ----------------------------

def test_model_loaded_exactly_once_across_requests(client, app):
    for _ in range(4):
        assert client.post("/predict",
                           json=valid_payload()).status_code == 200
    assert app.state.model_service.load_count == 1
