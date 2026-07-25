# DemandAI API Documentation

FastAPI service for retail demand prediction. The model bundle is loaded **once** at application startup and shared across requests.

When the server is running, interactive documentation is available at:

- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`
- **OpenAPI schema:** `http://127.0.0.1:8000/openapi.json`

Base URL in local development: `http://127.0.0.1:8000`. All request/response bodies are JSON.

---

## GET /

Service metadata.

**Response 200**

```json
{
  "service": "DemandAI Prediction API",
  "status": "running"
}
```

---

## GET /health

Liveness plus whether the model was successfully loaded.

**Response 200**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## POST /predict

Predict demand for a single product on a single date.

### Request body

| Field | Type | Required | Constraints |
|---|---|---|---|
| `date` | string | yes | Strict `YYYY-MM-DD` (e.g. `2016-03-01`). |
| `product_id` | string | yes | Non-empty. |
| `price` | number | yes | Must be `> 0`. |
| `snap_day` | integer | no (default 0) | `0` or `1`. |
| `holiday` | integer | no (default 0) | `0` or `1`. |
| `has_named_event` | integer | no (default 0) | `0` or `1`. |
| `features` | array | no (default `[]`) | List of `{ "name": string, "value": number \| null }`. Feature names must be unique. `null` means "unknown" (treated as missing). |

Calendar features (year, month, day-of-week, weekend, etc.) are derived server-side from `date`. History-dependent features (`lag_*`, `rolling_*`, `expanding_mean`, `price_change`, ...) are supplied through `features`; any omitted feature is treated as missing.

**Example request**

```json
{
  "date": "2016-03-01",
  "product_id": "FOODS_3_090",
  "price": 3.48,
  "snap_day": 1,
  "holiday": 0,
  "has_named_event": 0,
  "features": [
    { "name": "lag_1", "value": 12.0 },
    { "name": "lag_7", "value": 9.0 },
    { "name": "rolling_mean_7", "value": 10.5 }
  ]
}
```

### Response 200

| Field | Type | Description |
|---|---|---|
| `product_id` | string | Echoed from the request. |
| `date` | string | Echoed from the request. |
| `predicted_sales` | number | Predicted units (clipped at 0). |
| `processing_time_ms` | number | Server-side prediction time. |
| `model_type` | string | The active model, e.g. `HistGradientBoosting`. |

```json
{
  "product_id": "FOODS_3_090",
  "date": "2016-03-01",
  "predicted_sales": 10.75,
  "processing_time_ms": 3.2,
  "model_type": "HistGradientBoosting"
}
```

### Error responses

- **422 Unprocessable Entity** — validation failure (missing field, wrong type, non-positive price, malformed date, duplicate feature names, or an unknown feature name). The body contains a `detail` field describing the problem.
- **503 Service Unavailable** — the model is not loaded.

---

## POST /predict/batch

Predict demand for many requests in one call.

### Request body

| Field | Type | Required | Constraints |
|---|---|---|---|
| `requests` | array | yes | At least one `PredictionRequest` (same shape as `POST /predict`). |

**Example request**

```json
{
  "requests": [
    {
      "date": "2016-03-01",
      "product_id": "FOODS_3_090",
      "price": 3.48,
      "features": [{ "name": "lag_1", "value": 12.0 }]
    },
    {
      "date": "2016-03-02",
      "product_id": "FOODS_3_120",
      "price": 2.98,
      "features": [{ "name": "lag_1", "value": 4.0 }]
    }
  ]
}
```

### Response 200

| Field | Type | Description |
|---|---|---|
| `count` | integer | Number of predictions returned. |
| `processing_time_ms` | number | Total server-side time. |
| `predictions` | array | List of `{ product_id, date, predicted_sales }`. |

```json
{
  "count": 2,
  "processing_time_ms": 5.1,
  "predictions": [
    { "product_id": "FOODS_3_090", "date": "2016-03-01", "predicted_sales": 10.75 },
    { "product_id": "FOODS_3_120", "date": "2016-03-02", "predicted_sales": 3.21 }
  ]
}
```

### Error responses

- **422 Unprocessable Entity** — empty `requests` list, or any single request fails validation (the whole batch is rejected).
- **503 Service Unavailable** — the model is not loaded.

---

## Notes

- `predicted_sales` is clipped at `0`, since negative demand is not physically meaningful.
- CSV batch input in the dashboard is converted to the `requests` array client-side before calling this endpoint; the backend only accepts JSON.
- This documentation reflects the Phase 8 API contract, which Phase 10 does not change.
