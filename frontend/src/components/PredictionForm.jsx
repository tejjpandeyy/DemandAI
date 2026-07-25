import { useState } from "react";
import { postPrediction } from "../services/api";
import { buildFeatureList, fmt, validatePredictionForm } from "../utils/format";
import { Alert, Card, Spinner } from "./ui";

const INITIAL = {
  date: "",
  product_id: "",
  price: "",
  snap_day: 0,
  holiday: 0,
  has_named_event: 0,
  lag_1: "",
  lag_7: "",
  rolling_mean_7: "",
};

/**
 * Single-prediction form. Validates client-side, submits to POST /predict,
 * and reports the result to the parent via onResult (for history).
 */
export default function PredictionForm({ onResult }) {
  const [form, setForm] = useState(INITIAL);
  const [errors, setErrors] = useState([]);
  const [result, setResult] = useState(null);
  const [serverError, setServerError] = useState("");
  const [loading, setLoading] = useState(false);

  const update = (key) => (e) => {
    const value =
      e.target.type === "checkbox" ? (e.target.checked ? 1 : 0) : e.target.value;
    setForm((f) => ({ ...f, [key]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setServerError("");
    setResult(null);
    const validationErrors = validatePredictionForm(form);
    setErrors(validationErrors);
    if (validationErrors.length) return;

    const payload = {
      date: form.date,
      product_id: form.product_id.trim(),
      price: Number(form.price),
      snap_day: Number(form.snap_day) || 0,
      holiday: Number(form.holiday) || 0,
      has_named_event: Number(form.has_named_event) || 0,
      features: buildFeatureList({
        lag_1: form.lag_1,
        lag_7: form.lag_7,
        rolling_mean_7: form.rolling_mean_7,
      }),
    };

    setLoading(true);
    try {
      const data = await postPrediction(payload);
      setResult(data);
      onResult?.({
        label: `${data.product_id} @ ${data.date}`,
        prediction: data.predicted_sales,
        status: "success",
      });
    } catch (err) {
      setServerError(err.message);
      onResult?.({
        label: `${payload.product_id} @ ${payload.date}`,
        prediction: null,
        status: "error",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card title="Single Prediction">
      <form onSubmit={handleSubmit} className="form" noValidate>
        <div className="form__grid">
          <label className="field">
            <span>Date</span>
            <input
              type="text"
              placeholder="2016-03-01"
              value={form.date}
              onChange={update("date")}
              aria-label="Date"
            />
          </label>
          <label className="field">
            <span>Product ID</span>
            <input
              type="text"
              placeholder="FOODS_3_090"
              value={form.product_id}
              onChange={update("product_id")}
              aria-label="Product ID"
            />
          </label>
          <label className="field">
            <span>Price</span>
            <input
              type="number"
              step="0.01"
              placeholder="3.48"
              value={form.price}
              onChange={update("price")}
              aria-label="Price"
            />
          </label>
          <label className="field">
            <span>lag_1</span>
            <input
              type="number"
              value={form.lag_1}
              onChange={update("lag_1")}
              aria-label="lag_1"
            />
          </label>
          <label className="field">
            <span>lag_7</span>
            <input
              type="number"
              value={form.lag_7}
              onChange={update("lag_7")}
              aria-label="lag_7"
            />
          </label>
          <label className="field">
            <span>rolling_mean_7</span>
            <input
              type="number"
              value={form.rolling_mean_7}
              onChange={update("rolling_mean_7")}
              aria-label="rolling_mean_7"
            />
          </label>
        </div>

        <div className="form__flags">
          <label className="checkbox">
            <input
              type="checkbox"
              checked={Number(form.snap_day) === 1}
              onChange={update("snap_day")}
              aria-label="SNAP day"
            />
            <span>SNAP day</span>
          </label>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={Number(form.holiday) === 1}
              onChange={update("holiday")}
              aria-label="Holiday"
            />
            <span>Holiday</span>
          </label>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={Number(form.has_named_event) === 1}
              onChange={update("has_named_event")}
              aria-label="Named event"
            />
            <span>Named event</span>
          </label>
        </div>

        {errors.length > 0 && (
          <Alert type="error">
            <ul className="error-list">
              {errors.map((msg) => (
                <li key={msg}>{msg}</li>
              ))}
            </ul>
          </Alert>
        )}
        {serverError && <Alert type="error">{serverError}</Alert>}

        <button type="submit" className="btn btn--primary" disabled={loading}>
          {loading ? <Spinner label="Predicting" /> : "Predict"}
        </button>
      </form>

      {result && (
        <Alert type="success">
          <div className="result">
            <div>
              <span className="result__label">Predicted Sales</span>
              <span className="result__value" data-testid="predicted-sales">
                {fmt(result.predicted_sales)}
              </span>
            </div>
            <div>
              <span className="result__label">Processing Time</span>
              <span data-testid="processing-time">
                {fmt(result.processing_time_ms)} ms
              </span>
            </div>
            <div>
              <span className="result__label">Model</span>
              <span>{result.model_type}</span>
            </div>
          </div>
        </Alert>
      )}
    </Card>
  );
}
