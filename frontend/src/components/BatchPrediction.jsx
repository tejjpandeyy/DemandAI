import { useState } from "react";
import { postBatchPrediction } from "../services/api";
import { fmt, parseCsv } from "../utils/format";
import { Alert, Card, Spinner } from "./ui";

const SAMPLE = JSON.stringify(
  [
    {
      date: "2016-03-01",
      product_id: "FOODS_3_090",
      price: 3.48,
      features: [{ name: "lag_1", value: 12 }],
    },
  ],
  null,
  2,
);

/**
 * Batch prediction via JSON text or CSV upload. Both paths converge to an
 * array of request objects, sent to POST /predict/batch and rendered as a
 * table.
 */
export default function BatchPrediction({ onResult }) {
  const [jsonText, setJsonText] = useState(SAMPLE);
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function runBatch(requests) {
    if (!Array.isArray(requests) || requests.length === 0) {
      setError("Provide at least one prediction request.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const data = await postBatchPrediction(requests);
      setRows(data.predictions ?? []);
      onResult?.({
        label: `Batch of ${data.count}`,
        prediction: data.count,
        status: "success",
      });
    } catch (err) {
      setError(err.message);
      setRows([]);
      onResult?.({ label: "Batch", prediction: null, status: "error" });
    } finally {
      setLoading(false);
    }
  }

  const handleJsonSubmit = () => {
    let parsed;
    try {
      parsed = JSON.parse(jsonText);
    } catch {
      setError("Invalid JSON.");
      return;
    }
    runBatch(parsed);
  };

  const handleCsv = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const requests = parseCsv(text);
      runBatch(requests);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <Card title="Batch Prediction">
      <div className="batch">
        <label className="field">
          <span>JSON requests</span>
          <textarea
            className="batch__json"
            rows={8}
            value={jsonText}
            onChange={(e) => setJsonText(e.target.value)}
            aria-label="Batch JSON input"
          />
        </label>

        <div className="batch__actions">
          <button
            type="button"
            className="btn btn--primary"
            onClick={handleJsonSubmit}
            disabled={loading}
          >
            {loading ? <Spinner label="Submitting" /> : "Submit JSON"}
          </button>

          <label className="btn btn--ghost file-btn">
            Upload CSV
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={handleCsv}
              aria-label="Upload CSV"
              hidden
            />
          </label>
        </div>

        {error && <Alert type="error">{error}</Alert>}

        {rows.length > 0 && (
          <table className="table" data-testid="batch-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Product</th>
                <th>Date</th>
                <th>Predicted Sales</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={`${r.product_id}-${i}`}>
                  <td>{i + 1}</td>
                  <td>{r.product_id}</td>
                  <td>{r.date}</td>
                  <td>{fmt(r.predicted_sales)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </Card>
  );
}
