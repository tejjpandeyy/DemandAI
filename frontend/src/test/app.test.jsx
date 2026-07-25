import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import HealthStatus from "../components/HealthStatus";
import PredictionForm from "../components/PredictionForm";
import BatchPrediction from "../components/BatchPrediction";
import HistoryTable from "../components/HistoryTable";
import { extractError } from "../services/api";
import {
  buildFeatureList,
  isValidDate,
  parseCsv,
  validatePredictionForm,
} from "../utils/format";

// The API layer is the single seam we mock, so components are tested
// without a live backend.
import * as api from "../services/api";

function renderWithRouter(ui) {
  return render(<MemoryRouter>{ui}</MemoryRouter>);
}

// ----------------------------- utils -------------------------------------

describe("format utils", () => {
  it("isValidDate accepts strict YYYY-MM-DD and rejects others", () => {
    expect(isValidDate("2016-03-01")).toBe(true);
    expect(isValidDate("20160301")).toBe(false);
    expect(isValidDate("2016-13-45")).toBe(false);
    expect(isValidDate("March 1")).toBe(false);
  });

  it("validatePredictionForm flags missing/invalid fields", () => {
    const errors = validatePredictionForm({
      date: "bad",
      product_id: "",
      price: "-1",
    });
    expect(errors.length).toBe(3);
  });

  it("validatePredictionForm passes a valid form", () => {
    const errors = validatePredictionForm({
      date: "2016-03-01",
      product_id: "P1",
      price: "3.5",
    });
    expect(errors).toEqual([]);
  });

  it("buildFeatureList skips blanks and coerces numbers", () => {
    const list = buildFeatureList({ lag_1: "12", lag_7: "", rolling: "3.5" });
    expect(list).toEqual([
      { name: "lag_1", value: 12 },
      { name: "rolling", value: 3.5 },
    ]);
  });

  it("parseCsv converts rows and detects missing columns", () => {
    const csv = "date,product_id,price,lag_1\n2016-03-01,P1,3.48,12";
    const rows = parseCsv(csv);
    expect(rows[0].product_id).toBe("P1");
    expect(rows[0].features).toEqual([{ name: "lag_1", value: 12 }]);
    expect(() => parseCsv("date,price\n2016-03-01,3")).toThrow(/product_id/);
  });

  it("extractError formats a Pydantic 422 detail array", () => {
    const msg = extractError(
      { detail: [{ loc: ["body", "price"], msg: "must be > 0" }] },
      422,
    );
    expect(msg).toContain("price");
    expect(msg).toContain("must be > 0");
  });
});

// ----------------------------- HealthStatus ------------------------------

describe("HealthStatus", () => {
  it("shows Healthy when the API is up", async () => {
    vi.spyOn(api, "getHealth").mockResolvedValue({
      status: "healthy",
      model_loaded: true,
    });
    renderWithRouter(<HealthStatus />);
    await waitFor(() =>
      expect(screen.getByTestId("status-badge")).toHaveTextContent("Healthy"),
    );
    expect(screen.getByTestId("model-status")).toHaveTextContent("Loaded");
  });

  it("shows Offline when the health call fails", async () => {
    vi.spyOn(api, "getHealth").mockRejectedValue(new Error("down"));
    renderWithRouter(<HealthStatus />);
    await waitFor(() =>
      expect(screen.getByTestId("status-badge")).toHaveTextContent("Offline"),
    );
  });

  it("refreshes on demand", async () => {
    const spy = vi
      .spyOn(api, "getHealth")
      .mockResolvedValue({ status: "healthy", model_loaded: true });
    renderWithRouter(<HealthStatus />);
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: /refresh status/i }));
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
  });
});

// ----------------------------- PredictionForm ----------------------------

describe("PredictionForm", () => {
  beforeEach(() => {
    vi.spyOn(api, "postPrediction").mockResolvedValue({
      product_id: "FOODS_3_090",
      date: "2016-03-01",
      predicted_sales: 42.5,
      processing_time_ms: 3.2,
      model_type: "HistGradientBoosting",
    });
  });

  function fillValid() {
    fireEvent.change(screen.getByLabelText("Date"), {
      target: { value: "2016-03-01" },
    });
    fireEvent.change(screen.getByLabelText("Product ID"), {
      target: { value: "FOODS_3_090" },
    });
    fireEvent.change(screen.getByLabelText("Price"), {
      target: { value: "3.48" },
    });
  }

  it("blocks submission and shows errors when invalid", async () => {
    render(<PredictionForm />);
    fireEvent.click(screen.getByRole("button", { name: /predict/i }));
    await waitFor(() =>
      expect(screen.getByRole("alert")).toBeInTheDocument(),
    );
    expect(api.postPrediction).not.toHaveBeenCalled();
  });

  it("submits a valid form and displays predicted sales", async () => {
    render(<PredictionForm />);
    fillValid();
    fireEvent.click(screen.getByRole("button", { name: /predict/i }));
    await waitFor(() =>
      expect(screen.getByTestId("predicted-sales")).toHaveTextContent("42.50"),
    );
    expect(screen.getByTestId("processing-time")).toHaveTextContent("3.20 ms");
    expect(api.postPrediction).toHaveBeenCalledTimes(1);
  });

  it("shows a loading indicator while awaiting the response", async () => {
    let resolve;
    vi.spyOn(api, "postPrediction").mockReturnValue(
      new Promise((r) => {
        resolve = r;
      }),
    );
    render(<PredictionForm />);
    fillValid();
    fireEvent.click(screen.getByRole("button", { name: /predict/i }));
    expect(await screen.findByRole("status")).toHaveTextContent(/predicting/i);
    resolve({
      product_id: "P",
      date: "2016-03-01",
      predicted_sales: 1,
      processing_time_ms: 1,
      model_type: "X",
    });
  });

  it("shows a server error alert on API failure", async () => {
    vi.spyOn(api, "postPrediction").mockRejectedValue(
      new Error("price: must be > 0"),
    );
    render(<PredictionForm />);
    fillValid();
    fireEvent.click(screen.getByRole("button", { name: /predict/i }));
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(/must be > 0/i),
    );
  });

  it("reports results to onResult for history", async () => {
    const onResult = vi.fn();
    render(<PredictionForm onResult={onResult} />);
    fillValid();
    fireEvent.click(screen.getByRole("button", { name: /predict/i }));
    await waitFor(() => expect(onResult).toHaveBeenCalledTimes(1));
    expect(onResult.mock.calls[0][0]).toMatchObject({
      prediction: 42.5,
      status: "success",
    });
  });
});

// ----------------------------- BatchPrediction ---------------------------

describe("BatchPrediction", () => {
  it("submits JSON and renders a results table", async () => {
    vi.spyOn(api, "postBatchPrediction").mockResolvedValue({
      count: 2,
      processing_time_ms: 5,
      predictions: [
        { product_id: "P1", date: "2016-03-01", predicted_sales: 10 },
        { product_id: "P2", date: "2016-03-01", predicted_sales: 20 },
      ],
    });
    render(<BatchPrediction />);
    fireEvent.click(screen.getByRole("button", { name: /submit json/i }));
    await waitFor(() =>
      expect(screen.getByTestId("batch-table")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("batch-table")).toHaveTextContent("P2");
  });

  it("shows an error for invalid JSON", async () => {
    render(<BatchPrediction />);
    fireEvent.change(screen.getByLabelText(/batch json input/i), {
      target: { value: "{not json" },
    });
    fireEvent.click(screen.getByRole("button", { name: /submit json/i }));
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(/invalid json/i),
    );
  });

  it("surfaces a backend error from the batch call", async () => {
    vi.spyOn(api, "postBatchPrediction").mockRejectedValue(
      new Error("batch failed"),
    );
    render(<BatchPrediction />);
    fireEvent.click(screen.getByRole("button", { name: /submit json/i }));
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(/batch failed/i),
    );
  });
});

// ----------------------------- HistoryTable ------------------------------

describe("HistoryTable", () => {
  it("renders the empty state with no entries", () => {
    render(<HistoryTable entries={[]} onClear={() => {}} />);
    expect(screen.getByTestId("history-empty")).toBeInTheDocument();
  });

  it("renders entries newest-first as provided", () => {
    const entries = [
      { id: 2, time: new Date(), label: "B @ d", prediction: 20, status: "success" },
      { id: 1, time: new Date(), label: "A @ d", prediction: 10, status: "error" },
    ];
    render(<HistoryTable entries={entries} onClear={() => {}} />);
    const rows = screen.getAllByRole("row");
    // row[0] is the header; row[1] is the first (newest) data row.
    expect(rows[1]).toHaveTextContent("B @ d");
    expect(screen.getByTestId("history-table")).toBeInTheDocument();
  });

  it("calls onClear when Clear is clicked", () => {
    const onClear = vi.fn();
    const entries = [
      { id: 1, time: new Date(), label: "A", prediction: 1, status: "success" },
    ];
    render(<HistoryTable entries={entries} onClear={onClear} />);
    fireEvent.click(screen.getByRole("button", { name: /clear/i }));
    expect(onClear).toHaveBeenCalledTimes(1);
  });
});
