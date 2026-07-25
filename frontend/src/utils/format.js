// Pure helpers (no React, no network) -- easy to unit test.

const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

/** Strict YYYY-MM-DD check, mirroring the backend's date validation. */
export function isValidDate(value) {
  if (typeof value !== "string" || !DATE_RE.test(value)) return false;
  const d = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(d.getTime()) && value === d.toISOString().slice(0, 10);
}

/** Validate a single-prediction form. Returns an array of error strings. */
export function validatePredictionForm({ date, product_id, price }) {
  const errors = [];
  if (!isValidDate(date)) errors.push("Date must be in YYYY-MM-DD format.");
  if (!product_id || !product_id.trim()) errors.push("Product ID is required.");
  const priceNum = Number(price);
  if (price === "" || price === null || price === undefined) {
    errors.push("Price is required.");
  } else if (Number.isNaN(priceNum) || priceNum <= 0) {
    errors.push("Price must be a positive number.");
  }
  return errors;
}

/**
 * Build the `features` list ({name, value}) for the API from an object of
 * engineered feature values, skipping blanks. Non-numeric values become
 * NaN -> null so the backend treats them as missing.
 */
export function buildFeatureList(featureObj = {}) {
  return Object.entries(featureObj)
    .filter(([, v]) => v !== "" && v !== null && v !== undefined)
    .map(([name, v]) => {
      const num = Number(v);
      return { name, value: Number.isNaN(num) ? null : num };
    });
}

/**
 * Parse CSV text into an array of prediction-request objects.
 * The header row names columns; date/product_id/price map to top-level
 * fields, snap_day/holiday/has_named_event to flags, and any column
 * starting with "lag_", "rolling_", "expanding" or "price_" becomes a
 * feature. Throws on a missing required column.
 */
export function parseCsv(text) {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  if (lines.length < 2) {
    throw new Error("CSV must have a header row and at least one data row.");
  }
  const headers = lines[0].split(",").map((h) => h.trim());
  const required = ["date", "product_id", "price"];
  const missing = required.filter((c) => !headers.includes(c));
  if (missing.length) {
    throw new Error(`CSV is missing required column(s): ${missing.join(", ")}`);
  }

  const flagCols = ["snap_day", "holiday", "has_named_event"];
  const isFeature = (h) =>
    /^(lag_|rolling_|expanding|price_change|price_pct)/.test(h);

  return lines.slice(1).map((line, idx) => {
    const cells = line.split(",").map((c) => c.trim());
    if (cells.length !== headers.length) {
      throw new Error(
        `Row ${idx + 1} has ${cells.length} values but ${headers.length} were expected.`,
      );
    }
    const row = Object.fromEntries(headers.map((h, i) => [h, cells[i]]));
    const req = {
      date: row.date,
      product_id: row.product_id,
      price: Number(row.price),
    };
    flagCols.forEach((f) => {
      if (row[f] !== undefined && row[f] !== "") req[f] = Number(row[f]);
    });
    const features = headers
      .filter(isFeature)
      .filter((h) => row[h] !== "" && row[h] !== undefined)
      .map((h) => ({ name: h, value: Number(row[h]) }));
    if (features.length) req.features = features;
    return req;
  });
}

/** Round a number for display; passes through non-numbers unchanged. */
export function fmt(n, digits = 2) {
  return typeof n === "number" && Number.isFinite(n) ? n.toFixed(digits) : n;
}
