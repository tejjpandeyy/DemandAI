// Central API layer -- the ONLY module that knows backend URLs and
// response shapes. Components and hooks import these functions; they
// never call fetch directly. This keeps the backend contract in one
// place (matches Phase 8: GET /health, POST /predict, POST /predict/batch).

const BASE_URL = import.meta.env?.VITE_API_BASE_URL ?? "/api";

/**
 * Low-level JSON fetch with uniform error handling.
 * Throws an Error whose message is the backend's `detail` when present.
 */
async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (networkError) {
    throw new Error("Network error: cannot reach the API.");
  }

  let body = null;
  const text = await response.text();
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }

  if (!response.ok) {
    throw new Error(extractError(body, response.status));
  }
  return body;
}

/** Turn a FastAPI/Pydantic error body into a readable string. */
export function extractError(body, status) {
  if (body && typeof body === "object") {
    if (typeof body.detail === "string") return body.detail;
    // Pydantic 422 returns detail as an array of {loc, msg, ...}.
    if (Array.isArray(body.detail)) {
      return body.detail
        .map((e) => {
          const field = Array.isArray(e.loc) ? e.loc.slice(1).join(".") : "";
          return field ? `${field}: ${e.msg}` : e.msg;
        })
        .join("; ");
    }
  }
  if (typeof body === "string" && body) return body;
  return `Request failed with status ${status}`;
}

export function getHealth() {
  return request("/health");
}

export function getRoot() {
  return request("/");
}

export function postPrediction(payload) {
  return request("/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function postBatchPrediction(requests) {
  return request("/predict/batch", {
    method: "POST",
    body: JSON.stringify({ requests }),
  });
}
