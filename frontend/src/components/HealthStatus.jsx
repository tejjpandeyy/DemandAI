import { useHealth } from "../hooks/useHealth";
import { Card, Spinner, StatusBadge } from "./ui";

/**
 * Live API/model health card. Polls every 30s via useHealth and lets the
 * user trigger a manual refresh.
 */
export default function HealthStatus() {
  const { status, modelLoaded, loading, lastChecked, refresh } = useHealth();

  return (
    <Card
      title="System Status"
      actions={
        <button
          type="button"
          className="btn btn--ghost"
          onClick={refresh}
          aria-label="Refresh status"
        >
          Refresh
        </button>
      }
    >
      <div className="status-grid">
        <div className="status-row">
          <span>API</span>
          {loading && status === "unknown" ? (
            <Spinner label="Checking" />
          ) : (
            <StatusBadge status={status} />
          )}
        </div>
        <div className="status-row">
          <span>Model</span>
          <span data-testid="model-status">
            {modelLoaded ? "Loaded" : "Not loaded"}
          </span>
        </div>
        {lastChecked && (
          <p className="status-meta">
            Last checked {lastChecked.toLocaleTimeString()} · auto-refreshes
            every 30s
          </p>
        )}
      </div>
    </Card>
  );
}
