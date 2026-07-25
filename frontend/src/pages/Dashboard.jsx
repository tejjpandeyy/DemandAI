import { Link } from "react-router-dom";
import HealthStatus from "../components/HealthStatus";
import PredictionForm from "../components/PredictionForm";
import { Card } from "../components/ui";

/**
 * Dashboard home: system status, a quick prediction card, and navigation
 * shortcuts. Receives onResult from App so quick predictions feed history.
 */
export default function Dashboard({ onResult }) {
  return (
    <div className="page">
      <h1 className="page__title">Retail Demand Dashboard</h1>
      <div className="grid grid--two">
        <HealthStatus />
        <Card title="Explore">
          <p className="muted">Jump into a workflow:</p>
          <div className="quick-links">
            <Link className="btn btn--primary" to="/predict">
              Single Prediction
            </Link>
            <Link className="btn btn--ghost" to="/batch">
              Batch Prediction
            </Link>
            <Link className="btn btn--ghost" to="/history">
              View History
            </Link>
          </div>
        </Card>
      </div>
      <PredictionForm onResult={onResult} />
    </div>
  );
}
