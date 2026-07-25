import { fmt } from "../utils/format";
import { Card } from "./ui";

/**
 * Session prediction history (newest first). Purely presentational: it
 * renders whatever `entries` the parent supplies from the useHistory hook.
 */
export default function HistoryTable({ entries, onClear }) {
  return (
    <Card
      title="Prediction History"
      actions={
        entries.length > 0 && (
          <button type="button" className="btn btn--ghost" onClick={onClear}>
            Clear
          </button>
        )
      }
    >
      {entries.length === 0 ? (
        <p className="empty" data-testid="history-empty">
          No predictions yet this session.
        </p>
      ) : (
        <table className="table" data-testid="history-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Prediction</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry) => (
              <tr key={entry.id}>
                <td>{entry.time.toLocaleTimeString()}</td>
                <td>
                  <span className="history-label">{entry.label}</span>
                  {entry.prediction !== null &&
                    entry.prediction !== undefined && (
                      <span className="history-value"> → {fmt(entry.prediction)}</span>
                    )}
                </td>
                <td>
                  <span className={`badge badge--${entry.status}`}>
                    {entry.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Card>
  );
}
