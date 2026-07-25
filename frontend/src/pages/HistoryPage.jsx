import HistoryTable from "../components/HistoryTable";

export default function HistoryPage({ entries, onClear }) {
  return (
    <div className="page">
      <h1 className="page__title">Prediction History</h1>
      <HistoryTable entries={entries} onClear={onClear} />
    </div>
  );
}
