import PredictionForm from "../components/PredictionForm";

export default function PredictPage({ onResult }) {
  return (
    <div className="page">
      <h1 className="page__title">Single Prediction</h1>
      <PredictionForm onResult={onResult} />
    </div>
  );
}
