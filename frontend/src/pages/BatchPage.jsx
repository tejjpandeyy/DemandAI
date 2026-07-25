import BatchPrediction from "../components/BatchPrediction";

export default function BatchPage({ onResult }) {
  return (
    <div className="page">
      <h1 className="page__title">Batch Prediction</h1>
      <BatchPrediction onResult={onResult} />
    </div>
  );
}
