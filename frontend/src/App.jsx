import { Route, Routes } from "react-router-dom";
import NavBar from "./components/NavBar";
import { useHistory } from "./hooks/useHistory";
import BatchPage from "./pages/BatchPage";
import Dashboard from "./pages/Dashboard";
import HistoryPage from "./pages/HistoryPage";
import PredictPage from "./pages/PredictPage";

/**
 * App shell. History lives here (lifted state) so predictions made on any
 * page appear on the History page -- session-only, no backend storage.
 */
export default function App() {
  const { entries, addEntry, clear } = useHistory();

  return (
    <div className="app">
      <NavBar />
      <main className="app__main">
        <Routes>
          <Route path="/" element={<Dashboard onResult={addEntry} />} />
          <Route path="/predict" element={<PredictPage onResult={addEntry} />} />
          <Route path="/batch" element={<BatchPage onResult={addEntry} />} />
          <Route
            path="/history"
            element={<HistoryPage entries={entries} onClear={clear} />}
          />
        </Routes>
      </main>
      <footer className="app__footer">
        DemandAI · Retail Demand Forecasting
      </footer>
    </div>
  );
}
