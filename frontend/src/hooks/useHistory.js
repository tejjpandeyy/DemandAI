import { useCallback, useState } from "react";

let counter = 0;

/**
 * Session-only prediction history (no backend storage, per spec).
 * Entries are kept newest-first. Each entry:
 *   { id, time, label, prediction, status }
 * status is "success" | "error".
 */
export function useHistory() {
  const [entries, setEntries] = useState([]);

  const addEntry = useCallback((entry) => {
    counter += 1;
    const record = {
      id: counter,
      time: new Date(),
      status: "success",
      ...entry,
    };
    setEntries((prev) => [record, ...prev]); // newest first
    return record;
  }, []);

  const clear = useCallback(() => setEntries([]), []);

  return { entries, addEntry, clear };
}
