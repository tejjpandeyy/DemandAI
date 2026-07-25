import { useCallback, useEffect, useRef, useState } from "react";
import { getHealth } from "../services/api";

/**
 * Poll GET /health on an interval (default 30s per spec).
 * Returns { status, modelLoaded, loading, lastChecked, refresh }.
 * status is "healthy" | "offline" | "unknown".
 */
export function useHealth(intervalMs = 30000) {
  const [status, setStatus] = useState("unknown");
  const [modelLoaded, setModelLoaded] = useState(false);
  const [loading, setLoading] = useState(true);
  const [lastChecked, setLastChecked] = useState(null);
  const timerRef = useRef(null);

  const check = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getHealth();
      setStatus(data?.status === "healthy" ? "healthy" : "offline");
      setModelLoaded(Boolean(data?.model_loaded));
    } catch {
      setStatus("offline");
      setModelLoaded(false);
    } finally {
      setLoading(false);
      setLastChecked(new Date());
    }
  }, []);

  useEffect(() => {
    check();
    timerRef.current = setInterval(check, intervalMs);
    return () => clearInterval(timerRef.current);
  }, [check, intervalMs]);

  return { status, modelLoaded, loading, lastChecked, refresh: check };
}
