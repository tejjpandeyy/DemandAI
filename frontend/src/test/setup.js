import "@testing-library/jest-dom";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// Unmount React trees and reset mocks between tests to prevent leakage.
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.clearAllTimers();
});
