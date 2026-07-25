// Small, reusable presentational primitives shared across pages.

export function Spinner({ label = "Loading" }) {
  return (
    <span className="spinner" role="status" aria-label={label}>
      <span className="spinner__dot" />
      {label}
    </span>
  );
}

export function Alert({ type = "info", children }) {
  // type: "error" | "success" | "info"
  const role = type === "error" ? "alert" : "status";
  return (
    <div className={`alert alert--${type}`} role={role}>
      {children}
    </div>
  );
}

export function StatusBadge({ status }) {
  // status: "healthy" | "offline" | "unknown"
  const label =
    status === "healthy" ? "Healthy" : status === "offline" ? "Offline" : "…";
  return (
    <span className={`badge badge--${status}`} data-testid="status-badge">
      {label}
    </span>
  );
}

export function Card({ title, children, actions }) {
  return (
    <section className="card">
      {title && (
        <header className="card__header">
          <h2 className="card__title">{title}</h2>
          {actions}
        </header>
      )}
      <div className="card__body">{children}</div>
    </section>
  );
}
