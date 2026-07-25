import { NavLink } from "react-router-dom";

const LINKS = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/predict", label: "Single" },
  { to: "/batch", label: "Batch" },
  { to: "/history", label: "History" },
];

export default function NavBar() {
  return (
    <nav className="navbar">
      <div className="navbar__brand">
        <span className="navbar__logo">◆</span> DemandAI
      </div>
      <ul className="navbar__links">
        {LINKS.map((link) => (
          <li key={link.to}>
            <NavLink
              to={link.to}
              end={link.end}
              className={({ isActive }) =>
                `navbar__link${isActive ? " navbar__link--active" : ""}`
              }
            >
              {link.label}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}
