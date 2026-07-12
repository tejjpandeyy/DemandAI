"""Phase 1 sanity check.

Verifies that all core dependencies are installed and the folder
architecture is in place. Run this after `pip install -r requirements.txt`.
"""

import importlib
import sys
from pathlib import Path

REQUIRED_PACKAGES: list[str] = [
    "pandas",
    "numpy",
    "sklearn",
    "xgboost",
    "joblib",
    "fastapi",
    "uvicorn",
    "sqlalchemy",
    "pydantic",
    "pytest",
]

REQUIRED_DIRS: list[str] = [
    "app/api",
    "app/models",
    "app/schemas",
    "app/services",
    "app/database",
    "app/utils",
    "ml",
    "saved_models",
    "datasets",
    "tests",
]


def check_packages() -> bool:
    """Import each required package and report its version."""
    ok = True
    for name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"  [OK] {name:12s} {version}")
        except ImportError:
            print(f"  [MISSING] {name} -> run: pip install -r requirements.txt")
            ok = False
    return ok


def check_folders() -> bool:
    """Verify the backend folder architecture exists."""
    base = Path(__file__).parent
    ok = True
    for rel in REQUIRED_DIRS:
        path = base / rel
        if path.is_dir():
            print(f"  [OK] {rel}/")
        else:
            print(f"  [MISSING] {rel}/")
            ok = False
    return ok


if __name__ == "__main__":
    print("Checking packages...")
    packages_ok = check_packages()
    print("\nChecking folder architecture...")
    folders_ok = check_folders()

    if packages_ok and folders_ok:
        print("\n✅ Phase 1 setup verified. Ready for Phase 2.")
        sys.exit(0)
    print("\n❌ Setup incomplete. Fix the items marked MISSING above.")
    sys.exit(1)
