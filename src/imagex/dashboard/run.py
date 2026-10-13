"""Launcher for the Streamlit dashboard.

Sets PYTHONPATH so `imagex.dashboard.app` is importable, then delegates
to Streamlit's CLI. Runs the dashboard as a research prototype only.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    """Return the project root, inferred from this file's location."""
    # this file lives at <root>/src/imagex/dashboard/run.py
    return Path(__file__).resolve().parents[3]


def app_path() -> Path:
    return Path(__file__).resolve().parent / "app.py"


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    root = project_root()
    src = root / "src"

    if not src.is_dir():
        print(f"error: expected source directory not found: {src}", file=sys.stderr)
        return 2

    env = {"PYTHONPATH": str(src)}
    import os
    full_env = {**os.environ, **env}

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path()),
        "--server.headless", "true",
        *args,
    ]

    return subprocess.call(cmd, env=full_env)


if __name__ == "__main__":
    raise SystemExit(main())
