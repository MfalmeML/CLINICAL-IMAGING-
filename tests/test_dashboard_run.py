"""Tests for the dashboard launcher. No Streamlit server started."""

from pathlib import Path

from imagex.dashboard.run import app_path, project_root


def test_project_root_is_repo_root() -> None:
    root = project_root()
    assert (root / "src").is_dir()
    assert (root / "tests").is_dir()
    assert (root / "README.md").is_file()


def test_app_path_exists() -> None:
    p = app_path()
    assert p.is_file()
    assert p.name == "app.py"
    assert p.parent.name == "dashboard"


def test_app_path_is_under_project_root() -> None:
    root = project_root()
    p = app_path()
    assert str(p).startswith(str(root))
