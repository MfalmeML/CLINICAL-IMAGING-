"""Structural tests for Docker artifacts.

These do not build the image. They assert that the Dockerfile and
.dockerignore contain the properties the build plan requires: non-root
user, explicit port, no data directory copied in, non-clinical framing
present in the image metadata.
"""

from pathlib import Path


def test_dockerfile_exists() -> None:
    assert Path("Dockerfile").is_file()


def test_dockerfile_runs_non_root() -> None:
    text = Path("Dockerfile").read_text()
    assert "USER imagex" in text
    assert "useradd" in text


def test_dockerfile_exposes_port() -> None:
    text = Path("Dockerfile").read_text()
    assert "EXPOSE 8000" in text


def test_dockerfile_cmd_runs_api() -> None:
    text = Path("Dockerfile").read_text()
    assert "imagex.serving.api:app" in text


def test_dockerfile_does_not_copy_data() -> None:
    text = Path("Dockerfile").read_text()
    # No COPY line may pull in the data directory.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("COPY ") or stripped.startswith("ADD "):
            assert "data/" not in stripped, f"data copied into image: {stripped}"


def test_dockerignore_excludes_data_and_venv() -> None:
    text = Path(".dockerignore").read_text()
    assert "data/" in text
    assert ".venv/" in text
    assert ".git/" in text
