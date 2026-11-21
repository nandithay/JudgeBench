"""Streamlit smoke tests use its built-in headless test runner, without a browser."""
from streamlit.testing.v1 import AppTest


def test_dashboard_without_run_is_an_honest_empty_state(tmp_path, monkeypatch):
    # Default runs/first intentionally does not exist in the shipped project.
    monkeypatch.chdir(tmp_path)
    from pathlib import Path
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    assert not app.exception
    assert any("Initialize a run first" in item.value for item in app.info)
