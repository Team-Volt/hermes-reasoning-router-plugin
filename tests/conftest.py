import pytest


@pytest.fixture(autouse=True)
def isolated_hermes_home(tmp_path, monkeypatch):
    """Keep every test from reading the live ~/.hermes reasoning-router config."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
