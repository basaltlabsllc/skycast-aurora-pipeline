import pytest

import run_aurora


@pytest.fixture(autouse=True)
def _no_live_noaa_map(monkeypatch):
    """No test reads SWPC's live map: a test that wants one passes fetch_map."""
    monkeypatch.setattr(run_aurora, "fetch_noaa", lambda: None)
