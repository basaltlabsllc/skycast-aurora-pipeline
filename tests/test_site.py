import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from pipeline import site, solarwind
from pipeline.encode import EAVG_RANGE, EFLUX_RANGE, H, TYPES, W
from tests.feeds import DELAY_S, RUN, feeds

ROOT = Path(__file__).parent.parent


def _ts(iso):
    return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


class Recorder:
    """Stand-in for model.run_frame: remembers each dF, returns a lit cell."""

    def __init__(self, lit=True, value=1.0):
        self.calls, self.lit, self.value = [], lit, value

    def __call__(self, dF, when):
        self.calls.append((when, dF))
        out = {t: (np.zeros((H, W)), np.zeros((H, W))) for t in TYPES}
        if self.lit:
            out["diffuse"][0][160, 0] = self.value
        return out


def _build(*args, rec=None, run=RUN, **kw):
    rec = rec or Recorder()
    m, files = site.build_site(*feeds(*args, run=run, **kw), run.timestamp(), rec,
                               log=lambda s: None)
    return m, files, rec


def test_manifest_window():
    m, files, rec = _build()
    assert set(m) == {"v", "model", "runTime", "spacecraft", "swNewest", "swArrival",
                      "windowStart", "windowEnd", "stepMin", "enc", "frames"}
    assert m["v"] == 1 and m["stepMin"] == 10
    assert m["enc"] == {"eflux": list(EFLUX_RANGE), "eavg": list(EAVG_RANGE)}
    assert m["runTime"] == "2026-11-17T12:03:27Z"
    # run - 60 min = 11:03:27 -> floored to 11:00
    assert m["windowStart"] == "2026-11-17T11:00:00Z"
    # newest wind 12:01 (2 min before the run) + 45 min = 12:46 -> 12:40
    assert m["swNewest"] == "2026-11-17T12:01:00Z"
    assert m["swArrival"] == "2026-11-17T12:46:00Z"
    assert m["windowEnd"] == "2026-11-17T12:40:00Z"
    ts = [_ts(f["t"]) for f in m["frames"]]
    assert ts[0] == _ts(m["windowStart"]) and ts[-1] == _ts(m["windowEnd"])
    assert np.all(np.diff(ts) == 600)                      # oldest first, step 10
    for f in m["frames"]:
        assert re.fullmatch(r"frames/\d{8}T\d{4}Z\.aur1", f["path"])
        assert f["path"] == datetime.fromtimestamp(_ts(f["t"]), timezone.utc).strftime(
            "frames/%Y%m%dT%H%MZ.aur1")
    assert sorted(files) == sorted(f["path"] for f in m["frames"])
    assert len(rec.calls) == len(m["frames"]) == 11


def test_window_start_on_a_step_boundary():
    run = datetime(2026, 11, 17, 12, 0, 0, tzinfo=timezone.utc)
    m, _, _ = _build(run=run)
    assert m["windowStart"] == "2026-11-17T11:00:00Z"


def test_model_names_the_pinned_commit():
    req = (ROOT / "requirements.txt").read_text()
    assert f"OvationPyme@{site.OVATIONPYME_COMMIT}" in req
    assert site.MODEL == f"ovation-prime-2010 (OvationPyme {site.OVATIONPYME_COMMIT})"


def test_frame_uses_only_arrived_wind():
    m, _, rec = _build()
    wind, mag, prop = feeds()
    series = solarwind.load_series(wind, mag)
    for f, (when, dF) in zip(m["frames"], rec.calls):
        t = _ts(f["t"])
        assert when.timestamp() == t
        assert _ts(f["swUsedTo"]) <= t - DELAY_S
        # swUsedTo is the newest sample that has arrived, not an older one
        assert _ts(f["swUsedTo"]) == series.t[series.t <= t - DELAY_S].max()
        # the frame's coupling is the same with every not-yet-arrived sample gone
        arrived = series.t <= t - DELAY_S
        cut = solarwind.Series(t=series.t[arrived], v=series.v[arrived],
                               bx=series.bx[arrived], by=series.by[arrived],
                               bz=series.bz[arrived], spacecraft=series.spacecraft)
        assert solarwind.coupling_at(cut, t, DELAY_S) == dF
    # control: the frames really do see different wind
    assert len({round(dF, 6) for _, dF in rec.calls}) > 1


def test_spacecraft_switch_mid_series():
    switch = RUN - timedelta(hours=2)
    active = lambda t: "ACE" if t < switch else "DSCOVR"
    wind, mag, prop = feeds(sources=("ACE", "DSCOVR"), active_at=active)
    s = solarwind.load_series(wind, mag)
    assert s.spacecraft == "DSCOVR"                         # source of the newest sample
    assert np.all(np.diff(s.t) == 60)                       # one continuous series
    assert s.t.size == 6 * 60
    assert s.v.max() < 1000 and s.bz.min() > -20            # no inactive row leaked in
    m, _, _ = _build(sources=("ACE", "DSCOVR"), active_at=active)
    assert m["spacecraft"] == "DSCOVR"
    # the newest sample decides, even if the older half of the window was ACE
    m2, _, _ = _build(sources=("ACE", "DSCOVR"),
                      active_at=lambda t: "DSCOVR" if t < switch else "ACE")
    assert m2["spacecraft"] == "ACE"


def test_gap_in_one_frames_hours_raises():
    # 20-min hole 5 h before the run: only the earliest frames average across it.
    hole = RUN.replace(second=0) - timedelta(hours=5)
    with pytest.raises(solarwind.SolarWindGap):
        _build(drop=[(hole, hole + timedelta(minutes=20))])
    # control: the same hole 6.5 h back is older than any frame's 4 h -> fine
    old = RUN.replace(second=0) - timedelta(hours=6, minutes=30)
    _build(hours=7, drop=[(old, old + timedelta(minutes=20))])


def test_stale_wind_raises():
    # newest wind 3 h old: it arrives long before the window starts
    with pytest.raises(solarwind.SolarWindGap):
        _build(newest_age_min=180, hours=9)


def test_all_zero_raises():
    with pytest.raises(site.NoAurora):
        _build(rec=Recorder(lit=False))


def test_all_below_encode_floor_raises():
    # Flux under the encoder's floor becomes byte 0: the frames would be blank.
    with pytest.raises(site.NoAurora):
        _build(rec=Recorder(value=EFLUX_RANGE[0] * 0.9))
    _build(rec=Recorder(value=EFLUX_RANGE[0]))  # exactly the floor is encoded


def test_delay_ignores_null_time_tags():
    _, _, prop = feeds()
    prop = prop + [[None] + prop[-1][1:-1] + ["2026-11-17T23:00:00Z"],
                   [prop[-1][0]] + prop[-1][1:-1] + [None]]
    assert solarwind.arrival_delay_s(prop) == DELAY_S


@pytest.mark.parametrize("delay_s", [9 * 60, 181 * 60, -600])
def test_delay_out_of_range_raises(delay_s):
    with pytest.raises(solarwind.BadArrivalDelay):
        solarwind.arrival_delay_s(feeds(delay_s=delay_s)[2])
    # control: the ends of the range are accepted
    for ok in (10 * 60, 180 * 60):
        assert solarwind.arrival_delay_s(feeds(delay_s=ok)[2]) == ok


def test_delay_with_no_usable_rows_raises():
    head = feeds()[2][0]
    with pytest.raises(solarwind.BadArrivalDelay):
        solarwind.arrival_delay_s([head, [None] * len(head)])
