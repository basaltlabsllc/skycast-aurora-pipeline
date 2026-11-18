import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.solarwind import (Series, SolarWindGap, arrival_delay_s, check_gaps,
                                coupling_at, load_series, newell)

FIX = Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FIX / name).read_text())


def _fixture_series():
    return load_series(_load("rtsw_wind.json"), _load("rtsw_mag.json"))


def _synthetic(t0, minutes, v=400.0, by=0.0, bz=-5.0):
    t = t0 + 60.0 * np.arange(minutes)
    n = t.size
    return Series(t=t, v=np.full(n, v), bx=np.zeros(n), by=np.full(n, by),
                  bz=np.full(n, bz), spacecraft="TEST")


def test_newell_known_value():
    got = newell(np.array([400.0]), np.array([0.0]), np.array([-5.0]))[0]
    want = 400 ** (4 / 3) * 5 ** (2 / 3)
    assert got == pytest.approx(want, rel=1e-6)


def test_northward_is_zero():
    got = newell(np.array([400.0]), np.array([0.0]), np.array([5.0]))[0]
    assert got == pytest.approx(0.0, abs=1e-9)


def test_only_active_rows():
    wind, mag = _load("rtsw_wind.json"), _load("rtsw_mag.json")
    sources = {r["source"] for r in wind}
    assert len(sources) >= 2                      # the fixture really interleaves spacecraft
    active = {r["source"] for r in wind if r["active"]}
    assert len(active) == 1
    (src,) = active
    s = load_series(wind, mag)
    assert s.spacecraft == src
    usable = {r["time_tag"][:16] for r in wind
              if r["active"] and r["proton_speed"] is not None}
    usable &= {r["time_tag"][:16] for r in mag
               if r["active"] and r["by_gsm"] is not None and r["bz_gsm"] is not None}
    assert s.t.size == len(usable)
    assert np.all(np.diff(s.t) > 0)               # ascending, no duplicate minutes
    assert np.all(s.t % 60 == 0)
    assert not np.any(np.isnan(s.v)) and not np.any(np.isnan(s.bz))


def test_arrival_delay_from_newest_row():
    rows = _load("propagated.json")
    # newest row 2026-11-18T00:21:00Z -> propagated 2026-11-18T01:17:21Z
    assert arrival_delay_s(rows) == 56 * 60 + 21


def test_gap_raises():
    s = _fixture_series()
    t_from, t_to = s.t[0], s.t[-1]
    check_gaps(s, t_from, t_to)                   # control: the real fixture passes
    mid = s.t[s.t.size // 2]
    keep = (s.t < mid) | (s.t >= mid + 20 * 60)
    holed = Series(t=s.t[keep], v=s.v[keep], bx=s.bx[keep], by=s.by[keep],
                   bz=s.bz[keep], spacecraft=s.spacecraft)
    with pytest.raises(SolarWindGap):
        check_gaps(holed, t_from, t_to)


def test_gap_at_window_edge_raises():
    s = _synthetic(0.0, 120)
    with pytest.raises(SolarWindGap):
        check_gaps(s, s.t[0] - 20 * 60, s.t[-1])


def test_no_future_wind():
    delay = 3000.0
    s = _synthetic(0.0, 6 * 60)
    t_target = s.t[0] + delay + 4.5 * 3600
    base = coupling_at(s, t_target, delay)
    late = s.t + delay > t_target
    assert late.any()                             # there really are not-yet-arrived samples
    wild = Series(t=s.t, v=np.where(late, 900.0, s.v), bx=s.bx,
                  by=np.where(late, 20.0, s.by), bz=np.where(late, -30.0, s.bz),
                  spacecraft=s.spacecraft)
    assert coupling_at(wild, t_target, delay) == base
    # control: changing an arrived sample does change the result
    early = s.t + delay <= t_target
    moved = Series(t=s.t, v=np.where(early, 900.0, s.v), bx=s.bx, by=s.by, bz=s.bz,
                   spacecraft=s.spacecraft)
    assert coupling_at(moved, t_target, delay) != base


def test_weights_match_ovationpyme():
    s = _synthetic(0.0, 6 * 60)
    want = newell(np.array([400.0]), np.array([0.0]), np.array([-5.0]))[0]
    assert coupling_at(s, s.t[-1], 0.0) == pytest.approx(want, rel=1e-12)


def test_hour_weights_are_065_powers():
    # One distinct speed per hour bin; OvationPyme weights the hour h before
    # the target by 0.65**h (ovation_utilities.calc_avg_solarwind).
    t_target = 4 * 3600.0
    s = _synthetic(60.0, 4 * 60)                 # samples at 60 s .. 4 h, all arrived
    hours_before = (t_target - s.t) / 3600
    v = np.select([hours_before < 1, hours_before < 2, hours_before < 3],
                  [300.0, 400.0, 500.0], 600.0)
    s = Series(t=s.t, v=v, bx=s.bx, by=s.by, bz=s.bz, spacecraft="TEST")
    ec = {sp: newell(np.array([sp]), np.array([0.0]), np.array([-5.0]))[0]
          for sp in (300.0, 400.0, 500.0, 600.0)}
    w = np.array([0.65 ** 3, 0.65 ** 2, 0.65, 1.0])
    want = (w * [ec[600.0], ec[500.0], ec[400.0], ec[300.0]]).sum() / w.sum()
    assert coupling_at(s, t_target, 0.0) == pytest.approx(want, rel=1e-12)


def test_empty_hour_raises():
    s = _synthetic(0.0, 2 * 60)                  # only two hours of wind
    with pytest.raises(SolarWindGap):
        coupling_at(s, s.t[-1], 0.0)
