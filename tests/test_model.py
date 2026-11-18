from datetime import datetime, timezone

import numpy as np
import pytest

from ovationpyme import ovation_utilities

from pipeline import model
from pipeline.model import run_frame

# 06 UT: local midnight falls on 90 W, the longitude where the north magnetic
# pole tilts the oval furthest south.
WHEN = datetime(2026, 3, 1, 6, 0, tzinfo=timezone.utc)
LATS = np.arange(-90, 91)        # row i is latitude i - 90
TYPES = ("diffuse", "mono", "wave")


@pytest.fixture(scope="module")
def frame6000():
    return run_frame(6000.0, WHEN)


def _midnight_col(when):
    ut_h = when.hour + when.minute / 60.0
    lon = ((0.0 - ut_h) * 15.0 + 180.0) % 360.0 - 180.0   # where local time is 00
    return int(round(lon + 180.0)) % 360                  # col 0 is -180


def _boundary_lat(eflux, col, north, level=0.5):
    """|lat| of the most equatorward cell at `col` with eflux >= level."""
    rows = np.flatnonzero(eflux[:, col] >= level)
    lats = np.abs(LATS[rows][(LATS[rows] > 0) if north else (LATS[rows] < 0)])
    assert lats.size, "no cell reaches the level in this hemisphere"
    return lats.min()


def test_shape_and_units(frame6000):
    assert set(frame6000) == set(TYPES)
    for t in TYPES:
        eflux, eavg = frame6000[t]
        assert eflux.shape == (181, 360) and eavg.shape == (181, 360)
        assert np.all(np.isfinite(eflux)) and np.all(np.isfinite(eavg))
        assert eflux.min() >= 0.0 and eavg.min() >= 0.0
        assert 0.1 <= eflux.max() <= 50.0, (t, eflux.max())
        lit = eflux > 0
        assert lit.any()
        known = eavg[lit][eavg[lit] != 0.0]      # eavg 0 = energy unknown
        # AverageEnergyEstimator's rules: 0 below 0.2 keV, capped at 30 keV
        assert known.size and known.min() >= 0.2 and known.max() <= 30.0, (
            t, known.min(), known.max())


def test_unknown_energy_keeps_raw_eflux(frame6000):
    mlat, mlt = model._magnetic_coords(WHEN.replace(tzinfo=None))
    doy = WHEN.timetuple().tm_yday
    ests = model._estimators()
    unknown_total = 0
    for t in TYPES:
        eflux, eavg = frame6000[t]
        raw = model._geo(ests[(t, "energy")],
                         6000.0, doy, mlat, mlt)
        unknown = (eflux > 0) & (eavg == 0)
        unknown_total += unknown.sum()
        np.testing.assert_array_equal(eflux[unknown], raw[unknown])
        np.testing.assert_array_equal(eflux, raw)
    assert unknown_total > 0      # the case exists at dF=6000 (mono/wave)


def test_oval_is_at_high_latitude(frame6000):
    eflux = frame6000["diffuse"][0]
    north = LATS > 0
    south = LATS < 0
    peak_n = LATS[north][np.argmax(eflux[north].max(axis=1))]
    peak_s = LATS[south][np.argmax(eflux[south].max(axis=1))]
    assert 55 <= abs(peak_n) <= 80, peak_n
    assert 55 <= abs(peak_s) <= 80, peak_s


def test_stronger_coupling_moves_equatorward():
    weak = run_frame(2000.0, WHEN)
    strong = run_frame(12000.0, WHEN)
    total = lambda f: sum(f[t][0] for t in TYPES)
    col = _midnight_col(WHEN)
    for north in (True, False):
        b_weak = _boundary_lat(total(weak), col, north)
        b_strong = _boundary_lat(total(strong), col, north)
        assert b_strong < b_weak, (north, b_weak, b_strong)


def test_equator_is_zero(frame6000):
    band = np.abs(LATS) <= 30
    for t in TYPES:
        eflux, eavg = frame6000[t]
        assert np.all(eflux[band] == 0.0) and np.all(eavg[band] == 0.0)


@pytest.mark.parametrize("name", TYPES)
def test_hemisphere_grids_match_upstream(monkeypatch, name):
    # Upstream get_flux_for_time downloads OMNI for dF; pin calc_dF to a fixed
    # value and compare the whole season/hemisphere combination with ours.
    dF = 6000.0
    monkeypatch.setattr(ovation_utilities, "calc_dF", lambda dt: dF)
    when = WHEN.replace(tzinfo=None)
    doy = when.timetuple().tm_yday
    for kind in ("energy", "number"):
        est = model._estimators()[(name, kind)]
        _, ours_n, ours_s = model._hemisphere_grids(est, dF, doy)
        _, _, up_n = est.get_flux_for_time(when, hemi="N")
        _, _, up_s = est.get_flux_for_time(when, hemi="S")
        assert np.any(up_n > 0) and np.any(up_s > 0)
        np.testing.assert_array_equal(ours_n, up_n)
        np.testing.assert_array_equal(ours_s, up_s)


def test_diffuse_peaks_on_the_night_side(frame6000):
    # Diffuse aurora is strongest from late evening through the morning
    # sector. Bin the geographic diffuse eflux by its own AACGM MLT, weighting
    # each cell by its area (cos lat), and find the peak hour per hemisphere.
    mlat, mlt = model._magnetic_coords(WHEN.replace(tzinfo=None))
    eflux = frame6000["diffuse"][0]
    area = np.cos(np.radians(LATS))[:, None] * np.ones_like(eflux)
    ok = np.isfinite(mlat) & np.isfinite(mlt)          # AACGM is undefined near the equator
    hour = np.floor(np.where(ok, mlt, 0.0)).astype(int) % 24
    for north in (True, False):
        side = ok & ((mlat > 0) if north else (mlat < 0))
        power = np.bincount(hour[side], weights=(eflux * area)[side], minlength=24)
        peak = int(np.argmax(power))
        assert peak >= 22 or peak < 6, (north, peak, power.round(1))
