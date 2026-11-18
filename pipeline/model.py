"""OVATION Prime run for one coupling value, resampled onto a geographic grid.

Units, as read in OvationPyme (github.com/lkilcommons/OvationPyme @ dd774c3):
  energy flux   erg cm^-2 s^-1 -- ovationpyme/ovation_prime.py:333 turns it into
                eV with `/1.6e-12` (erg per eV; the comment there says "Joules",
                the constant is the erg one), and ovation_utilities.py:224-227
                feeds numflux*eavg*1.6022e-9 (keV->erg) into Robinson's formula,
                which takes erg cm^-2 s^-1. The caps in correct_flux
                (ovation_prime.py:733-736: >10 -> 0.5, >5 -> 5) fit erg too.
  number flux   cm^-2 s^-1 -- ovation_utilities.py:208 ("#/cm^2/s").
  average energy keV -- ovation_prime.py:333-334: (eflux / 1.6e-12) / numflux
                is eV, then `/1000.` "eV to keV".
                AverageEnergyEstimator.get_eavg_for_time (ovation_prime.py:317-349)
                cannot be called with our own dF (it goes through get_flux_for_time,
                which downloads OMNI data first), so its rules are applied here
                unmodified, to the energy and number flux sampled onto the
                geographic grid: the same 1.6e-12 and /1000. conversion, eavg = 0
                where number flux < AverageEnergyEstimator's numflux_threshold
                default (5e7, read from its signature), capped at 30 keV, 0 below
                0.2 keV. Note the order differs from upstream: upstream applies
                these energy rules on the magnetic grid, whereas here they are
                applied after the energy and number flux are sampled to the
                geographic grid.

eavg == 0 where eflux > 0 means "energy unknown", NOT low energy: the model's
energy and number regressions disagree in some mono/wave bins (number flux 0 or
tiny under real energy flux, up to ~1.7 erg). Energy flux is never altered;
consumers must not colour such cells by energy.

Model grid (SeasonalFluxEstimator.get_gridded_flux): 80 magnetic latitudes per
hemisphere, 50..90 as OvationPyme labels them (ovation_prime.py:585-586), by
96 magnetic local times. OvationPyme labels the MLT bins linspace(0, 24, 96)
(line 588), but its own comment calls them 15-minute bins (line 568) and the
last bin is not a copy of the first (measured: |col0 - col95| equals the
neighbour-to-neighbour step), so bin i is MLT i * 0.25 h and the grid wraps
from 23.75 h back to 0 h.
"""
import inspect
from datetime import datetime, timezone
from functools import lru_cache

import aacgmv2
import numpy as np
from ovationpyme import ovation_prime

TYPES = {"diffuse": "diff", "mono": "mono", "wave": "wave"}
# AverageEnergyEstimator's rules (ovation_prime.py:317, 333-334, 336-349)
NUMFLUX_THRESHOLD = inspect.signature(
    ovation_prime.AverageEnergyEstimator.__init__).parameters["numflux_threshold"].default
ERG_PER_EV = 1.6e-12
EAVG_MAX_KEV = 30.0
EAVG_MIN_KEV = 0.2
ALT_KM = 110.0                       # OvationPyme's own height for AACGM (ovation_prime.py:293)
LAT = np.arange(-90.0, 91.0)         # row 0 = -90
LON = np.arange(-180.0, 180.0)       # col 0 = -180
N_MLT = 96
MLT_STEP_H = 24.0 / N_MLT


@lru_cache(maxsize=None)
def _estimators():
    """FluxEstimators per (type, 'energy'|'number'); loading the tables is the slow part."""
    return {(name, kind): ovation_prime.FluxEstimator(atype, kind)
            for name, atype in TYPES.items() for kind in ("energy", "number")}


def _hemisphere_grids(est, dF, doy):
    """Season-weighted, hemisphere-combined flux for N and S, as
    FluxEstimator.get_flux_for_time does (ovation_prime.py:503-533), but with our
    own dF: that method always fetches OMNI data for dF first (line 509)."""
    w_n = est.season_weights(doy)
    w_s = est.season_weights(365.0 - doy)
    union = {s: 1.0 if (w_n[s] or w_s[s]) else 0.0 for s in w_n}
    mlats, _, fluxN, fluxS = est.get_season_fluxes(dF, union)
    out = []
    for w in (w_n, w_s):
        g = np.zeros_like(mlats, dtype=float)
        for season, W in w.items():
            if W:
                g += W * (fluxN[season] + fluxS[season]) / 2.0
        out.append(g)
    return mlats[:, 0].astype(float), out[0], out[1]


@lru_cache(maxsize=8)
def _magnetic_coords(when):
    """AACGM-v2 MLAT and MLT of every geographic cell centre at `when`.

    aacgmv2.convert_mlt costs ~0.2 ms per element (13 s for this grid), but for
    one time MLT is exactly linear in magnetic longitude (measured: max
    difference 7e-15 h over 4000 random longitudes at three times), so it is
    called once, for longitude 0, and the rest is ref + mlon / 15."""
    lat, lon = np.meshgrid(LAT, LON, indexing="ij")
    mlat, mlon, _ = aacgmv2.convert_latlon_arr(lat.ravel(), lon.ravel(), ALT_KM,
                                               when, method_code="G2A")
    ref = float(aacgmv2.convert_mlt(0.0, when, m2a=False)[0])
    mlt = np.mod(ref + np.asarray(mlon, dtype=float) / 15.0, 24.0)
    return mlat.reshape(lat.shape), mlt.reshape(lat.shape)


def _sample(grid, mlat_nodes, abs_mlat, mlt):
    """Bilinear sample of grid[mlat, mlt]; MLT wraps; 0 equatorward of the grid."""
    inside = np.isfinite(abs_mlat) & np.isfinite(mlt) & (abs_mlat >= mlat_nodes[0])
    out = np.zeros(abs_mlat.shape)
    m = np.clip(abs_mlat[inside], mlat_nodes[0], mlat_nodes[-1])
    fj = np.interp(m, mlat_nodes, np.arange(mlat_nodes.size))
    j0 = np.minimum(np.floor(fj).astype(int), mlat_nodes.size - 2)
    wj = fj - j0
    fi = np.mod(mlt[inside], 24.0) / MLT_STEP_H
    i0 = np.floor(fi).astype(int) % N_MLT
    wi = fi - np.floor(fi)
    i1 = (i0 + 1) % N_MLT
    out[inside] = ((1 - wj) * ((1 - wi) * grid[j0, i0] + wi * grid[j0, i1])
                   + wj * ((1 - wi) * grid[j0 + 1, i0] + wi * grid[j0 + 1, i1]))
    return out


def _geo(est, dF, doy, mlat, mlt):
    nodes, g_n, g_s = _hemisphere_grids(est, dF, doy)
    north = mlat >= 0
    a = np.abs(mlat)
    return np.where(north, _sample(g_n, nodes, a, mlt), _sample(g_s, nodes, a, mlt))


def average_energy(eflux, nflux):
    """keV, by AverageEnergyEstimator.get_eavg_for_time's rules, in its order;
    0 = energy unknown."""
    with np.errstate(divide="ignore", invalid="ignore"):
        eavg = (eflux / ERG_PER_EV) / nflux
    eavg = eavg / 1000.0
    eavg[nflux < NUMFLUX_THRESHOLD] = 0.0
    eavg[eavg > EAVG_MAX_KEV] = EAVG_MAX_KEV
    eavg[eavg < EAVG_MIN_KEV] = 0.0
    return eavg


def run_frame(dF: float, when: datetime) -> dict:
    """OVATION Prime at coupling dF for time `when` on a 1-degree geographic grid.

    Returns {'diffuse'|'mono'|'wave': (eflux, eavg)}, each (181, 360) with
    row 0 = 90 S and col 0 = 180 W; eflux in erg cm^-2 s^-1, eavg in keV
    (0 = energy unknown).
    """
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc).replace(tzinfo=None)
    doy = when.timetuple().tm_yday
    mlat, mlt = _magnetic_coords(when)
    ests = _estimators()
    frame = {}
    for name in TYPES:
        eflux = _geo(ests[(name, "energy")], dF, doy, mlat, mlt)
        nflux = _geo(ests[(name, "number")], dF, doy, mlat, mlt)
        frame[name] = (eflux, average_energy(eflux, nflux))
    return frame
