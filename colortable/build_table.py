"""Build colortable.json from NCAR GLOW: volume emission rate of green 557.7 nm,
red 630.0 nm and blue 427.8 nm (photons cm^-3 s^-1) per unit energy flux
(1 erg cm^-2 s^-1) of precipitating electrons, for each average energy and
spectral shape, with GLOW's own no-precipitation airglow subtracted.

Every number in the table is GLOW's own printed output: the precipitation run
minus the no-precipitation run at each GLOW altitude, clamped at 0, linearly
interpolated from GLOW's altitude grid onto the table's 10-km levels and
rounded to 4 significant figures. Nothing is fitted, extrapolated or adjusted.

GLOW source read (NCAR GLOW 0.981 as packaged by space-physics/ncar-glow
("ncarglow" 1.4.0), commit 2c099fdc2bd02b69fd1c3f36e1cc603ad3d34336; paths are
relative to src/ncarglow/ in that repo; driver line numbers are before
colortable/glowpython.patch):

* Driver: fortran/glowpython.f90 (built as glow.bin). Arguments, lines 92-101:
  `yyyyddd UTsec glat glon F107a F107 F107p Ap`, then the precipitation mode
  read at line 104. The default mode (lines 145-157) takes `Ef` (energy flux,
  erg cm-2 s-1), `Ec` (characteristic energy, eV) and `Nbins`; `egrid`
  (fortran/egrid.f90 lines 17-46) builds the energy grid and widths, and
  `maxt(ef, ec, ener, del, nbins, itail, fmono, emono, phitop)`
  (fortran/maxt.f90 lines 28-70) fills the spectrum. Unpatched, the driver
  fixes itail = fmono = emono = 0 (lines 68-70) and calls MAXT only when
  ef > .001 and ec > 1 (line 157).

* colortable/glowpython.patch (applied before the build) adds optional
  arguments 12-14 = itail, fmono, emono, passed to MAXT unchanged, and also
  calls MAXT when fmono > 0 with ef = 0. With no extra arguments, or with
  `0 0 0`, the patched driver's output is byte-identical to the unpatched one
  (checked). Every shape is therefore made by MAXT itself and normalised by
  MAXT to its flux on GLOW's own bin widths `del` from egrid.

* maxwellian (diffuse): `Ef=1 Ec=E/2 Nbins` -- MAXT with itail=0:
  phi(E) = (E/Ec) exp(-E/Ec), scaled so sum(phi * E * del) * 1.6022e-12 = Ef
  (maxt.f90 lines 49-59). Its mean energy is 2 Ec, and the table's energy
  axis is average energy (OVATION Prime's eavg = energy flux / number flux),
  hence Ec = E/2.

* gaussian (mono): GLOW's own monoenergetic spectrum, not a Gaussian (GLOW
  has none): `Ef=0 Ec=E Nbins itail=0 fmono=1 emono=E` -- MAXT puts
  fmono / (1.6022e-12 * del(k) * ener(k)) into the single bin k whose
  [ener - del/2, ener + del/2] contains emono (maxt.f90 lines 61-66); the
  Maxwellian part is Ef * ... / te = 0. Its mean energy is that bin's centre.

* broadband (wave): an approximation of a broadband spectrum: GLOW's
  Maxwellian with its Meier et al. (1989) low-energy tail, `Ef=1 Ec
  Nbins itail=1 fmono=0 emono=0` -- phi gains
  0.4 * exp(-1) * (Ec/E) * exp(-E/b), b = 0.8 Ec below 500 eV else
  0.1 Ec + 350 (maxt.f90 lines 41-45, 53), before the same normalisation.
  The tail lowers the mean energy (to 0.42-0.57 x E at Ec = E/2), so Ec is
  chosen per axis energy by bisection in log Ec until the mean energy of the
  spectrum GLOW prints (energy flux / number flux on GLOW's own widths) is E
  within 2 % (stopping at 0.2 %); the achieved mean is in meanEnergyKeV and
  the Ec used in broadbandCharacteristicEnergyKeV.

* Output: glowpython.f90 lines 223-226 print ZETA, the volume emission rates
  in photons cm-3 s-1, one row per altitude (`(1x,f5.1,15f8.2)`) under the
  header `Z 3371 4278 5200 5577 6300 ...`; the 4278, 5577 and 6300 Angstrom
  columns are blue, green and red. ZETA is filled by GLOW's `gchem`
  (fortran/gchem.f90). The printed format carries 2 decimal places, so the
  table's absolute resolution is 0.01 photons cm-3 s-1. Lines 234-237 print
  the energy grid and the incident spectrum (`f15.1`), which the builder
  reads back to check each shape's energy flux and mean energy.

* Background: the same date and place with `-noprecip` (lines 130-143; phitop
  = 0) gives GLOW's airglow without precipitation, subtracted from every
  profile so that only light the precipitation causes scales with flux.

Conditions (task brief): 65 deg N geographic, longitude 0, 00:00 UT (local
solar time 0, glowpython.f90 lines 164-165), day 079 of 2026 (March equinox),
F10.7 = F10.7a = previous-day F10.7 = 120, Ap = 15, 250 energy bins (the
ncarglow examples' value). At this time and place the solar zenith angle is
about 115 deg, above GLOW's 1.85 rad (106 deg) photoionisation cut-off
(fortran/glow.f90 line 173), so there is no solar input.

GLOW licence: fortran/Glowlicense.txt, "GLOW MODEL OPEN SOURCE ACADEMIC
RESEARCH LICENSE AGREEMENT" (research, academic and non-profit use; no sale).
"This software is part of the GLOW model. Use is governed by the Open Source
Academic Research License Agreement contained in the file glowlicense.txt."
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ENERGIES_KEV = np.logspace(math.log10(0.1), math.log10(30.0), 16)
ALT_KM = np.arange(80, 501, 10)
SHAPES = ("maxwellian", "gaussian", "broadband")
LINES = {"green": "5577", "red": "6300", "blue": "4278"}
ERG_PER_EV = 1.6022e-12  # maxt.f90 line 54

# Run conditions (see the module docstring).
BASE_ARGS = ["2026079", "0", "65", "0", "120", "120", "120", "15"]
NBINS = 250
NALT = 250  # glowpython.f90 line 79

SHAPE_TEXT = {
    "maxwellian": "GLOW Maxwellian (maxt, itail=0), characteristic energy = axis energy / 2, "
                  "so its mean energy is the axis energy",
    "gaussian": "GLOW monoenergetic, not a Gaussian: all flux in the one GLOW energy bin "
                "containing the axis energy (maxt fmono/emono); mean energy = that bin's centre. "
                "Its layers are 7-13 km thick, so on the 10-km levels the peak altitude is "
                "resolved only to the level; columns agree with GLOW's own grid within 0.5 %",
    "broadband": "approximation of a broadband spectrum: GLOW Maxwellian with GLOW's Meier et al. "
                 "(1989) low-energy tail (maxt itail=1); characteristic energy chosen by bisection "
                 "so the mean energy of the spectrum GLOW builds is the axis energy within 2 % "
                 "(meanEnergyKeV, broadbandCharacteristicEnergyKeV). Holding the mean at the axis "
                 "energy against the soft tail gives it a hard core: energy-weighted mean 2.6-3.7 x "
                 "the axis energy (Maxwellian 1.5 x), so it emits lower and less red than a "
                 "Maxwellian of the same average energy, the opposite of a soft sub-keV broadband "
                 "population; the app does not use this shape for now. At 20-30 keV part of the "
                 "blue and green lies below the 80 km floor (at 30 keV the 80-km blue value is 0.86 "
                 "of its peak; at least 13 % of the blue column is in 73.5-80 km)",
}
PRECISION = ("GLOW prints emission rates with 2 decimal places (glowpython.f90 format f8.2): "
             "absolute resolution 0.01 photons cm-3 s-1 before background subtraction; "
             "values are then rounded to 4 significant figures")


def sig4(x: float) -> float:
    return float(f"{x:.4g}")


def egrid_widths() -> np.ndarray:
    """GLOW's own bin widths, egrid.f90 lines 26-38."""
    n = np.arange(1, NBINS + 1)
    edge = np.where(n <= 21, 0.5 * n, np.exp(0.05 * (n + 26)))
    d = np.empty(NBINS)
    d[0] = 0.5
    d[1:] = np.diff(edge)
    return d


def parse(stdout: str) -> dict:
    """Altitudes, the three lines of ZETA, and the printed energy grid and spectrum."""
    lines = stdout.splitlines()
    hdr_idx = 2 + NALT  # run line, density header, NALT density rows
    header = lines[hdr_idx].split()
    if header[0] != "Z":
        raise ValueError(f"unexpected GLOW output: VER header is {lines[hdr_idx]!r}")
    rows = lines[hdr_idx + 1 : hdr_idx + 1 + NALT]
    table = np.array([[float(v) for v in r.split()] for r in rows])
    if table.shape != (NALT, len(header)):
        raise ValueError(f"VER block shape {table.shape}, header has {len(header)} columns")
    dens_alt = np.array([float(r.split()[0]) for r in lines[2 : 2 + NALT]])
    alt = table[:, 0]
    if not np.array_equal(alt, dens_alt) or np.any(np.diff(alt) <= 0):
        raise ValueError("VER altitudes do not match the density block or are not increasing")
    k = hdr_idx + 1 + 3 * NALT  # after ZETA, production and loss
    if int(lines[k]) != NBINS:
        raise ValueError(f"expected {NBINS} energy bins, GLOW printed {lines[k]!r}")
    ener = np.array(lines[k + 1].split(), dtype=float)
    phi = np.array(lines[k + 2].split(), dtype=float)
    return {
        "alt": alt,
        "ver": {name: table[:, header.index(col)] for name, col in LINES.items()},
        "ener": ener,
        "phi": phi,
    }


def run_glow(glow_exe: Path, mode_args: list[str]) -> dict:
    cmd = [str(glow_exe), *BASE_ARGS, *mode_args]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        raise RuntimeError(f"glow.bin exit {res.returncode}: {res.stderr.strip()[-500:]}")
    return parse(res.stdout)


def shape_args(shape: str, eavg_kev: float, ec_ev: float | None = None) -> list[str]:
    e_ev = eavg_kev * 1000.0
    if shape == "maxwellian":
        return ["1", repr(e_ev / 2), str(NBINS)]
    if shape == "gaussian":
        return ["0", repr(e_ev), str(NBINS), "0", "1", repr(e_ev)]
    if shape == "broadband":
        return ["1", repr(ec_ev), str(NBINS), "1", "0", "0"]
    raise ValueError(shape)


def check_spectrum(shape: str, run: dict, widths: np.ndarray) -> tuple[float, float]:
    """Energy flux of the printed spectrum on GLOW's own widths must be 1 erg;
    returns the spectrum's mean energy (energy flux / number flux) and its
    energy-flux-weighted mean energy, both in keV."""
    phi, ener = run["phi"], run["ener"]
    eflux = float((phi * ener * widths).sum() * ERG_PER_EV)
    if abs(eflux - 1.0) > 1e-3:
        raise ValueError(f"{shape}: printed spectrum carries {eflux} erg, not 1")
    if shape == "gaussian" and np.count_nonzero(phi) != 1:
        raise ValueError(f"gaussian: {np.count_nonzero(phi)} bins carry flux, not 1")
    mean = (phi * ener * widths).sum() / (phi * widths).sum() / 1000.0
    ew_mean = (phi * ener**2 * widths).sum() / (phi * ener * widths).sum() / 1000.0
    return float(mean), float(ew_mean)


BROADBAND_TOL = 0.02   # ruling: achieved mean within 2 % of the axis value
BISECT_STOP = 0.002    # stop bisecting once within 0.2 %


def broadband_run(glow_exe: Path, eavg_kev: float, widths: np.ndarray) -> tuple[dict, float]:
    """Characteristic energy for the tailed Maxwellian chosen by bisection in
    log Ec so that the mean energy of the spectrum GLOW prints equals the axis
    value. Returns GLOW's run and the Ec used (eV)."""
    target = eavg_kev
    lo, hi = math.log(eavg_kev * 1000.0 / 4), math.log(eavg_kev * 1000.0 * 4)

    def mean_at(log_ec: float) -> tuple[dict, float]:
        # Probe only: the final run's 1-erg check is made in build(). (At the
        # bracket's top, Ec = 4E, GLOW's f15.1 print drops enough far tail
        # that the printed flux reads 0.9986 erg.)
        run = run_glow(glow_exe, shape_args("broadband", eavg_kev, math.exp(log_ec)))
        phi, ener = run["phi"], run["ener"]
        return run, float((phi * ener * widths).sum() / (phi * widths).sum() / 1000.0)

    r_lo, m_lo = mean_at(lo)
    r_hi, m_hi = mean_at(hi)
    if not (m_lo < target < m_hi):
        raise ValueError(f"broadband {eavg_kev} keV: mean {m_lo}-{m_hi} keV over Ec "
                         f"{math.exp(lo):.4g}-{math.exp(hi):.4g} eV does not bracket the axis")
    best = min(((r_lo, m_lo, lo), (r_hi, m_hi, hi)), key=lambda x: abs(x[1] / target - 1))
    for _ in range(40):
        mid = (lo + hi) / 2
        r, m = mean_at(mid)
        if abs(m / target - 1) < abs(best[1] / target - 1):
            best = (r, m, mid)
        if abs(m / target - 1) < BISECT_STOP:
            break
        lo, hi = (mid, hi) if m < target else (lo, mid)
    run, m, log_ec = best
    if abs(m / target - 1) > BROADBAND_TOL:
        raise ValueError(f"broadband {eavg_kev} keV: best mean {m} keV is more than 2 % off")
    return run, math.exp(log_ec)


def bracket(alt: np.ndarray) -> slice:
    lo, hi = np.searchsorted(alt, ALT_KM[0], side="right") - 1, np.searchsorted(alt, ALT_KM[-1])
    if lo < 0 or hi >= alt.size:
        raise ValueError("GLOW's grid does not bracket the table's altitude range")
    return slice(lo, hi + 1)


def on_table_levels(alt: np.ndarray, ver: np.ndarray) -> np.ndarray:
    """Linear interpolation between the two GLOW levels around each table level.
    Both bracketing GLOW values must be finite; nothing is extrapolated."""
    if not np.all(np.isfinite(ver[bracket(alt)])):
        raise ValueError("GLOW returned a non-finite emission rate inside 80-500 km")
    return np.interp(ALT_KM, alt, ver)


def column_rayleigh(alt: np.ndarray, ver: np.ndarray) -> float:
    """Column brightness over 80-500 km on GLOW's own grid, R (1e6 photons cm-2 s-1)."""
    sel = (alt >= ALT_KM[0]) & (alt <= ALT_KM[-1])
    return float(np.trapezoid(ver[sel], alt[sel] * 1e5) / 1e6)


def build(glow_exe: Path) -> dict:
    widths = egrid_widths()
    bg = run_glow(glow_exe, ["-noprecip", str(NBINS)])
    if np.any(bg["phi"] != 0):
        raise ValueError("-noprecip run printed a non-zero spectrum")
    out: dict = {
        "energiesKeV": [sig4(e) for e in ENERGIES_KEV],
        "altKm": [int(a) for a in ALT_KM],
        "shapes": {},
        "shapeNotes": dict(SHAPE_TEXT),
        "meanEnergyKeV": {},
        "energyWeightedMeanKeV": {},
        "broadbandCharacteristicEnergyKeV": [],
        "background": {
            "unit": "R, column 80-500 km on GLOW's grid, subtracted from every profile",
            **{k: sig4(column_rayleigh(bg["alt"], bg["ver"][k])) for k in LINES},
        },
        "precision": PRECISION,
        "source": "NCAR GLOW 0.981 (space-physics/ncar-glow 2c099fdc + colortable/glowpython.patch); "
                  "65N 0E 2026-079 00 UT, F10.7 120, Ap 15, 1 erg cm-2 s-1",
    }
    for shape in SHAPES:
        per_line: dict[str, list] = {k: [] for k in LINES}
        means, ew_means = [], []
        for e in ENERGIES_KEV:
            if shape == "broadband":
                run, ec_ev = broadband_run(glow_exe, float(e), widths)
                out["broadbandCharacteristicEnergyKeV"].append(sig4(ec_ev / 1000.0))
            else:
                run = run_glow(glow_exe, shape_args(shape, float(e)))
            if not np.array_equal(run["alt"], bg["alt"]):
                raise ValueError("precipitation and background runs use different altitudes")
            mean, ew_mean = check_spectrum(shape, run, widths)
            means.append(sig4(mean))
            ew_means.append(sig4(ew_mean))
            for k in LINES:
                net = net_emission(run["alt"], run["ver"][k], bg["ver"][k],
                                   f"{shape} {float(e):.4g} keV {k}")
                per_line[k].append([sig4(v) for v in on_table_levels(run["alt"], net)])
        out["shapes"][shape] = per_line
        out["meanEnergyKeV"][shape] = means
        out["energyWeightedMeanKeV"][shape] = ew_means
    return out


NEGATIVE_TOL = 0.05  # photons cm-3 s-1: 5 x GLOW's print resolution


def net_emission(alt, ver, bg_ver, label):
    """Precipitation minus background, clamped at 0. A residual below
    -NEGATIVE_TOL inside the table's 80-500 km is a real difference, not print
    rounding, and the clamp must not hide it: raise with where it happened."""
    resid = np.asarray(ver, dtype=float) - np.asarray(bg_ver, dtype=float)
    alt = np.asarray(alt, dtype=float)
    bad = (alt >= ALT_KM[0]) & (alt <= ALT_KM[-1]) & (resid < -NEGATIVE_TOL)
    if bad.any():
        i = int(np.flatnonzero(bad)[np.argmin(resid[bad])])
        raise ValueError(f"{label}: precipitation minus background is {resid[i]:.3g} "
                         f"photons cm-3 s-1 at {alt[i]:.4g} km")
    return np.maximum(resid, 0.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--glow-exe", type=Path, required=True,
                    help="glow.bin built with colortable/glowpython.patch")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    table = build(args.glow_exe.resolve())
    args.out.write_text(json.dumps(table, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
