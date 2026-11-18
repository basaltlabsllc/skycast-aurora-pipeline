"""Checks on the GLOW colour table (colortable/build_table.py's output).

Reads ./colortable.json, or the file named by $COLORTABLE; skips when the file
is absent locally (it is produced by the colortable workflow, which builds
GLOW). With $CI set, a missing file fails every test instead.
"""

import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

PATH = Path(os.environ.get("COLORTABLE", Path(__file__).resolve().parent.parent / "colortable.json"))

_spec = importlib.util.spec_from_file_location(
    "build_table", Path(__file__).resolve().parent.parent / "colortable" / "build_table.py")
build_table = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_table)


@pytest.fixture(scope="module")
def table():
    if not PATH.is_file():
        if os.environ.get("CI"):
            pytest.fail(f"{PATH} not built (CI is set, so a missing table fails)")
        pytest.skip(f"{PATH} not built")
    return json.loads(PATH.read_text())


def test_net_emission_clamps_print_rounding_only():
    alt = np.array([75.0, 80.0, 120.0, 500.0, 510.0])
    bg = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    # -0.04 inside the table is print rounding: clamped to 0, no error.
    # -5 outside 80-500 km is not the table's business.
    ver = np.array([-4.0, 0.96, 3.0, 1.0, -4.0])
    net = build_table.net_emission(alt, ver, bg, "maxwellian 3.064 keV green")
    assert np.array_equal(net, [0.0, 0.0, 2.0, 0.0, 0.0])


def test_net_emission_raises_on_real_negative():
    alt = np.array([80.0, 120.0, 300.0, 500.0])
    bg = np.array([1.0, 1.0, 2.0, 1.0])
    ver = np.array([1.0, 1.0, 1.5, 0.9])
    with pytest.raises(ValueError, match=r"broadband 30 keV red.*-0\.5.*300 km"):
        build_table.net_emission(alt, ver, bg, "broadband 30 keV red")


def _shape(table, name):
    if name not in table["shapes"]:
        pytest.fail(f"shape {name} missing")
    return {k: np.array(v, dtype=float) for k, v in table["shapes"][name].items()}


def _energy_index(table, kev):
    e = np.array(table["energiesKeV"])
    return int(np.argmin(np.abs(np.log(e / kev))))


def test_axes(table):
    e = np.array(table["energiesKeV"])
    assert e.size == 16
    # log-spaced 0.1-30, stored to 4 significant figures (relative error <= 5e-4)
    assert np.allclose(e, np.logspace(math.log10(0.1), math.log10(30), 16), rtol=5e-4, atol=0)
    assert table["altKm"] == list(range(80, 501, 10))
    assert set(table["shapes"]) == {"maxwellian", "gaussian", "broadband"}
    for lines in table["shapes"].values():
        assert set(lines) == {"green", "red", "blue"}
        for v in lines.values():
            assert np.array(v).shape == (16, 43)


def test_green_peak_altitude(table):
    m = _shape(table, "maxwellian")
    i = _energy_index(table, 3.0)
    peak = table["altKm"][int(np.argmax(m["green"][i]))]
    assert 100 <= peak <= 130


def test_soft_electrons_make_red_high(table):
    m = _shape(table, "maxwellian")
    alt = np.array(table["altKm"], dtype=float)
    soft = _energy_index(table, 0.3)
    assert alt[np.argmax(m["red"][soft])] > 200
    assert np.trapezoid(m["red"][soft], alt) > np.trapezoid(m["green"][soft], alt)
    hard = _energy_index(table, 10.0)
    assert np.trapezoid(m["green"][hard], alt) > 10 * np.trapezoid(m["red"][hard], alt)


def test_non_negative_finite(table):
    for lines in table["shapes"].values():
        for v in lines.values():
            a = np.array(v, dtype=float)
            assert np.all(np.isfinite(a)) and np.all(a >= 0)


def _column_r(table, values):
    return np.trapezoid(np.asarray(values, dtype=float), np.array(table["altKm"], dtype=float) * 1e5) / 1e6


def test_record_fields(table):
    assert set(table["shapeNotes"]) == {"maxwellian", "gaussian", "broadband"}
    assert "approximation" in table["shapeNotes"]["broadband"]
    assert "monoenergetic" in table["shapeNotes"]["gaussian"]
    assert isinstance(table["precision"], str) and "0.01" in table["precision"]
    for field in ("meanEnergyKeV", "energyWeightedMeanKeV"):
        assert set(table[field]) == {"maxwellian", "gaussian", "broadband"}
        for shape, means in table[field].items():
            assert len(means) == 16 and all(m > 0 for m in means), (field, shape)


def test_background_subtracted(table):
    # GLOW's no-precipitation airglow (about 31 R of red at these conditions)
    # was 88 % of the 30 keV Maxwellian's red column before subtraction. With
    # it removed, the 30 keV red column must be well under that floor.
    bg_red = table["background"]["red"]
    assert bg_red > 0
    hard = _energy_index(table, 30.0)
    red30 = _column_r(table, table["shapes"]["maxwellian"]["red"][hard])
    assert red30 < 0.5 * bg_red


def _fwhm(alt, v):
    above = alt[v >= v.max() / 2]
    return above.max() - above.min()


def test_mono_narrower_than_maxwellian(table):
    # At the same axis energy (average energy per electron) GLOW's
    # monoenergetic layer is narrower than the Maxwellian's. On the 10-km
    # levels the widths tie where both layers are a level or two thick
    # (0.1, 6.6, 20.5 and 30 keV), so strictly narrower is asserted at ~3 keV
    # and never wider everywhere.
    alt = np.array(table["altKm"], dtype=float)
    mono = np.array(table["shapes"]["gaussian"]["green"], dtype=float)
    maxw = np.array(table["shapes"]["maxwellian"]["green"], dtype=float)
    i = _energy_index(table, 3.0)
    assert _fwhm(alt, mono[i]) < _fwhm(alt, maxw[i])
    for k in range(len(table["energiesKeV"])):
        assert _fwhm(alt, mono[k]) <= _fwhm(alt, maxw[k]), table["energiesKeV"][k]


def test_mono_peak_matches_maxwellian_of_equal_energy_weighted_mean(table):
    # Deposition follows energy flux: mono at E peaks where the Maxwellian
    # whose energy-weighted mean energy is closest to E peaks, within one level.
    alt = np.array(table["altKm"], dtype=float)
    ew = np.array(table["energyWeightedMeanKeV"]["maxwellian"], dtype=float)
    mono = np.array(table["shapes"]["gaussian"]["green"], dtype=float)
    maxw = np.array(table["shapes"]["maxwellian"]["green"], dtype=float)
    for k, e in enumerate(table["energiesKeV"]):
        j = int(np.argmin(np.abs(np.log(ew / e))))
        assert abs(alt[mono[k].argmax()] - alt[maxw[j].argmax()]) <= 10, e


def test_mean_energy_matches_axis(table):
    e = np.array(table["energiesKeV"], dtype=float)
    bb = np.array(table["meanEnergyKeV"]["broadband"], dtype=float)
    assert np.all(np.abs(bb / e - 1) <= 0.02)
    mx = np.array(table["meanEnergyKeV"]["maxwellian"], dtype=float)
    assert np.all(np.abs(mx / e - 1) <= 0.02)
    assert len(table["broadbandCharacteristicEnergyKeV"]) == 16
