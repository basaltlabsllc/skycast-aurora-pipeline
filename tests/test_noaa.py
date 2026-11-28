"""NOAA's aurora map, kept as history for the same ten hours as the frames."""
import copy
import gzip
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

import fetch_prev
import run_aurora
from pipeline import noaa, site
from tests.feeds import RUN, feeds
from tests.test_history import _frame

# SWPC's file as served 2026-11-28 (Observation 17:12Z, Forecast 17:58Z).
REAL = json.loads(gzip.decompress(
    (Path(__file__).parent / "fixtures" / "ovation_aurora_latest.json.gz").read_bytes()))


def _map(obs, fcst, value=7):
    """A small, whole map: every cell 0 except one, at the given ISO times."""
    rows = [[lon, lat, 0] for lon in range(360) for lat in range(-90, 91)]
    rows[0][2] = value
    return {"Observation Time": obs, "Forecast Time": fcst,
            "Data Format": "[Longitude, Latitude, Aurora]", "coordinates": rows, "type": "MultiPoint"}


def test_the_real_file_round_trips_cell_for_cell():
    obs, fcst, grid = noaa.parse(REAL)
    assert site.iso(obs) == REAL["Observation Time"] and site.iso(fcst) == REAL["Forecast Time"]
    back = noaa.decode(noaa.encode(grid))
    for lon, lat, v in REAL["coordinates"]:
        assert back[lat + 90, lon] == v
    assert noaa.encode(grid) == noaa.encode(grid.copy())      # same map, same bytes


def test_a_map_is_a_few_kilobytes():
    # Measured on the real file: 3,525 bytes (the JSON is 925,891 bytes; 170 kB gzipped).
    assert len(noaa.encode(noaa.parse(REAL)[2])) < 6000


@pytest.mark.parametrize("spoil", [
    lambda j: j["coordinates"].pop(),                        # a cell missing
    lambda j: j["coordinates"].__setitem__(1, list(j["coordinates"][0])),  # one cell twice
    lambda j: j["coordinates"][5].__setitem__(2, 12.5),      # not a whole percent
    lambda j: j["coordinates"][5].__setitem__(2, 101),       # past 100
    lambda j: j["coordinates"][5].__setitem__(2, None),
    lambda j: j.__setitem__("Forecast Time", "soon"),
    lambda j: j.pop("Observation Time"),
])
def test_a_map_that_is_not_whole_is_refused(spoil):
    j = copy.deepcopy(REAL)
    spoil(j)
    with pytest.raises(noaa.BadMap):
        noaa.parse(j)


def _base(run=RUN):
    return site.build_site(*feeds(run=run), run.timestamp(), _frame(1.0), log=lambda s: None)


def test_a_build_names_the_map_it_read():
    m, files = _base()
    m2, f2 = noaa.add_maps(m, files, REAL, None, lambda p: None, log=lambda s: None)
    path = "noaa/20261128T171200Z_20261128T175800Z.ova1"
    # RUN is 2026-11-17: this map is from the future, which is fine (it is inside the window's far side)
    assert m2["noaa"] == {"v": 1, "maps": [{"obs": "2026-11-28T17:12:00Z", "fcst": "2026-11-28T17:58:00Z", "path": path}]}
    assert noaa.decode(f2[path])[90 + REAL["coordinates"][0][1], REAL["coordinates"][0][0]] == REAL["coordinates"][0][2]
    assert {k: v for k, v in m2.items() if k != "noaa"} == m      # nothing else changes
    assert f2.items() >= files.items()


def test_no_map_and_no_history_leaves_the_build_alone():
    m, files = _base()
    assert noaa.add_maps(m, files, None, None, lambda p: None) == (m, files)
    bad = copy.deepcopy(REAL)
    bad["coordinates"].pop()
    said = []
    assert noaa.add_maps(m, files, bad, None, lambda p: None, log=said.append) == (m, files)
    assert said and "not recorded" in said[0]


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _chain(n, step_min=10, maps_every_build=True):
    """n builds step_min apart, each with the map SWPC served then (obs = run - 4 min,
    fcst = obs + 45 min), each merged onto the one before."""
    m = files = None
    for k in range(n):
        run = RUN + timedelta(minutes=k * step_min)
        m2, f2 = _base(run)
        obs = run - timedelta(minutes=4)
        latest = _map(_iso(obs), _iso(obs + timedelta(minutes=45)), value=k % 100) if maps_every_build else None
        prev_m, prev_f = (m, files) if m is not None else (None, {})
        if m is not None:
            m2, f2 = site.merge_history(m2, f2, m, files.get)
        m2, f2 = noaa.add_maps(m2, f2, latest, prev_m, prev_f.get, log=lambda s: None)
        m, files = m2, f2
    return m, files


def test_maps_are_kept_for_the_same_ten_hours_as_the_frames():
    m, files = _chain(90)                                    # 14 h 50 min of builds
    run = site.from_iso(m["runTime"])
    fc = [site.from_iso(e["fcst"]) for e in m["noaa"]["maps"]]
    assert fc == sorted(fc)
    assert min(fc) >= run - site.HISTORY_S
    assert min(fc) - (run - site.HISTORY_S) < 600            # and nothing in the window is dropped
    assert sorted(p for p in files if p.startswith("noaa/")) == sorted(e["path"] for e in m["noaa"]["maps"])
    # each kept map is still the bytes its own build recorded
    for e in m["noaa"]["maps"]:
        obs = site.from_iso(e["obs"])
        k = round((obs + 240 - RUN.timestamp()) / 600)
        assert noaa.decode(files[e["path"]])[0, 0] == k % 100


def test_the_same_map_read_twice_is_named_once():
    m, files = _base()
    a, fa = noaa.add_maps(m, files, REAL, None, lambda p: None)
    b, fb = noaa.add_maps(m, files, REAL, a, fa.get)
    assert b["noaa"] == a["noaa"] and len(b["noaa"]["maps"]) == 1


def test_a_build_that_cannot_read_noaa_keeps_the_earlier_maps():
    m, files = _chain(3)
    run = RUN + timedelta(minutes=30)
    m2, f2 = _base(run)
    m3, f3 = noaa.add_maps(m2, f2, None, m, files.get)
    assert m3["noaa"]["maps"] == m["noaa"]["maps"]


def test_an_earlier_map_is_kept_only_under_its_own_name_and_whole():
    m, files = _chain(3)
    entries = m["noaa"]["maps"]
    run = RUN + timedelta(minutes=30)
    m2, f2 = _base(run)
    renamed = copy.deepcopy(m)
    renamed["noaa"]["maps"][0]["path"] = "noaa/../../etc/passwd"
    broken = dict(files)
    broken[entries[1]["path"]] = b"\x1f\x8b" + b"junk"
    wrong_v = copy.deepcopy(m)
    wrong_v["noaa"]["v"] = 2
    got, _ = noaa.add_maps(m2, f2, None, renamed, files.get)
    assert [e["path"] for e in got["noaa"]["maps"]] == [e["path"] for e in entries[1:]]
    got, _ = noaa.add_maps(m2, f2, None, m, broken.get)
    assert [e["path"] for e in got["noaa"]["maps"]] == [entries[0]["path"], entries[2]["path"]]
    assert noaa.add_maps(m2, f2, None, wrong_v, files.get) == (m2, f2)


def _main(out, prev=None, run=RUN, latest=None):
    argv = ["--out", str(out)] + (["--prev", str(prev)] if prev else [])
    return run_aurora.main(argv, fetch=lambda: feeds(run=run), now=run, run_frame=_frame(1.0),
                           fetch_map=lambda: latest)


def test_run_writes_and_then_keeps_the_maps(tmp_path):
    first = _map("2026-11-17T11:59:00Z", "2026-11-17T12:44:00Z", value=9)
    assert _main(tmp_path / "a", latest=first) == 0
    m = json.loads((tmp_path / "a" / "manifest.json").read_text())
    path = m["noaa"]["maps"][0]["path"]
    assert (tmp_path / "a" / path).read_bytes()[:2] == b"\x1f\x8b"
    second = _map("2026-11-17T12:09:00Z", "2026-11-17T12:54:00Z", value=11)
    assert _main(tmp_path / "b", prev=tmp_path / "a", run=RUN + timedelta(minutes=10), latest=second) == 0
    m2 = json.loads((tmp_path / "b" / "manifest.json").read_text())
    assert [e["fcst"] for e in m2["noaa"]["maps"]] == ["2026-11-17T12:44:00Z", "2026-11-17T12:54:00Z"]
    assert (tmp_path / "b" / path).read_bytes() == (tmp_path / "a" / path).read_bytes()


def test_run_without_a_noaa_map_still_builds(tmp_path):
    assert _main(tmp_path / "a", latest=None) == 0
    assert "noaa" not in json.loads((tmp_path / "a" / "manifest.json").read_text())


def test_fetch_prev_copies_the_maps(tmp_path):
    m, files = _chain(2)
    m["noaa"]["maps"].append({"obs": "2026-11-17T12:00:00Z", "fcst": "2026-11-17T12:45:00Z",
                              "path": "noaa/../../x.ova1"})
    served = {"https://x.test/manifest.json": json.dumps(m).encode()}
    served.update({f"https://x.test/{p}": b for p, b in files.items()})

    def get(url):
        if url not in served:
            raise RuntimeError("404")
        return served[url]

    out = tmp_path / "prev"
    assert fetch_prev.main(["--base", "https://x.test", "--out", str(out)], get=get) == 0
    got = sorted(str(p.relative_to(out)) for p in (out / "noaa").iterdir())
    assert got == sorted(e["path"] for e in m["noaa"]["maps"][:2])
    assert not (tmp_path / "x.ova1").exists()
