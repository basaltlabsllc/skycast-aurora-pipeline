"""NOAA's own OVATION aurora map (SWPC ovation_aurora_latest.json), kept as history.

SWPC serves one map at a time: the newest, for about the next hour. Each build
records the map it read and keeps the maps earlier builds recorded, back to
runTime - site.HISTORY_S (the same window the Solar-1 frames keep), so the app
can show NOAA's own chance for a past time. Erik, 2026-11-28: "I'd like to save
the same amount of data for the percent as we do the solar 1 data."

A map file, noaa/<obs>_<fcst>.ova1, is a gzip stream (mtime 0, so the same map
is the same bytes) of:
  b"OVA1", uint16 LE width 360, uint16 LE height 181,
  then width x height bytes: row j is latitude j - 90, column i is longitude i
  degrees east (0..359), each byte NOAA's own percent (0..100) for that cell,
  exactly as published: nothing rounded, filled or blended.
A map NOAA publishes with any other shape (a cell missing or twice, a value
that is not a whole number from 0 to 100, a time that does not parse) is not
recorded: the app would have to invent a value to show it.

The manifest names the maps under "noaa":
  {"v": 1, "maps": [{"obs": iso, "fcst": iso, "path": "noaa/..."}, ...]}
oldest forecast time first. obs and fcst are the file's own Observation Time
and Forecast Time.

SWPC's pages are NWS pages (their footer's Disclaimer is weather.gov/disclaimer,
read 2026-11-28): public domain, usable for any lawful purpose, as long as the
copy is not presented as NOAA's official material or as endorsed by NOAA.
"""
import gzip
import struct
from datetime import datetime, timezone

import numpy as np

from pipeline import site

URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
MAGIC = b"OVA1"
W, H = 360, 181
HEADER = MAGIC + struct.pack("<HH", W, H)
VERSION = 1


class BadMap(Exception):
    """NOAA's file is not a whole 360 x 181 grid of percents with its two times."""


def _iso_s(text):
    t = site.from_iso(text)
    if t is None:
        raise BadMap(f"time {text!r} is not YYYY-MM-DDTHH:MM:SSZ")
    return int(t)


def map_path(obs, fcst):
    """The file name for a map: both of its times, to the second."""
    f = lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"noaa/{f(obs)}_{f(fcst)}.ova1"


def parse(json_obj):
    """(obs, fcst, uint8 grid [H, W]) from SWPC's ovation_aurora_latest.json, or BadMap."""
    if not isinstance(json_obj, dict):
        raise BadMap("not a JSON object")
    obs = _iso_s(json_obj.get("Observation Time"))
    fcst = _iso_s(json_obj.get("Forecast Time"))
    rows = json_obj.get("coordinates")
    if not isinstance(rows, list) or len(rows) != W * H:
        raise BadMap(f"{len(rows) if isinstance(rows, list) else 'no'} cells, not {W * H}")
    grid = np.zeros((H, W), np.uint8)
    seen = np.zeros((H, W), bool)
    for r in rows:
        if not (isinstance(r, list) and len(r) == 3):
            raise BadMap(f"cell {r!r} is not [lon, lat, value]")
        lon, lat, v = r
        ok = all(isinstance(x, (int, float)) and not isinstance(x, bool) and float(x).is_integer()
                 for x in (lon, lat, v))
        if not ok or not (0 <= lon < W and -90 <= lat <= 90 and 0 <= v <= 100):
            raise BadMap(f"cell {r!r} is not a whole lon 0..359, lat -90..90, percent 0..100")
        j, i = int(lat) + 90, int(lon)
        if seen[j, i]:
            raise BadMap(f"cell lon {i} lat {j - 90} is given twice")
        seen[j, i] = True
        grid[j, i] = int(v)
    return obs, fcst, grid


def encode(grid):
    grid = np.asarray(grid)
    if grid.shape != (H, W) or grid.dtype != np.uint8:
        raise ValueError("grid must be uint8 [181, 360]")
    return gzip.compress(HEADER + grid.tobytes(), compresslevel=9, mtime=0)


def decode(data):
    """uint8 grid [H, W] from a map file's bytes, or ValueError."""
    raw = gzip.decompress(data)
    if raw[:8] != HEADER or len(raw) != 8 + W * H:
        raise ValueError("not an OVA1 360 x 181 map")
    return np.frombuffer(raw, np.uint8, offset=8).reshape(H, W)


def _earlier(prev_manifest, read_prev, oldest):
    """{path: (entry, bytes)} for the earlier build's maps still inside the window."""
    block = prev_manifest.get("noaa") if isinstance(prev_manifest, dict) else None
    if not isinstance(block, dict) or block.get("v") != VERSION:
        return {}
    out = {}
    for e in block.get("maps") or []:
        if not isinstance(e, dict):
            continue
        obs, fcst = site.from_iso(e.get("obs")), site.from_iso(e.get("fcst"))
        if obs is None or fcst is None or fcst < oldest:
            continue
        path = map_path(int(obs), int(fcst))
        if e.get("path") != path:               # only the pipeline's own name is ever read
            continue
        data = read_prev(path)
        if not data or data[:2] != b"\x1f\x8b":
            continue
        try:
            decode(data)
        except (ValueError, OSError, EOFError):
            continue
        out[path] = ({"obs": site.iso(obs), "fcst": site.iso(fcst), "path": path}, data)
    return out


def add_maps(manifest, files, latest, prev_manifest, read_prev, keep_s=site.HISTORY_S, log=print):
    """(manifest, files) with the "noaa" block: the earlier build's maps whose
    forecast time is within keep_s of runTime, plus `latest` (SWPC's JSON as
    read this build, or None when it could not be read). A map that is not a
    whole grid of percents is left out and said so in the log. A gap in the
    maps is fine: the app shows no number for a time no map covers.
    Returns the inputs themselves when there is no map to name."""
    oldest = site.from_iso(manifest["runTime"]) - keep_s
    maps = _earlier(prev_manifest, read_prev, oldest)
    if latest is not None:
        try:
            obs, fcst, grid = parse(latest)
        except BadMap as e:
            log(f"NOAA's aurora map not recorded: {e}")
        else:
            if fcst >= oldest:
                path = map_path(obs, fcst)
                maps[path] = ({"obs": site.iso(obs), "fcst": site.iso(fcst), "path": path},
                              encode(grid))
    if not maps:
        return manifest, files
    ordered = sorted(maps.values(), key=lambda m: (m[0]["fcst"], m[0]["obs"]))
    merged = dict(manifest)
    merged["noaa"] = {"v": VERSION, "maps": [e for e, _ in ordered]}
    return merged, {**files, **{e["path"]: data for e, data in ordered}}
