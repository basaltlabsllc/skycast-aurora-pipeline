"""One pipeline run: SWPC feeds -> a frame per 10 minutes + manifest.json.

Times:
  runTime      when the run started (UTC, whole seconds).
  swNewest     L1 time of the newest usable solar-wind sample.
  swArrival    swNewest + the L1 -> Earth delay (newest propagated row).
  windowStart  runTime - 60 min, rounded down to the 10-min step.
  windowEnd    swArrival rounded down to the step.
A frame at target time t (at Earth) uses only wind that has arrived by t, i.e.
L1 samples with time <= t - delay; its swUsedTo is the newest such sample. Before
each frame the hours that frame averages, (t - delay - 4 h, t - delay], are
checked for holes longer than 15 min (SolarWindGap).

History (merge_history): a build is extended backwards with frames an earlier
build published, back to runTime - 6 h, each kept byte for byte as it was last
built (with the L1 -> Earth delay measured near its own time, not today's).
windowStart then names the oldest kept frame and liveStart the first frame
this run built.
"""
from datetime import datetime, timezone

import numpy as np

from pipeline import encode, solarwind

STEP_S = 600
WINDOW_BACK_S = 3600
HISTORY_S = 6 * 3600
AVERAGE_S = solarwind.N_HOURS_IN_AVERAGE * 3600
OVATIONPYME_COMMIT = "dd774c317d628e5cc9f72d58612282c118681735"
MODEL = f"ovation-prime-2010 (OvationPyme {OVATIONPYME_COMMIT})"


class NoAurora(Exception):
    """Every frame's energy flux is zero in all three types: nothing to publish."""


def floor_step(ts):
    return int(ts // STEP_S) * STEP_S


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def from_iso(text):
    """Unix seconds for a manifest time, or None when it is not one."""
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def frame_path(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("frames/%Y%m%dT%H%MZ.aur1")


def window(run_time, sw_arrival):
    """(windowStart, windowEnd) in unix seconds."""
    return floor_step(run_time - WINDOW_BACK_S), floor_step(sw_arrival)


def build_manifest(run_time, spacecraft, sw_newest, sw_arrival, frames):
    """manifest.json as a dict. Times are unix seconds; frames is a list of
    (t, sw_used_to), one per step from windowStart to windowEnd."""
    start, end = window(run_time, sw_arrival)
    ts = [t for t, _ in frames]
    if ts != list(range(start, end + 1, STEP_S)):
        raise ValueError("frames do not cover windowStart..windowEnd at the step")
    return {
        "v": 1,
        "model": MODEL,
        "runTime": iso(run_time),
        "spacecraft": spacecraft,
        "swNewest": iso(sw_newest),
        "swArrival": iso(sw_arrival),
        "windowStart": iso(start),
        "windowEnd": iso(end),
        "stepMin": STEP_S // 60,
        "enc": {"eflux": list(encode.EFLUX_RANGE), "eavg": list(encode.EAVG_RANGE)},
        "frames": [{"t": iso(t), "path": frame_path(t), "swUsedTo": iso(u)}
                   for t, u in frames],
    }


def build_site(wind_rows, mag_rows, propagated_rows, run_time, run_frame, log=print):
    """Everything a run publishes, in memory: (manifest dict, {path: bytes}).

    run_frame(dF, datetime) -> {'diffuse'|'mono'|'wave': (eflux, eavg)}.
    Raises solarwind.SolarWindGap / BadArrivalDelay when the wind cannot support
    every frame, NoAurora when no frame has energy flux at or above the encoder's floor.
    """
    run_time = int(run_time)
    delay = solarwind.arrival_delay_s(propagated_rows)
    series = solarwind.load_series(wind_rows, mag_rows)
    if series.t.size == 0:
        raise solarwind.SolarWindGap("no usable active solar-wind rows")
    sw_newest = float(series.t[-1])
    sw_arrival = sw_newest + delay
    start, end = window(run_time, sw_arrival)
    if end < start:
        raise solarwind.SolarWindGap(
            f"newest wind {iso(sw_newest)} arrives {iso(sw_arrival)}, "
            f"before the window starts {iso(start)}")
    frames, files, any_aurora = [], {}, False
    for t in range(start, end + 1, STEP_S):
        l1_to = t - delay
        solarwind.check_gaps(series, l1_to - AVERAGE_S, l1_to)
        used = series.t[series.t <= l1_to]
        dF = solarwind.coupling_at(series, t, delay)
        types = run_frame(dF, datetime.fromtimestamp(t, timezone.utc))
        # Only energy flux at or above the encoder's floor is carried in a frame.
        any_aurora |= any(np.any(np.asarray(types[k][0]) >= encode.EFLUX_RANGE[0])
                          for k in encode.TYPES)
        files[frame_path(t)] = encode.encode_frame(types)
        frames.append((t, float(used[-1])))
        log(f"{iso(t)}  dF {dF:8.1f}  wind to {iso(used[-1])}")
    if not any_aurora:
        raise NoAurora(f"every frame's energy flux is below {encode.EFLUX_RANGE[0]} erg cm-2 s-1")
    manifest = build_manifest(run_time, series.spacecraft, sw_newest, sw_arrival, frames)
    return manifest, files


def merge_history(manifest, files, prev_manifest, read_prev, keep_s=HISTORY_S):
    """(manifest, files) extended backwards with an earlier build's frames.

    prev_manifest: the manifest that build published (any JSON value);
    read_prev(path) -> that build's bytes for a frame path, or None.
    Walks back one step at a time from the frame before this build's first and
    stops at the first step that is missing, unreadable or older than
    runTime - keep_s: the kept frames always run on without a hole, so a feed
    that was down for longer than the live window starts its history afresh.
    Nothing is kept from a build with another version, model, step or encoding.
    Returns the inputs themselves when there is nothing to keep.
    """
    if not isinstance(prev_manifest, dict):
        return manifest, files
    if any(prev_manifest.get(k) != manifest[k] for k in ("v", "model", "stepMin", "enc")):
        return manifest, files
    earlier = {}
    for f in prev_manifest.get("frames") or []:
        t = from_iso(f.get("t")) if isinstance(f, dict) else None
        if t is not None:
            earlier[int(t)] = f
    live_start = int(from_iso(manifest["windowStart"]))
    oldest = floor_step(from_iso(manifest["runTime"]) - keep_s)
    kept, kept_files = [], {}
    t = live_start - STEP_S
    while t >= oldest:
        f = earlier.get(t)
        path = frame_path(t)
        if f is None or f.get("path") != path:
            break
        entry = {"t": iso(t), "path": path}
        if "swUsedTo" in f:
            if from_iso(f["swUsedTo"]) is None:
                break
            entry["swUsedTo"] = f["swUsedTo"]
        data = read_prev(path)
        if not data:
            break
        kept.append(entry)
        kept_files[path] = data
        t -= STEP_S
    if not kept:
        return manifest, files
    kept.reverse()
    merged = {}
    for k, v in manifest.items():
        merged[k] = v
        if k == "windowStart":
            merged[k] = kept[0]["t"]
            merged["liveStart"] = v
    merged["frames"] = kept + manifest["frames"]
    return merged, {**kept_files, **files}
