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
"""
from datetime import datetime, timezone

import numpy as np

from pipeline import encode, solarwind

STEP_S = 600
WINDOW_BACK_S = 3600
AVERAGE_S = solarwind.N_HOURS_IN_AVERAGE * 3600
OVATIONPYME_COMMIT = "dd774c317d628e5cc9f72d58612282c118681735"
MODEL = f"ovation-prime-2010 (OvationPyme {OVATIONPYME_COMMIT})"


class NoAurora(Exception):
    """Every frame's energy flux is zero in all three types: nothing to publish."""


def floor_step(ts):
    return int(ts // STEP_S) * STEP_S


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
