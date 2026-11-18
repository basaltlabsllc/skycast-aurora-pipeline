"""Solar wind at L1 -> Newell coupling, weighted the way OVATION Prime expects.

Inputs are NOAA SWPC's real-time solar wind feeds:
  json/rtsw/rtsw_wind_1m.json, json/rtsw/rtsw_mag_1m.json  (rows from several
  spacecraft interleaved, newest first; `active` marks the one in use)
  products/geospace/propagated-solar-wind-1-hour.json      (first row is a header)
"""
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

# ovation_utilities.calc_avg_solarwind / hourly_solarwind_for_average
PREV_HOUR_WEIGHT = 0.65
N_HOURS_IN_AVERAGE = 4


class SolarWindGap(Exception):
    """The solar wind record has a hole too long to average across."""


class BadArrivalDelay(Exception):
    """The propagated feed's L1 -> Earth delay is missing or outside 10-180 min."""


# Sanity range for the L1 -> Earth delay. At L1 (~1.5e6 km) that spans solar wind
# from ~140 to ~2500 km/s; anything outside is a broken feed, not space weather.
DELAY_MIN_S = 10 * 60
DELAY_MAX_S = 180 * 60


@dataclass
class Series:
    t: np.ndarray        # unix seconds at L1, ascending, whole minutes
    v: np.ndarray        # proton speed, km/s
    bx: np.ndarray       # IMF, nT GSM
    by: np.ndarray
    bz: np.ndarray
    spacecraft: str      # source of the newest sample in the series


def _unix(time_tag):
    s = time_tag[:-1] if time_tag.endswith("Z") else time_tag
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


def _minute(time_tag):
    return int(_unix(time_tag) // 60) * 60


def load_series(wind_rows, mag_rows):
    """Join active, non-null wind and field rows on their whole minute.

    IMAP stamps rows a few seconds past the minute, so rows are matched by the
    minute they fall in. A minute that appears twice keeps its first usable row.
    When SWPC switches the active spacecraft mid-series, each minute comes from
    whichever spacecraft was active then. `spacecraft` is the source of the
    newest sample in the joined series (None when the series is empty).
    """
    speed, source = {}, {}
    for r in wind_rows:
        if not r.get("active") or r.get("proton_speed") is None:
            continue
        m = _minute(r["time_tag"])
        if m not in speed:
            speed[m], source[m] = r["proton_speed"], r["source"]
    field = {}
    for r in mag_rows:
        if not r.get("active") or None in (r.get("bx_gsm"), r.get("by_gsm"), r.get("bz_gsm")):
            continue
        field.setdefault(_minute(r["time_tag"]), (r["bx_gsm"], r["by_gsm"], r["bz_gsm"]))
    t = np.array(sorted(speed.keys() & field.keys()), dtype=float)
    b = np.array([field[int(m)] for m in t], dtype=float).reshape(-1, 3)
    return Series(t=t, v=np.array([speed[int(m)] for m in t], dtype=float),
                  bx=b[:, 0], by=b[:, 1], bz=b[:, 2],
                  spacecraft=source[int(t[-1])] if t.size else None)


def arrival_delay_s(propagated_rows):
    """L1 -> Earth delay from the newest propagated row (row 0 is the header).

    Rows with a null time_tag or propagated_time_tag are ignored. Raises
    BadArrivalDelay when no row is left or the delay is outside 10-180 min.
    """
    head, rows = propagated_rows[0], propagated_rows[1:]
    i_t, i_p = head.index("time_tag"), head.index("propagated_time_tag")
    rows = [r for r in rows if r[i_t] is not None and r[i_p] is not None]
    if not rows:
        raise BadArrivalDelay("no propagated row has both time tags")
    newest = max(rows, key=lambda r: _unix(r[i_t]))
    delay = _unix(newest[i_p]) - _unix(newest[i_t])
    if not DELAY_MIN_S <= delay <= DELAY_MAX_S:
        raise BadArrivalDelay(f"L1 -> Earth delay {delay:.0f} s at {newest[i_t]} "
                              f"is outside {DELAY_MIN_S}-{DELAY_MAX_S} s")
    return delay


def check_gaps(series, t_from, t_to, max_gap_s=900):
    """Raise SolarWindGap if [t_from, t_to] (L1 time) has a hole > max_gap_s,
    counting the stretch from t_from to the first sample and the last to t_to."""
    t = series.t[(series.t >= t_from) & (series.t <= t_to)]
    edges = np.concatenate(([t_from], t, [t_to]))
    gaps = np.diff(edges)
    if gaps.max() > max_gap_s:
        i = int(gaps.argmax())
        raise SolarWindGap(f"{gaps[i]:.0f} s without solar wind after "
                           f"{datetime.fromtimestamp(edges[i], timezone.utc).isoformat()}")


def newell(v, by, bz):
    """Newell coupling dPhi/dt = v^4/3 Bt^2/3 sin^8/3(|theta_c|/2)."""
    bt = np.hypot(by, bz)
    theta = np.arctan2(by, bz)
    return v ** (4 / 3) * bt ** (2 / 3) * np.sin(np.abs(theta) / 2) ** (8 / 3)


def coupling_at(series, t_target, delay_s):
    """OVATION's hourly-weighted Newell coupling at t_target (unix s, at Earth).

    Reproduces ovation_utilities.hourly_solarwind_for_average + calc_avg_solarwind:
    per-minute coupling, mean within each of the 4 hours before the target
    (hour h holds samples with h <= hours_before < h+1), weighted 0.65**h.
    Samples count from their arrival time t + delay_s, so wind that has not
    reached Earth by t_target is never used. An empty hour raises SolarWindGap
    (OvationPyme would silently count it as zero).
    """
    hours_before = (t_target - (series.t + delay_s)) / 3600.0
    ec = newell(series.v, series.by, series.bz)
    total, wsum = 0.0, 0.0
    for h in range(N_HOURS_IN_AVERAGE):
        in_hour = (hours_before >= h) & (hours_before < h + 1)
        if not in_hour.any():
            raise SolarWindGap(f"no solar wind {h}-{h + 1} h before the target")
        w = PREV_HOUR_WEIGHT ** h
        total += w * ec[in_hour].mean()
        wsum += w
    return total / wsum
