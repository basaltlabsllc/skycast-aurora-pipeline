"""Synthetic SWPC feeds in the real feeds' row formats (see tests/fixtures).

Rows are newest first, like the live JSON. Every minute has a row from each
spacecraft in `sources`; only the one `active_at(t)` names is flagged active, and
inactive rows carry deliberately wild values so a test notices if they leak in.
"""
import math
from datetime import datetime, timedelta, timezone

RUN = datetime(2026, 11, 17, 12, 3, 27, tzinfo=timezone.utc)
DELAY_S = 45 * 60


def _tag(dt, z=False):
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + ("Z" if z else "")


def feeds(run=RUN, hours=6.0, newest_age_min=2, delay_s=DELAY_S,
          drop=(), sources=("ACE",), active_at=None):
    """(wind_rows, mag_rows, propagated_rows) ending `newest_age_min` before `run`.

    drop: (start, end) datetimes; minutes in [start, end) have no rows at all.
    active_at: function(datetime) -> source name; default the first source.
    """
    active_at = active_at or (lambda t: sources[0])
    end = run.replace(second=0, microsecond=0) - timedelta(minutes=newest_age_min)
    n = int(hours * 60)
    wind, mag = [], []
    for k in range(n):
        t = end - timedelta(minutes=k)
        if any(a <= t < b for a, b in drop):
            continue
        phase = (t.timestamp() / 3600.0) * 2 * math.pi / 3     # 3 h period
        for src in sources:
            act = src == active_at(t)
            wind.append({"time_tag": _tag(t), "active": act, "source": src,
                         "proton_speed": 450.0 + 50 * math.sin(phase) if act else 2000.0,
                         "proton_density": 5.0})
            mag.append({"time_tag": _tag(t), "active": act, "source": src,
                        "bx_gsm": 1.0,
                        "by_gsm": 2.0 if act else 80.0,
                        "bz_gsm": -6.0 + 3 * math.cos(phase) if act else -90.0})
    head = ["time_tag", "speed", "density", "temperature", "bx", "by", "bz", "bt",
            "vx", "vy", "vz", "propagated_time_tag"]
    prop = [head]                                   # oldest first, like the live file
    for k in reversed(range(60)):
        t = end - timedelta(minutes=k)
        prop.append([_tag(t, True), 450.0, 5.0, 1e5, 1.0, 2.0, -6.0, 6.4,
                     -450.0, 0.0, 0.0, _tag(t + timedelta(seconds=delay_s), True)])
    return wind, mag, prop
