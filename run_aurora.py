"""Fetch SWPC's live solar wind, run OVATION Prime per 10-min frame, write a site.

  python run_aurora.py --out site

Writes <out>/manifest.json and <out>/frames/<YYYYMMDDTHHMMZ>.aur1, prints the
frame count, bytes and seconds. Exit codes, each with nothing written:
  2  solar-wind gap (> 15 min in the hours a frame averages) or a missing /
     implausible L1 -> Earth delay
  3  every frame's energy flux is zero
  1  anything else (a feed fetch failed, <out> is not empty, ...)
"Nothing written" is literal: the site is built in memory, written to a temp dir
beside <out>, and renamed into place only after every frame succeeded.
"""
import argparse
import gzip
import http.client
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import zlib
from datetime import datetime, timezone

from pipeline import site, solarwind

FEEDS = {
    "wind": "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json",
    "mag": "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json",
    "propagated": "https://services.swpc.noaa.gov/products/geospace/propagated-solar-wind-1-hour.json",
}
EXIT_GAP, EXIT_NO_AURORA = 2, 3


def _get_json(url, tries=3):
    # gzip: the 3 MB wind feed arrived truncated (IncompleteRead) on every
    # uncompressed try on 2026-11-17, and whole when compressed.
    req = urllib.request.Request(url, headers={"User-Agent": "skycast-aurora-pipeline",
                                               "Accept-Encoding": "gzip"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 200:
                    raise OSError(f"HTTP {r.status}")
                body = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                return json.loads(body)
        # EOFError: a truncated gzip stream; zlib.error: a corrupt one
        # (both from gzip.decompress) are retried too.
        except (OSError, ValueError, EOFError, zlib.error, http.client.HTTPException) as e:
            if k == tries - 1:
                raise RuntimeError(f"{url}: {e}") from e
            time.sleep(5 * (k + 1))


def fetch_feeds():
    return tuple(_get_json(FEEDS[k]) for k in ("wind", "mag", "propagated"))


def _write_site(out, manifest, files):
    """Write everything into a temp dir beside `out`, then rename it to `out`."""
    out = os.path.abspath(out)
    parent = os.path.dirname(out)
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".aurora-", dir=parent)
    try:
        os.makedirs(os.path.join(tmp, "frames"))
        for path, data in files.items():
            with open(os.path.join(tmp, path), "wb") as f:
                f.write(data)
        with open(os.path.join(tmp, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
            f.write("\n")
        if os.path.isdir(out):
            os.rmdir(out)               # empty (checked before the run)
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def main(argv=None, fetch=fetch_feeds, now=None, run_frame=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out) and (not os.path.isdir(args.out) or os.listdir(args.out)):
        print(f"{args.out} exists and is not an empty directory", file=sys.stderr)
        return 1
    if run_frame is None:
        from pipeline.model import run_frame
    t0 = time.monotonic()
    run_time = int((now or datetime.now(timezone.utc)).timestamp())
    wind, mag, prop = fetch()
    try:
        manifest, files = site.build_site(wind, mag, prop, run_time, run_frame)
    except (solarwind.SolarWindGap, solarwind.BadArrivalDelay) as e:
        print(f"solar-wind gap, nothing written: {e}", file=sys.stderr)
        return EXIT_GAP
    except site.NoAurora as e:
        print(f"nothing written: {e}", file=sys.stderr)
        return EXIT_NO_AURORA
    _write_site(args.out, manifest, files)
    frame_bytes = sum(len(b) for b in files.values())
    total = frame_bytes + os.path.getsize(os.path.join(args.out, "manifest.json"))
    print(f"{len(files)} frames, {frame_bytes} bytes of frames "
          f"({frame_bytes / len(files):.0f} per frame), {total} bytes in all, "
          f"{manifest['windowStart']} .. {manifest['windowEnd']}, "
          f"spacecraft {manifest['spacecraft']}, {time.monotonic() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
