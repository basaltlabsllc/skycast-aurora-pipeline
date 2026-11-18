"""Fetch the published site's manifest and frames, for run_aurora.py --prev.

  python fetch_prev.py --base https://<owner>.github.io/<repo> --out prev

A runner starts with an empty disk; this gives the first build of a run the
frames the previous run published, so the 6 hours of history carry across
runs. Writes <out>/manifest.json and each frame the manifest names whose path
is exactly the pipeline's own name for its time. A frame that cannot be
fetched is skipped (history then starts after it). Exit 1 when the manifest
itself cannot be read; the caller carries on with no history.
"""
import argparse
import http.client
import json
import os
import sys
import time
import urllib.request

from pipeline import site


def _get(url, tries=3):
    req = urllib.request.Request(url, headers={"User-Agent": "skycast-aurora-pipeline"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 200:
                    raise OSError(f"HTTP {r.status}")
                return r.read()
        # A connection cut mid-body raises IncompleteRead, which is no OSError.
        except (OSError, http.client.HTTPException) as e:
            if k == tries - 1:
                raise RuntimeError(f"{url}: {e}") from e
            time.sleep(5 * (k + 1))


def main(argv=None, get=_get):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    base = args.base.rstrip("/")
    try:
        raw = get(f"{base}/manifest.json")
        manifest = json.loads(raw)
        frames = manifest["frames"]
        if not isinstance(frames, list):
            raise ValueError("frames is not a list")
    except (RuntimeError, ValueError, KeyError, TypeError) as e:
        print(f"no published manifest: {e}", file=sys.stderr)
        return 1
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
    with open(os.path.join(args.out, "manifest.json"), "wb") as f:
        f.write(raw)
    got = 0
    for fr in frames:
        t = site.from_iso(fr.get("t")) if isinstance(fr, dict) else None
        # Only the pipeline's own name for that time is ever fetched or written.
        if t is None or fr.get("path") != site.frame_path(t):
            continue
        path = site.frame_path(t)
        try:
            data = get(f"{base}/{path}")
        except RuntimeError as e:
            print(f"skipped {path}: {e}", file=sys.stderr)
            continue
        with open(os.path.join(args.out, path), "wb") as f:
            f.write(data)
        got += 1
    print(f"{got} of {len(frames)} published frames fetched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
