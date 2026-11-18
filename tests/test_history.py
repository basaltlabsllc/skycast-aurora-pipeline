"""Six hours of history: a build keeps the frames earlier builds published."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np

import fetch_prev
import run_aurora
from pipeline import site
from pipeline.encode import H, TYPES, W
from tests.feeds import RUN, feeds


def _frame(value):
    def run_frame(dF, when):
        out = {t: (np.zeros((H, W)), np.zeros((H, W))) for t in TYPES}
        out["diffuse"][0][160, 0] = value
        return out
    return run_frame


def _build(run, value=1.0):
    return site.build_site(*feeds(run=run), run.timestamp(), _frame(value), log=lambda s: None)


def _chain(n, step_min=10, first=RUN):
    """n builds step_min apart, each merged onto the one before."""
    m = files = None
    for k in range(n):
        run = first + timedelta(minutes=k * step_min)
        m2, f2 = _build(run, value=1.0 + k)
        if m is not None:
            m2, f2 = site.merge_history(m2, f2, m, files.get)
        m, files = m2, f2
    return m, files


def _ts(m):
    return [site.from_iso(f["t"]) for f in m["frames"]]


def test_first_build_has_no_history():
    m, files = _build(RUN)
    assert site.merge_history(m, files, None, lambda p: None) == (m, files)
    assert "liveStart" not in m


def test_a_later_build_keeps_the_frames_that_left_its_window():
    old_m, old_files = _build(RUN, value=1.0)
    new_m, new_files = _build(RUN + timedelta(minutes=30), value=2.0)
    m, files = site.merge_history(new_m, new_files, old_m, old_files.get)
    # RUN is 12:03, so the old window starts 11:00 and the new one 11:30.
    assert m["windowStart"] == "2026-11-17T11:00:00Z"
    assert m["liveStart"] == new_m["windowStart"] == "2026-11-17T11:30:00Z"
    assert m["windowEnd"] == new_m["windowEnd"] and m["runTime"] == new_m["runTime"]
    assert np.all(np.diff(_ts(m)) == 600)
    assert sorted(files) == sorted(f["path"] for f in m["frames"])
    for f in m["frames"]:
        kept = site.from_iso(f["t"]) < site.from_iso(m["liveStart"])
        # kept frames are the old build's bytes; the rest are this build's
        assert files[f["path"]] == (old_files if kept else new_files)[f["path"]]
    assert [f for f in m["frames"] if f["t"] >= m["liveStart"]] == new_m["frames"]
    assert m["frames"][0] == old_m["frames"][0]            # swUsedTo travels with it


def test_history_stops_six_hours_before_the_run():
    m, files = _chain(60)                                   # 9 h 50 min of builds
    last = RUN + timedelta(minutes=590)
    assert m["windowStart"] == site.iso(site.floor_step(last.timestamp() - 6 * 3600))
    assert np.all(np.diff(_ts(m)) == 600)
    assert len(files) == len(m["frames"])


def test_history_restarts_after_an_outage_longer_than_the_last_window():
    old_m, old_files = _chain(6)
    # The old build's window runs ahead to the solar wind's arrival, so a short
    # outage leaves no hole. This one starts 20 min past the old window's end.
    late = datetime.fromtimestamp(site.from_iso(old_m["windowEnd"]), timezone.utc) + timedelta(minutes=83)
    new_m, new_files = _build(late)
    assert site.from_iso(new_m["windowStart"]) - site.from_iso(old_m["windowEnd"]) >= 1200
    m, files = site.merge_history(new_m, new_files, old_m, old_files.get)
    assert (m, files) == (new_m, new_files)


def test_a_short_outage_leaves_no_hole():
    old_m, old_files = _chain(6)
    late = RUN + timedelta(minutes=50 + 80)                 # one hour never built
    new_m, new_files = _build(late)
    m, files = site.merge_history(new_m, new_files, old_m, old_files.get)
    assert m["windowStart"] == old_m["windowStart"]
    assert np.all(np.diff(_ts(m)) == 600)


def test_a_missing_frame_file_ends_the_history_there():
    old_m, old_files = _chain(4)
    new_m, new_files = _build(RUN + timedelta(minutes=60))
    hole = "2026-11-17T11:20:00Z"
    gone = site.frame_path(site.from_iso(hole))
    m, files = site.merge_history(new_m, new_files, old_m,
                                  lambda p: None if p == gone else old_files.get(p))
    assert m["windowStart"] == "2026-11-17T11:30:00Z"
    assert gone not in files


def test_nothing_is_kept_from_another_encoding_or_model():
    old_m, old_files = _build(RUN)
    new_m, new_files = _build(RUN + timedelta(minutes=30))
    for key, value in (("enc", {"eflux": [1, 2], "eavg": [1, 2]}), ("model", "other"),
                       ("stepMin", 5), ("v", 2)):
        other = {**old_m, key: value}
        assert site.merge_history(new_m, new_files, other, old_files.get) == (new_m, new_files)


def test_a_frame_named_by_another_path_is_not_kept():
    old_m, old_files = _build(RUN)
    new_m, new_files = _build(RUN + timedelta(minutes=30))
    bad = json.loads(json.dumps(old_m))
    bad["frames"][2]["path"] = "../secrets"                 # 11:20, the newest kept one
    m, files = site.merge_history(new_m, new_files, bad, old_files.get)
    assert (m, files) == (new_m, new_files)


def test_garbage_for_the_earlier_manifest_keeps_nothing():
    new_m, new_files = _build(RUN)
    for junk in ([], "x", 3, {"frames": "no"}, {**new_m, "frames": [None, 4, {"t": "nope"}]}):
        assert site.merge_history(new_m, new_files, junk, lambda p: b"x") == (new_m, new_files)


def _run(out, run, prev=None, value=1.0):
    argv = ["--out", str(out)] + (["--prev", str(prev)] if prev else [])
    return run_aurora.main(argv, fetch=lambda: feeds(run=run), now=run, run_frame=_frame(value))


def test_run_with_prev_writes_the_kept_frames(tmp_path):
    assert _run(tmp_path / "a", RUN) == 0
    assert _run(tmp_path / "b", RUN + timedelta(minutes=30), prev=tmp_path / "a", value=2.0) == 0
    m = json.loads((tmp_path / "b" / "manifest.json").read_text())
    assert m["windowStart"] == "2026-11-17T11:00:00Z" and m["liveStart"] == "2026-11-17T11:30:00Z"
    assert sorted(p.name for p in (tmp_path / "b" / "frames").iterdir()) == sorted(
        f["path"].split("/")[1] for f in m["frames"])
    first = m["frames"][0]["path"]
    assert (tmp_path / "b" / first).read_bytes() == (tmp_path / "a" / first).read_bytes()


def test_run_with_a_missing_or_broken_prev_still_builds(tmp_path):
    assert _run(tmp_path / "a", RUN, prev=tmp_path / "nowhere") == 0
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("{not json")
    assert _run(tmp_path / "b", RUN, prev=broken) == 0
    assert "liveStart" not in json.loads((tmp_path / "b" / "manifest.json").read_text())


def test_run_does_not_keep_a_frame_file_that_is_not_gzip(tmp_path):
    assert _run(tmp_path / "a", RUN) == 0
    (tmp_path / "a" / "frames" / "20261117T1120Z.aur1").write_bytes(b"<html>404</html>")
    assert _run(tmp_path / "b", RUN + timedelta(minutes=30), prev=tmp_path / "a") == 0
    m = json.loads((tmp_path / "b" / "manifest.json").read_text())
    assert "liveStart" not in m and m["windowStart"] == "2026-11-17T11:30:00Z"


def test_fetch_prev_copies_the_published_site(tmp_path):
    assert _run(tmp_path / "a", RUN) == 0
    m = json.loads((tmp_path / "a" / "manifest.json").read_text())
    m["frames"].append({"t": "2026-11-17T13:00:00Z", "path": "../../etc/passwd"})
    asked = []

    def get(url):
        asked.append(url)
        rel = url.split("https://x.test/site/")[1]
        if rel == "manifest.json":
            return json.dumps(m).encode()
        if rel.endswith("1110Z.aur1"):
            raise RuntimeError("HTTP 404")
        return (tmp_path / "a" / rel).read_bytes()

    out = tmp_path / "prev"
    assert fetch_prev.main(["--base", "https://x.test/site/", "--out", str(out)], get=get) == 0
    assert not any("passwd" in u for u in asked)
    names = sorted(p.name for p in (out / "frames").iterdir())
    assert len(names) == 10 and "20261117T1110Z.aur1" not in names
    # and a build can keep frames from what was fetched
    assert _run(tmp_path / "b", RUN + timedelta(minutes=30), prev=out) == 0
    b = json.loads((tmp_path / "b" / "manifest.json").read_text())
    assert b["windowStart"] == "2026-11-17T11:20:00Z"       # stops at the 11:10 hole


def test_fetch_prev_without_a_manifest_exits_1(tmp_path):
    def get(url):
        raise RuntimeError("HTTP 404")
    assert fetch_prev.main(["--base", "https://x.test", "--out", str(tmp_path / "p")], get=get) == 1
    assert not (tmp_path / "p").exists()


def test_a_download_cut_short_is_retried_then_reported(monkeypatch):
    import http.client
    calls = []

    def cut(req, timeout):
        calls.append(req.full_url)
        raise http.client.IncompleteRead(b"abc", 9)
    monkeypatch.setattr(fetch_prev.urllib.request, "urlopen", cut)
    monkeypatch.setattr(fetch_prev.time, "sleep", lambda s: None)
    try:
        fetch_prev._get("https://x.test/f")
        raise AssertionError("no error")
    except RuntimeError as e:
        assert "https://x.test/f" in str(e)
    assert len(calls) == 3
