import gzip
import json
import zlib
from datetime import timedelta

import numpy as np
import pytest

import run_aurora
from pipeline.encode import H, TYPES, W, decode_frame
from tests.feeds import RUN, feeds


def _run(out, rows=None, run_frame=None):
    rows = rows or feeds()
    kw = {"run_frame": run_frame} if run_frame else {}
    return run_aurora.main(["--out", str(out)], fetch=lambda: rows, now=RUN, **kw)


def _nothing_written(tmp_path, out):
    assert not out.exists() or list(out.iterdir()) == []
    assert [p.name for p in tmp_path.iterdir() if p != out] == []   # no stray temp dir


def test_run_writes_site(tmp_path):
    out = tmp_path / "site"
    assert _run(out) == 0                                  # the real model
    m = json.loads((out / "manifest.json").read_text())
    assert len(m["frames"]) == 11
    assert sorted(p.name for p in (out / "frames").iterdir()) == sorted(
        f["path"].split("/")[1] for f in m["frames"])
    lit = 0
    for f in m["frames"]:
        frame = decode_frame((out / f["path"]).read_bytes())
        assert set(frame) == set(TYPES)
        lit += sum(int((frame[t][0] > 0).sum()) for t in TYPES)
    assert lit > 0
    assert [p.name for p in tmp_path.iterdir()] == ["site"]  # temp dir renamed away


def test_gap_writes_nothing(tmp_path):
    out = tmp_path / "site"
    out.mkdir()
    hole = RUN.replace(second=0) - timedelta(hours=2)
    rows = feeds(drop=[(hole, hole + timedelta(minutes=16))])
    assert _run(out, rows) == 2
    _nothing_written(tmp_path, out)


def test_bad_delay_writes_nothing(tmp_path):
    out = tmp_path / "site"
    assert _run(out, feeds(delay_s=4 * 3600)) == 2
    _nothing_written(tmp_path, out)


def test_all_zero_writes_nothing(tmp_path):
    out = tmp_path / "site"
    zeros = lambda dF, when: {t: (np.zeros((H, W)), np.zeros((H, W))) for t in TYPES}
    assert _run(out, run_frame=zeros) == 3
    _nothing_written(tmp_path, out)


def test_failure_midway_writes_nothing(tmp_path):
    out = tmp_path / "site"
    calls = []

    def flaky(dF, when):
        calls.append(when)
        if len(calls) == 3:
            raise RuntimeError("model failed")
        out_ = {t: (np.zeros((H, W)), np.zeros((H, W))) for t in TYPES}
        out_["diffuse"][0][160, 0] = 1.0
        return out_

    with pytest.raises(RuntimeError):
        _run(out, run_frame=flaky)
    _nothing_written(tmp_path, out)


def test_refuses_non_empty_out(tmp_path):
    out = tmp_path / "site"
    out.mkdir()
    (out / "old.txt").write_text("x")
    assert _run(out) == 1
    assert [p.name for p in out.iterdir()] == ["old.txt"]


class _Resp:
    status = 200

    def __init__(self, body):
        self._body = body
        self.headers = {"Content-Encoding": "gzip"}

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fetch_with(monkeypatch, bodies):
    calls = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        return _Resp(bodies[min(len(calls), len(bodies)) - 1])

    monkeypatch.setattr(run_aurora.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(run_aurora.time, "sleep", lambda s: None)
    return calls


GOOD = gzip.compress(b'{"ok": 1}')
TRUNCATED = GOOD[:-12]                      # gzip.decompress -> EOFError
CORRUPT = GOOD[:10] + b"\xff" * 8 + GOOD[18:]  # valid header, bad deflate -> zlib.error


def test_corrupt_bodies_are_what_the_tests_say():
    with pytest.raises(EOFError):
        gzip.decompress(TRUNCATED)
    with pytest.raises(zlib.error):
        gzip.decompress(CORRUPT)


@pytest.mark.parametrize("bad", [TRUNCATED, CORRUPT], ids=["truncated", "corrupt"])
def test_bad_gzip_is_retried(monkeypatch, bad):
    calls = _fetch_with(monkeypatch, [bad, GOOD])
    assert run_aurora._get_json("https://example.invalid/x") == {"ok": 1}
    assert len(calls) == 2


@pytest.mark.parametrize("bad", [TRUNCATED, CORRUPT], ids=["truncated", "corrupt"])
def test_bad_gzip_every_try_is_wrapped(monkeypatch, bad):
    calls = _fetch_with(monkeypatch, [bad])
    with pytest.raises(RuntimeError, match="example.invalid"):
        run_aurora._get_json("https://example.invalid/x", tries=3)
    assert len(calls) == 3
