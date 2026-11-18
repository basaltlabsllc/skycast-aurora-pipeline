import gzip
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from pipeline.encode import (EAVG_RANGE, EFLUX_RANGE, FRAME_BYTES, H, TYPES, W,
                             decode_frame, encode_frame)

SITE = Path(__file__).parent / "fixtures" / "site"


def _zeros():
    return {t: (np.zeros((H, W)), np.zeros((H, W))) for t in TYPES}


def _planes(data):
    raw = gzip.decompress(data)
    return np.frombuffer(raw, dtype=np.uint8, offset=8).reshape(6, H, W)


def test_roundtrip_within_quantisation():
    rng = np.random.default_rng(1)
    types = {}
    for t in TYPES:
        eflux = np.exp(rng.uniform(np.log(EFLUX_RANGE[0]), np.log(EFLUX_RANGE[1]), (H, W)))
        eavg = np.exp(rng.uniform(np.log(EAVG_RANGE[0]), np.log(EAVG_RANGE[1]), (H, W)))
        eflux[::7] = 0.0
        eavg[::5] = 0.0
        eflux[0, :2] = EFLUX_RANGE                       # both ends of the range
        eavg[1, :2] = EAVG_RANGE
        types[t] = (eflux, eavg)
    back = decode_frame(encode_frame(types))
    for t in TYPES:
        for x, y in zip(types[t], back[t]):
            assert y.dtype == np.float32 and y.shape == (H, W)
            zero = x == 0
            assert np.all(y[zero] == 0.0)                 # 0 stays 0
            rel = np.abs(y[~zero] / x[~zero] - 1)
            assert rel.max() <= 0.025, (t, rel.max())


def test_out_of_range_clamps():
    types = _zeros()
    eflux, eavg = types["diffuse"]
    eflux[90, 0], eflux[90, 1] = 200.0, 0.001
    eavg[90, 0], eavg[90, 1] = 100.0, 0.01
    p = _planes(encode_frame(types))
    assert p[0, 90, 0] == 255 and p[0, 90, 1] == 0
    assert p[1, 90, 0] == 255 and p[1, 90, 1] == 0


def test_header():
    data = encode_frame(_zeros())
    raw = gzip.decompress(data)
    assert raw[:4] == b"AUR1"
    assert struct.unpack("<HH", raw[4:8]) == (360, 181)
    assert len(raw) == FRAME_BYTES == 8 + 6 * 360 * 181


def test_plane_order_and_orientation():
    # Distinct value per plane in the cell at row 3 (lat -87), col 5 (lon -175).
    types = _zeros()
    values = [0.1, 0.2, 1.0, 2.0, 10.0, 20.0]
    for i, t in enumerate(TYPES):
        types[t][0][3, 5] = values[2 * i]
        types[t][1][3, 5] = values[2 * i + 1]
    raw = gzip.decompress(encode_frame(types))
    lo_f, hi_f = EFLUX_RANGE
    lo_a, hi_a = EAVG_RANGE
    for k, v in enumerate(values):
        lo, hi = (lo_f, hi_f) if k % 2 == 0 else (lo_a, hi_a)
        b = raw[8 + k * W * H + 3 * W + 5]
        assert b == 1 + round(254 * np.log(v / lo) / np.log(hi / lo)), k
    assert sum(1 for x in raw[8:] if x) == 6                # nothing else lit


def test_unknown_energy_is_kept():
    # eflux > 0 with eavg == 0 means "energy unknown": eflux encodes normally,
    # eavg stays byte 0.
    types = _zeros()
    types["wave"][0][170, 100] = 1.7
    p = _planes(encode_frame(types))
    assert p[4, 170, 100] > 0 and p[5, 170, 100] == 0
    back = decode_frame(encode_frame(types))["wave"]
    assert back[0][170, 100] == pytest.approx(1.7, rel=0.025)
    assert back[1][170, 100] == 0.0


def test_rejects_bad_input():
    for bad in (np.nan, -1.0, np.inf):
        types = _zeros()
        types["mono"][0][10, 10] = bad
        with pytest.raises(ValueError):
            encode_frame(types)
    types = _zeros()
    types["mono"] = (np.zeros((180, 360)), np.zeros((180, 360)))
    with pytest.raises(ValueError):
        encode_frame(types)


# Recorded from the live run at 2026-11-18T00:55:57Z (runTime): manifest.json
# plus the one frame with the largest diffuse energy flux, decoded. The app's
# decoder test checks the same number.
FIXTURE_FRAME = "frames/20261118T0020Z.aur1"
FIXTURE_MAX_DIFFUSE = 2.3025269508361816       # erg cm^-2 s^-1 (2.303)


def test_recorded_site_frame():
    manifest = json.loads((SITE / "manifest.json").read_text())
    paths = [f["path"] for f in manifest["frames"]]
    assert FIXTURE_FRAME in paths
    assert manifest["enc"] == {"eflux": list(EFLUX_RANGE), "eavg": list(EAVG_RANGE)}
    frame = decode_frame((SITE / FIXTURE_FRAME).read_bytes())
    assert float(frame["diffuse"][0].max()) == pytest.approx(FIXTURE_MAX_DIFFUSE, rel=1e-6)
