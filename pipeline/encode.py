"""AUR1 frame files: one model time, three precipitation types, gzip'd.

Layout (before gzip): b"AUR1", uint16 LE width (360), uint16 LE height (181),
then six uint8 planes of width x height in the order
  diffuse.eflux, diffuse.eavg, mono.eflux, mono.eavg, wave.eflux, wave.eavg,
row 0 = latitude -90, column 0 = longitude -180 (the grid of model.run_frame).

Byte 0 = none. Bytes 1-255 are log-spaced over [lo, hi]:
  value = lo * (hi / lo) ** ((b - 1) / 254)
energy flux over 0.01-100 erg cm^-2 s^-1, average energy over 0.05-50 keV.
Encoding picks the nearest byte in log space; values above hi clamp to 255;
values below lo (including 0) become byte 0.

Average energy 0 means "energy unknown" in model.run_frame (energy flux > 0 but
OvationPyme's number-flux rules gave no energy). It is kept as eavg byte 0 while
the cell's eflux byte encodes normally: eavg is encoded independently of eflux.
"""
import gzip
import struct

import numpy as np

MAGIC = b"AUR1"
W, H = 360, 181
TYPES = ("diffuse", "mono", "wave")
EFLUX_RANGE = (0.01, 100.0)     # erg cm^-2 s^-1
EAVG_RANGE = (0.05, 50.0)       # keV
HEADER_BYTES = 8
FRAME_BYTES = HEADER_BYTES + 2 * len(TYPES) * W * H


def encode_plane(x, lo, hi):
    """float array -> uint8 bytes (0 = none, 1..255 log-spaced over [lo, hi])."""
    x = np.asarray(x, dtype=float)
    if x.shape != (H, W):
        raise ValueError(f"plane shape {x.shape}, want {(H, W)}")
    if not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("plane has NaN, inf or negative values")
    b = np.zeros(x.shape, dtype=np.uint8)
    on = x >= lo
    k = 1 + np.rint(254 * np.log(x[on] / lo) / np.log(hi / lo))
    b[on] = np.clip(k, 1, 255).astype(np.uint8)
    return b


def decode_plane(b, lo, hi):
    """uint8 array -> float32 physical values (byte 0 -> 0)."""
    b = np.asarray(b, dtype=np.uint8)
    out = (lo * (hi / lo) ** ((b.astype(np.float64) - 1) / 254)).astype(np.float32)
    out[b == 0] = 0.0
    return out


def encode_frame(types):
    """{'diffuse'|'mono'|'wave': (eflux, eavg)} -> gzip'd AUR1 bytes."""
    planes = [MAGIC, struct.pack("<HH", W, H)]
    for name in TYPES:
        eflux, eavg = types[name]
        planes.append(encode_plane(eflux, *EFLUX_RANGE).tobytes())
        planes.append(encode_plane(eavg, *EAVG_RANGE).tobytes())
    raw = b"".join(planes)
    assert len(raw) == FRAME_BYTES
    return gzip.compress(raw, compresslevel=9, mtime=0)


def decode_frame(data):
    """gzip'd AUR1 bytes -> {'diffuse'|'mono'|'wave': (eflux, eavg)} float32 (H, W)."""
    raw = gzip.decompress(data)
    if raw[:4] != MAGIC:
        raise ValueError("not an AUR1 frame")
    w, h = struct.unpack("<HH", raw[4:8])
    if (w, h) != (W, H) or len(raw) != HEADER_BYTES + 6 * w * h:
        raise ValueError(f"AUR1 frame {w}x{h}, {len(raw)} bytes")
    planes = np.frombuffer(raw, dtype=np.uint8, offset=HEADER_BYTES).reshape(6, H, W)
    return {name: (decode_plane(planes[2 * i], *EFLUX_RANGE),
                   decode_plane(planes[2 * i + 1], *EAVG_RANGE))
            for i, name in enumerate(TYPES)}
