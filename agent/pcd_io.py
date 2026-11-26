"""
PCD Loader cho Local V3 Agent (ho tro ascii, binary, binary_compressed).
Da kiem chung tren CVAT 3D cvat.edufun.ai: DATA binary, FIELDS x y z intensity (34,720 diem/frame).
"""

import re
import struct
import numpy as np


def _lzf_decompress(src: bytes, out_len: int) -> bytes:
    dst = bytearray(out_len)
    iidx = 0
    oidx = 0
    in_len = len(src)
    while iidx < in_len:
        ctrl = src[iidx]
        iidx += 1
        if ctrl < 32:
            ctrl += 1
            dst[oidx : oidx + ctrl] = src[iidx : iidx + ctrl]
            oidx += ctrl
            iidx += ctrl
        else:
            length = ctrl >> 5
            ref = oidx - ((ctrl & 0x1F) << 8) - 1
            if length == 7:
                length += src[iidx]
                iidx += 1
            ref -= src[iidx]
            iidx += 1
            length += 2
            for _ in range(length):
                dst[oidx] = dst[ref]
                oidx += 1
                ref += 1
    return bytes(dst)


_TYPE_MAP = {
    ("F", 4): "<f4",
    ("F", 8): "<f8",
    ("I", 1): "<i1",
    ("I", 2): "<i2",
    ("I", 4): "<i4",
    ("U", 1): "<u1",
    ("U", 2): "<u2",
    ("U", 4): "<u4",
}


def parse_pcd_bytes(raw: bytes) -> dict:
    """Tra ve dict {'xyz': (N,3) float32, 'intensity': (N,) float32 | None, 'format': str, 'fields': list}."""
    head_probe = raw[:4096].decode("ascii", errors="ignore")
    m = re.search(r"^DATA\s+(\S+)\r?\n", head_probe, flags=re.MULTILINE)
    if not m:
        raise ValueError("Khong tim thay dong DATA trong header PCD.")

    header_end = m.end()
    header_lines = head_probe[:header_end].splitlines()
    data_mode = m.group(1).lower()

    fields, sizes, types, counts = [], [], [], []
    points = 0
    width, height = 0, 1

    for line in header_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0].upper()
        vals = parts[1:]
        if key == "FIELDS":
            fields = [v.lower() for v in vals]
        elif key == "SIZE":
            sizes = [int(v) for v in vals]
        elif key == "TYPE":
            types = [v.upper() for v in vals]
        elif key == "COUNT":
            counts = [int(v) for v in vals]
        elif key == "WIDTH":
            width = int(vals[0])
        elif key == "HEIGHT":
            height = int(vals[0])
        elif key == "POINTS":
            points = int(vals[0])

    if not counts:
        counts = [1] * len(fields)
    if not points:
        points = width * height

    payload = raw[header_end:]

    if data_mode == "ascii":
        arr = np.loadtxt(payload.decode("ascii", errors="ignore").splitlines(), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        xi, yi, zi = fields.index("x"), fields.index("y"), fields.index("z")
        xyz = arr[:, [xi, yi, zi]].astype(np.float32)
        intensity = arr[:, fields.index("intensity")].astype(np.float32) if "intensity" in fields else None
    elif data_mode == "binary":
        dtype_fields = []
        for f, t, s, c in zip(fields, types, sizes, counts):
            dt = _TYPE_MAP.get((t, s), "<f4")
            dtype_fields.append((f, dt) if c == 1 else (f, dt, (c,)))
        rec_dtype = np.dtype(dtype_fields)
        max_pts = min(points, len(payload) // rec_dtype.itemsize)
        structured = np.frombuffer(payload[: max_pts * rec_dtype.itemsize], dtype=rec_dtype)
        xyz = np.column_stack([structured["x"], structured["y"], structured["z"]]).astype(np.float32)
        intensity = structured["intensity"].astype(np.float32) if "intensity" in fields else None
    elif data_mode == "binary_compressed":
        comp_size, uncomp_size = struct.unpack_from("<II", payload, 0)
        uncomp = _lzf_decompress(payload[8 : 8 + comp_size], uncomp_size)
        cursor = 0
        cols = {}
        for f, t, s, c in zip(fields, types, sizes, counts):
            dt = np.dtype(_TYPE_MAP.get((t, s), "<f4"))
            nbytes = s * c * points
            cols[f] = np.frombuffer(uncomp[cursor : cursor + nbytes], dtype=dt)
            cursor += nbytes
        xyz = np.column_stack([cols["x"], cols["y"], cols["z"]]).astype(np.float32)
        intensity = cols["intensity"].astype(np.float32) if "intensity" in cols else None
    else:
        raise ValueError(f"Dinh dang PCD chua ho tro: {data_mode}")

    valid = np.isfinite(xyz).all(axis=1)
    xyz = xyz[valid]
    if intensity is not None:
        intensity = intensity[valid]

    return {
        "xyz": xyz,
        "intensity": intensity,
        "format": data_mode,
        "fields": fields,
        "valid_points": int(xyz.shape[0]),
    }
