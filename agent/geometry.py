"""
Hinh hoc 3D (Da Guideline / Da Task):
  - Uoc luong mat dat cuc bo (GroundGrid + ring_ground)
  - Chuan hoa truc Barrier (canonical L >= W truoc khi refine/IoU, xuat theo barrier_axis_mode)
  - Ho tro bang kich thuoc dong (size_overrides) va tuy chon tat clamp kich thuoc khi task khong ep bang size
  - Search-Based L-Shape Fitting & snap day box cham dat cuc bo
"""

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Bang kich thuoc mac dinh (dung lam fallback khi Profile khong truyen size_overrides)
CLASS_SIZE = {
    "Car": ((3.8, 5.3), (1.7, 2.1), (1.4, 2.1)),
    "Truck": ((5.2, 12.0), (1.85, 2.6), (1.8, 4.0)),
    "Trailer": ((2.0, 16.0), (1.5, 2.6), (0.8, 4.1)),
    "Bus": ((10.0, 18.5), (2.5, 2.6), (3.0, 3.5)),
    "Construction_vehicle": ((4.0, 12.0), (2.0, 3.2), (2.5, 4.0)),
    "Pedestrian": ((0.3, 0.9), (0.3, 0.9), (0.9, 1.95)),
    "Two_wheels": ((1.6, 2.3), (0.5, 0.95), (1.0, 1.9)),
    "Bicycle": ((1.5, 2.0), (0.45, 0.8), (1.0, 1.9)),
    "Motorcycle": ((1.7, 2.4), (0.6, 1.0), (1.0, 1.85)),
    "Barrier": ((1.0, 3.0), (0.4, 0.7), (0.8, 1.1)),
    "Traffic_cone": ((0.25, 0.65), (0.25, 0.65), (0.45, 1.15)),
}
CLASS_GROUP = {
    "Car": "Vehicle", "Truck": "Vehicle", "Trailer": "Vehicle", "Bus": "Vehicle",
    "Construction_vehicle": "Vehicle", "Pedestrian": "VRU", "Two_wheels": "VRU",
    "Bicycle": "VRU", "Motorcycle": "VRU",
    "Barrier": "Static", "Traffic_cone": "Static",
}
VRU_CLASSES = {"Pedestrian", "Two_wheels", "Bicycle", "Motorcycle"}
GROUP_TOL = {"Vehicle": 0.05, "VRU": 0.10, "Static": 0.10}
GROUP_MAX_DIST = {"Vehicle": 50.0, "VRU": 40.0, "Static": 30.0}


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def canonicalize_barrier(d: Dict) -> Dict:
    out = dict(d)
    if out.get("className") == "Barrier" and out["dx"] < out["dy"]:
        out["dx"], out["dy"] = out["dy"], out["dx"]
        out["rz"] = wrap_pi(out["rz"] + math.pi * 0.5)
    return out


def format_barrier_points(points: List[float], cls: str, barrier_axis_mode: str) -> List[float]:
    pts = list(points)
    if cls == "Barrier":
        dx, dy, rz = pts[6], pts[7], pts[5]
        if barrier_axis_mode == "width_is_x" and dx > dy:
            pts[6], pts[7] = round(dy, 3), round(dx, 3)
            pts[5] = round(wrap_pi(rz + math.pi * 0.5), 4)
        elif barrier_axis_mode != "width_is_x" and dx < dy:
            pts[6], pts[7] = round(dy, 3), round(dx, 3)
            pts[5] = round(wrap_pi(rz + math.pi * 0.5), 4)
    return pts


def class_range(
    cls: str,
    slack: float = 1.0,
    size_overrides: Optional[Dict] = None,
) -> Optional[List[Tuple[float, float]]]:
    raw_rng = None
    if size_overrides and cls in size_overrides:
        cand = size_overrides[cls]
        if isinstance(cand, (list, tuple)) and len(cand) == 3:
            raw_rng = cand
    if raw_rng is None:
        raw_rng = CLASS_SIZE.get(cls)
    if raw_rng is None:
        return None
    grp = CLASS_GROUP.get(cls, "Vehicle")
    tol = GROUP_TOL.get(grp, 0.10) * slack
    return [(float(lo) * (1.0 - tol), float(hi) * (1.0 + tol)) for lo, hi in raw_rng]


def size_fit_score(
    cls: str,
    L: float,
    W: float,
    H: float,
    size_overrides: Optional[Dict] = None,
) -> float:
    if cls == "Barrier" and L < W:
        L, W = W, L
    rng = class_range(cls, size_overrides=size_overrides)
    if not rng:
        return 0.0
    err = 0.0
    for v, (lo, hi) in zip((L, W, H), rng):
        if v < lo:
            err += (lo - v) / max(lo, 1e-6)
        elif v > hi:
            err += (v - hi) / max(hi, 1e-6)
    return err


def classify_by_size(
    L: float,
    W: float,
    H: float,
    exclude: Iterable[str] = ("Trailer", "Construction_vehicle"),
    allowed_classes: Optional[Iterable[str]] = None,
    size_overrides: Optional[Dict] = None,
) -> Tuple[Optional[str], float]:
    excl = set(exclude or ())
    allowed = set(allowed_classes) if allowed_classes is not None else None
    pool = list((size_overrides or {}).keys()) + [c for c in CLASS_SIZE if c not in (size_overrides or {})]
    best, best_err = None, 1e9
    for cls in pool:
        if cls in excl:
            continue
        if allowed is not None and cls not in allowed:
            continue
        if class_range(cls, size_overrides=size_overrides) is None:
            continue
        e = size_fit_score(cls, L, W, H, size_overrides=size_overrides)
        if e < best_err:
            best, best_err = cls, e
    return (best, best_err) if best_err <= 0.32 else (None, best_err)


class GroundGrid:
    """Mat dat cuc bo theo luoi 2m x 2m (loai bo o tren noc xe)."""

    def __init__(self, xyz: np.ndarray, cell: float = 2.0):
        self.cell = cell
        r = np.hypot(xyz[:, 0], xyz[:, 1])
        near = xyz[(r > 3.0) & (r < 22.0)]
        self.global_z = (
            float(np.percentile(near[:, 2], 8))
            if near.shape[0] > 50
            else float(np.percentile(xyz[:, 2], 8))
        )
        ix = np.floor(xyz[:, 0] / cell).astype(np.int64)
        iy = np.floor(xyz[:, 1] / cell).astype(np.int64)
        keys = (ix + 100000) * 200003 + (iy + 100000)
        uniq, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
        zmin = np.full(uniq.shape[0], np.inf)
        np.minimum.at(zmin, inv, xyz[:, 2])
        bad = (cnt < 4) | (zmin > self.global_z + 0.55) | (zmin < self.global_z - 1.2)
        zmin[bad] = np.nan
        self.table = dict(zip(uniq.tolist(), zmin.tolist()))

    def height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ix = np.floor(x / self.cell).astype(np.int64)
        iy = np.floor(y / self.cell).astype(np.int64)
        keys = ((ix + 100000) * 200003 + (iy + 100000)).tolist()
        g = np.array([self.table.get(k, np.nan) for k in keys], dtype=np.float64)
        g[np.isnan(g)] = self.global_z
        return g


def ring_ground(xyz: np.ndarray, cx: float, cy: float, r_box: float, z_hint: float) -> Optional[float]:
    d2 = (xyz[:, 0] - cx) ** 2 + (xyz[:, 1] - cy) ** 2
    ring = (d2 >= (r_box + 0.3) ** 2) & (d2 <= (r_box + 3.0) ** 2)
    z = xyz[ring, 2]
    z = z[np.abs(z - z_hint) <= 1.5]
    if z.size < 10:
        return None
    return float(np.percentile(z, 12))


def to_local(xy: np.ndarray, cx: float, cy: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = xy[:, 0] - cx, xy[:, 1] - cy
    return np.stack([dx * c + dy * s, -dx * s + dy * c], axis=1)


def from_local(lx: float, ly: float, cx: float, cy: float, yaw: float) -> Tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return cx + lx * c - ly * s, cy + lx * s + ly * c


def box_corners(b: Dict) -> np.ndarray:
    out = []
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            px, py = from_local(sx * b["dx"], sy * b["dy"], b["x"], b["y"], b["rz"])
            for sz in (-0.5, 0.5):
                out.append((px, py, b["z"] + sz * b["dz"]))
    return np.array(out, dtype=np.float64)


def bev_polygon(b: Dict) -> np.ndarray:
    return np.array([
        from_local(sx * b["dx"] * 0.5, sy * b["dy"] * 0.5, b["x"], b["y"], b["rz"])
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ])


def _clip(subject, clipper):
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def inter(p1, p2, a, b):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-12:
            return p2
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    out = list(subject)
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        inp, out = out, []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            if inside(e, a, b):
                if not inside(s, a, b):
                    out.append(inter(s, e, a, b))
                out.append(e)
            elif inside(s, a, b):
                out.append(inter(s, e, a, b))
            s = e
    return out


def _area(poly) -> float:
    if len(poly) < 3:
        return 0.0
    p = np.array(poly)
    return float(0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1))))


def bev_overlap_metrics(a: Dict, b: Dict) -> Tuple[float, float]:
    ra = 0.5 * math.hypot(a["dx"], a["dy"])
    rb = 0.5 * math.hypot(b["dx"], b["dy"])
    if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) > ra + rb:
        return 0.0, 0.0
    pa = [tuple(p) for p in bev_polygon(a)]
    pb = [tuple(p) for p in bev_polygon(b)]
    inter = _area(_clip(pa, pb))
    if inter <= 0.0:
        return 0.0, 0.0
    aa = a["dx"] * a["dy"]
    ab = b["dx"] * b["dy"]
    ua = aa + ab - inter
    iou = inter / ua if ua > 0 else 0.0
    iomin = inter / min(aa, ab) if min(aa, ab) > 0 else 0.0
    return iou, iomin


def bev_iou(a: Dict, b: Dict) -> float:
    return bev_overlap_metrics(a, b)[0]


def points_in_box(
    xyz: np.ndarray,
    b: Dict,
    margin_xy: float = 0.0,
    scale: float = 1.0,
    z_lo: Optional[float] = None,
    z_hi: Optional[float] = None,
) -> np.ndarray:
    r = 0.5 * math.hypot(b["dx"], b["dy"]) * scale + margin_xy
    near = ((xyz[:, 0] - b["x"]) ** 2 + (xyz[:, 1] - b["y"]) ** 2) <= r * r
    zl = b["z"] - b["dz"] * 0.5 if z_lo is None else z_lo
    zh = b["z"] + b["dz"] * 0.5 if z_hi is None else z_hi
    near &= (xyz[:, 2] >= zl) & (xyz[:, 2] <= zh)
    idx = np.nonzero(near)[0]
    if idx.size == 0:
        return idx
    loc = to_local(xyz[idx, :2], b["x"], b["y"], b["rz"])
    ok = (np.abs(loc[:, 0]) <= b["dx"] * scale * 0.5 + margin_xy) & (
        np.abs(loc[:, 1]) <= b["dy"] * scale * 0.5 + margin_xy
    )
    return idx[ok]


def l_shape_rect(xy: np.ndarray, n_angles: int = 60) -> Tuple[float, float, float, float, float]:
    if xy.shape[0] < 4:
        mn, mx = xy.min(0), xy.max(0)
        c = (mn + mx) * 0.5
        dx, dy = max(float(mx[0] - mn[0]), 0.25), max(float(mx[1] - mn[1]), 0.25)
        return (float(c[0]), float(c[1]), max(dx, dy), min(dx, dy), 0.0 if dx >= dy else math.pi * 0.5)

    pts = xy if xy.shape[0] <= 400 else xy[:: max(1, xy.shape[0] // 400)]
    thetas = np.linspace(0.0, math.pi * 0.5, n_angles, endpoint=False)
    cos_t = np.cos(thetas)[None, :]
    sin_t = np.sin(thetas)[None, :]
    px = pts[:, 0:1]
    py = pts[:, 1:2]

    rx = px * cos_t + py * sin_t
    ry = -px * sin_t + py * cos_t

    x_lo = np.percentile(rx, 2, axis=0)
    x_hi = np.percentile(rx, 98, axis=0)
    y_lo = np.percentile(ry, 2, axis=0)
    y_hi = np.percentile(ry, 98, axis=0)

    dx_edge = np.where(np.abs(x_lo) < np.abs(x_hi), np.abs(rx - x_lo), np.abs(x_hi - rx))
    dy_edge = np.where(np.abs(y_lo) < np.abs(y_hi), np.abs(ry - y_lo), np.abs(y_hi - ry))
    d_wall = np.minimum(dx_edge, dy_edge)

    closeness = np.mean(1.0 / np.maximum(d_wall, 0.08), axis=0)
    area = np.maximum(x_hi - x_lo, 0.1) * np.maximum(y_hi - y_lo, 0.1)
    score = closeness - 0.04 * area
    best_k = int(np.argmax(score))
    th = float(thetas[best_k])

    loc = to_local(xy, 0.0, 0.0, th)
    mn = np.percentile(loc, 1, axis=0)
    mx = np.percentile(loc, 99, axis=0)
    cx, cy = from_local(float((mn[0] + mx[0]) * 0.5), float((mn[1] + mx[1]) * 0.5), 0.0, 0.0, th)
    w = float(max(mx[0] - mn[0], 0.2))
    h = float(max(mx[1] - mn[1], 0.2))
    yaw = th if w >= h else th + math.pi * 0.5
    return cx, cy, max(w, h), min(w, h), wrap_pi(yaw)


min_area_rect = l_shape_rect


def _axis_center(lo: float, hi: float, D: float, sensor: float) -> float:
    ext = hi - lo
    if ext >= 0.85 * D:
        return (lo + hi) * 0.5
    return lo + D * 0.5 if sensor < (lo + hi) * 0.5 else hi - D * 0.5


def realign(fit: Dict, L: float, W: float, target_yaw: Optional[float] = None) -> Tuple[float, float]:
    eff_L, eff_W = L, W
    if target_yaw is not None:
        diff = abs(wrap_pi(target_yaw - fit["yaw"]))
        if math.radians(45) < diff < math.radians(135):
            eff_L, eff_W = W, L
    lx = _axis_center(fit["lo"][0], fit["hi"][0], eff_L, fit["sensor"][0])
    ly = _axis_center(fit["lo"][1], fit["hi"][1], eff_W, fit["sensor"][1])
    return from_local(lx, ly, fit["cx"], fit["cy"], fit["yaw"])


def refine_box(
    xyz: np.ndarray,
    det: Dict,
    grid: GroundGrid,
    size_overrides: Optional[Dict] = None,
    enable_size_clamp: bool = True,
) -> Dict:
    b = canonicalize_barrier(det)
    cls = b["className"]
    orig_x, orig_y = float(b["x"]), float(b["y"])
    r_box = 0.5 * math.hypot(b["dx"], b["dy"])
    dist_xy = math.hypot(orig_x, orig_y)
    max_g = grid.global_z + max(0.35, 0.05 * dist_xy)
    min_g = grid.global_z - max(0.60, 0.06 * dist_xy)
    g_ring = ring_ground(xyz, orig_x, orig_y, r_box, grid.global_z)
    if g_ring is not None and min_g <= g_ring <= max_g:
        g = g_ring
    else:
        g_grid = float(grid.height(np.array([orig_x]), np.array([orig_y]))[0])
        g = clamp(g_grid, min_g, max_g)

    idx = points_in_box(
        xyz, b, margin_xy=0.15, scale=1.02, z_lo=g + 0.32, z_hi=g + max(b["dz"] * 1.15, b["dz"] + 0.35)
    )
    pts = xyz[idx]
    n = int(pts.shape[0])
    rng = class_range(cls, size_overrides=size_overrides) if enable_size_clamp else None
    b["pointCount"] = n
    b["ground"] = g

    if n >= 35 and cls not in VRU_CLASSES:
        _, _, _, _, ryaw = l_shape_rect(pts[:, :2])
        cands = [wrap_pi(ryaw + k * math.pi * 0.5) for k in range(4)]
        best = min(cands, key=lambda a: abs(wrap_pi(a - b["rz"])))
        if abs(wrap_pi(best - b["rz"])) < math.radians(12):
            b["rz"] = best

    if rng is not None:
        L = clamp(float(b["dx"]), rng[0][0], rng[0][1])
        W = clamp(float(b["dy"]), rng[1][0], rng[1][1])
        H = clamp(float(b["dz"]), rng[2][0], rng[2][1])
    else:
        L, W, H = max(float(b["dx"]), 0.25), max(float(b["dy"]), 0.25), max(float(b["dz"]), 0.35)

    fit = None
    if n >= 15:
        loc = to_local(pts[:, :2], orig_x, orig_y, b["rz"])
        lo = np.percentile(loc, 5, axis=0)
        hi = np.percentile(loc, 95, axis=0)
        sensor = to_local(np.zeros((1, 2)), orig_x, orig_y, b["rz"])[0]
        ext = hi - lo
        if ext[0] >= 0.85 * L:
            L = clamp(float(ext[0]) + 0.08, rng[0][0], rng[0][1]) if rng else float(ext[0]) + 0.08
        if ext[1] >= 0.85 * W:
            W = clamp(float(ext[1]) + 0.08, rng[1][0], rng[1][1]) if rng else float(ext[1]) + 0.08
        fit = {
            "cx": orig_x,
            "cy": orig_y,
            "yaw": float(b["rz"]),
            "lo": lo.tolist(),
            "hi": hi.tolist(),
            "sensor": sensor.tolist(),
        }
        if cls not in VRU_CLASSES:
            nx, ny = realign(fit, L, W)
            max_shift = 0.40 if (n >= 35 and (ext[0] >= 0.45 * L or ext[1] >= 0.45 * W)) else 0.18
            shift = math.hypot(nx - orig_x, ny - orig_y)
            if shift <= max_shift:
                b["x"], b["y"] = nx, ny
            elif shift > 1e-6:
                scale_s = max_shift / shift
                b["x"] = orig_x + (nx - orig_x) * scale_s
                b["y"] = orig_y + (ny - orig_y) * scale_s
        top = float(np.percentile(pts[:, 2], 96))
        h_obs = top - g + 0.05
        if n >= 25:
            if rng is None or (rng[2][0] <= h_obs <= rng[2][1]):
                H = 0.5 * H + 0.5 * clamp(h_obs, 0.35, 4.5)

    b["dx"], b["dy"], b["dz"] = float(L), float(W), float(H)
    b["z"] = g + H * 0.5
    b["fit"] = None if cls in VRU_CLASSES else fit
    return b
