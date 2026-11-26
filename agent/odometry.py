"""
LiDAR Odometry cho khoang cach 0.5s tren cloud 32 tia:
  - Buoc 1 (Global Init): 2D BEV FFT Phase-Correlation quet goc quay [-24..+24 do] tim cuc dai toan cuc (dx, dy, dyaw),
    khac phuc triet de hien tuong truot cuc bo (local minimum) khi xe ego nhay 4-10m trong 0.5s!
  - Buoc 2 (Fine Refine): Voxel-Centroid Trimmed SE(2)+Z ICP tinh chinh den do chinh xac cm & < 0.3 do.
  - Buoc 3 (Self-Verification): Kiem tra fitness, rmse va gioi han vat ly; tu dong tra valid=False neu chuyen canh.
"""

import math
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree


def _voxel_centroid_downsample(pts: np.ndarray, voxel_size: float = 0.40, max_pts: int = 2200) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts
    ijk = np.floor(pts / voxel_size).astype(np.int64)
    keys = (ijk[:, 0] + 50000) * 100003 * 100003 + (ijk[:, 1] + 50000) * 100003 + (ijk[:, 2] + 50000)
    uniq, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    sums = np.zeros((uniq.shape[0], 3), dtype=np.float64)
    np.add.at(sums, inv, pts.astype(np.float64))
    centroids = (sums / cnt[:, None]).astype(np.float32)
    if centroids.shape[0] > max_pts:
        step = max(1, centroids.shape[0] // max_pts)
        centroids = centroids[::step][:max_pts]
    return centroids


def extract_static_structure(xyz: np.ndarray, ground_z: float) -> np.ndarray:
    """Lay cac diem cau truc tinh (toa nha, le duong, cot, cay, hang rao) tren mat dat."""
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    hag = xyz[:, 2] - ground_z
    mask = (r >= 3.0) & (r <= 48.0) & (hag >= 0.35) & (hag <= 6.5)
    return _voxel_centroid_downsample(xyz[mask], voxel_size=0.40, max_pts=2200)


def _bev_grid(xy: np.ndarray, res: float = 0.5, size: int = 192) -> np.ndarray:
    half = size * res * 0.5
    gx = np.floor((xy[:, 0] + half) / res).astype(np.int32)
    gy = np.floor((xy[:, 1] + half) / res).astype(np.int32)
    ok = (gx >= 0) & (gx < size) & (gy >= 0) & (gy < size)
    grid = np.zeros((size, size), dtype=np.float32)
    grid[gy[ok], gx[ok]] = 1.0
    # Lam muot 3x3 nhe de chiu duoc cloud 32 tia thua
    p = np.pad(grid, 1)
    smooth = (
        p[1:-1, 1:-1] * 0.4
        + (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) * 0.15
    )
    return smooth - smooth.mean()


def _coarse_bev_fft_init(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    """Tim (dx, dy, dyaw) khoi tao toan cuc bang FFT cross-correlation tren luoi BEV 0.5m."""
    res = 0.5
    size = 192
    max_shift_cells = int(14.5 / res)
    dst_f = np.conj(np.fft.rfft2(_bev_grid(dst_xy, res, size)))

    best_score = -1e18
    best_T = np.eye(4, dtype=np.float64)

    for deg in np.linspace(-24.0, 24.0, 25):
        rad = math.radians(float(deg))
        c, s = math.cos(rad), math.sin(rad)
        rot_xy = np.column_stack([src_xy[:, 0] * c - src_xy[:, 1] * s, src_xy[:, 0] * s + src_xy[:, 1] * c])
        corr = np.fft.irfft2(np.fft.rfft2(_bev_grid(rot_xy, res, size)) * dst_f, s=(size, size))
        # Chi xet vung dich chuyen vat ly <= 14.5m quanh goc
        shifted = np.fft.fftshift(corr)
        mid = size // 2
        sub = shifted[mid - max_shift_cells : mid + max_shift_cells + 1, mid - max_shift_cells : mid + max_shift_cells + 1]
        idx_flat = int(np.argmax(sub))
        val = float(sub.flat[idx_flat])
        if val > best_score:
            best_score = val
            sy_i, sx_i = divmod(idx_flat, sub.shape[1])
            # corr = ifft(F(src_rot) * conj(F(dst))) dat dinh tai shift (src_rot - dst), nen t = -(shift)
            dx = -(sx_i - max_shift_cells) * res
            dy = -(sy_i - max_shift_cells) * res
            T = np.eye(4, dtype=np.float64)
            T[:2, :2] = [[c, -s], [s, c]]
            T[0, 3], T[1, 3] = dx, dy
            best_T = T
    return best_T


def _solve_se2_z(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    sxy = src[:, :2] - src_c[:2]
    dxy = dst[:, :2] - dst_c[:2]
    H = sxy.T @ dxy
    U, _, Vt = np.linalg.svd(H)
    R2 = Vt.T @ U.T
    if np.linalg.det(R2) < 0:
        Vt[-1, :] *= -1
        R2 = Vt.T @ U.T
    T = np.eye(4, dtype=np.float64)
    T[:2, :2] = R2
    T[:2, 3] = dst_c[:2] - R2 @ src_c[:2]
    T[2, 3] = float(np.median(dst[:, 2] - src[:, 2]))
    return T


def register_frames_icp(
    prev_struct: Optional[np.ndarray],
    cur_struct: Optional[np.ndarray],
    init_T: Optional[np.ndarray] = None,
) -> Dict:
    invalid = {
        "valid": False, "T": np.eye(4).tolist(),
        "dx": 0.0, "dy": 0.0, "dyaw_deg": 0.0, "fitness": 0.0, "rmse": 99.0,
    }
    if prev_struct is None or cur_struct is None or len(prev_struct) < 60 or len(cur_struct) < 60:
        return invalid

    src_orig = prev_struct.astype(np.float64)
    T = (
        np.array(init_T, dtype=np.float64)
        if init_T is not None
        else _coarse_bev_fft_init(prev_struct[:, :2], cur_struct[:, :2])
    )
    tree = cKDTree(cur_struct)

    for max_dist, trim_ratio, n_iter in [(2.2, 0.78, 10), (1.0, 0.82, 10)]:
        for _ in range(n_iter):
            cur_pts = (T[:3, :3] @ src_orig.T).T + T[:3, 3]
            dists, idx = tree.query(cur_pts, distance_upper_bound=max_dist)
            valid_m = np.isfinite(dists)
            if valid_m.sum() < 35:
                break
            v_src = cur_pts[valid_m]
            v_dst = cur_struct[idx[valid_m]].astype(np.float64)
            v_d = dists[valid_m]
            if trim_ratio < 1.0 and len(v_d) > 40:
                thr = np.quantile(v_d, trim_ratio)
                keep = v_d <= max(thr, 0.12)
                v_src, v_dst = v_src[keep], v_dst[keep]
            dT = _solve_se2_z(v_src, v_dst)
            T = dT @ T
            if np.linalg.norm(dT[:3, 3]) < 5e-4 and abs(math.atan2(dT[1, 0], dT[0, 0])) < 1e-4:
                break

    final_pts = (T[:3, :3] @ src_orig.T).T + T[:3, 3]
    dists, _ = tree.query(final_pts, distance_upper_bound=0.95)
    inliers = np.isfinite(dists)
    fitness = float(inliers.mean())
    rmse = float(np.sqrt(np.mean(dists[inliers] ** 2))) if inliers.any() else 99.0

    dx, dy, dz = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
    dyaw = math.atan2(float(T[1, 0]), float(T[0, 0]))
    trans_norm = math.hypot(dx, dy)

    ok = (
        fitness >= 0.40
        and rmse <= 0.55
        and trans_norm <= 14.5
        and abs(math.degrees(dyaw)) <= 38.0
        and abs(dz) <= 0.70
    )
    if not ok:
        return {**invalid, "fitness": round(fitness, 3), "rmse": round(rmse, 3)}

    return {
        "valid": True,
        "T": T.tolist(),
        "dx": round(dx, 3),
        "dy": round(dy, 3),
        "dyaw_deg": round(math.degrees(dyaw), 2),
        "fitness": round(fitness, 3),
        "rmse": round(rmse, 3),
    }
