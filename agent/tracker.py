"""
Tracker da nguon theo chuoi (Sequence-aware Ego-Compensated Tracker — Da Guideline):
  - Uu tien bu chuyen dong ego bang SE(3) LiDAR Odometry (ICP), fallback 2D SE(2) RANSAC.
  - Tham so dong theo Profile: frame_dt, max_frame_gap, occluded_min_points, lock_track_size, barrier_axis_mode.
  - Ket thuc track an toan (outside=True) khi vat bien mat hoac het sequence.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from geometry import CLASS_GROUP, VRU_CLASSES, format_barrier_points, realign, wrap_pi

GATE_BASE = 2.8


def _ransac_se2_from_dets(prev: List[Dict], cur: List[Dict]) -> np.ndarray:
    cands = []
    for a in prev:
        for b in cur:
            if CLASS_GROUP.get(a["className"], "Vehicle") == CLASS_GROUP.get(b["className"], "Vehicle"):
                d = np.array([b["x"] - a["x"], b["y"] - a["y"]], dtype=np.float64)
                if np.linalg.norm(d) < 10.0:
                    cands.append(d)
    T = np.eye(4, dtype=np.float64)
    if not cands:
        return T
    arr = np.array(cands)
    best_n, best = 0, np.zeros(2)
    for c in arr:
        inl = np.linalg.norm(arr - c, axis=1) < 0.85
        if int(inl.sum()) > best_n:
            best_n = int(inl.sum())
            best = arr[inl].mean(axis=0)
    if best_n >= 2:
        T[0, 3], T[1, 3] = best[0], best[1]
    return T


def _transform_xy_yaw(x: float, y: float, yaw: float, T: np.ndarray) -> Tuple[float, float, float]:
    p = T[:2, :2] @ np.array([x, y], dtype=np.float64) + T[:2, 3]
    dyaw = math.atan2(float(T[1, 0]), float(T[0, 0]))
    return float(p[0]), float(p[1]), wrap_pi(yaw + dyaw)


def build_tracks(
    frames: Dict[int, Dict],
    barrier_axis_mode: str = "length_is_x",
    frame_dt: float = 0.5,
    max_frame_gap: int = 4,
    occluded_min_points: int = 5,
    lock_track_size: bool = True,
) -> List[Dict]:
    all_frames = sorted(frames.keys())
    next_global_frame: Dict[int, Optional[int]] = {}
    for i, f in enumerate(all_frames):
        next_global_frame[f] = all_frames[i + 1] if i + 1 < len(all_frames) else None

    gap_limit = max(1, int(max_frame_gap))
    by_seq: Dict[str, List[int]] = defaultdict(list)
    cur_seq_id = 0
    prev_f = None
    prev_raw_seq = None
    for f in all_frames:
        raw_seq = frames[f]["seq"]
        if prev_f is None or raw_seq != prev_raw_seq or (f - prev_f) > 1:
            cur_seq_id += 1
        by_seq[f"{raw_seq}#{cur_seq_id}"].append(f)
        prev_f = f
        prev_raw_seq = raw_seq

    raw_tracks: List[Dict] = []
    for seq_key, flist in by_seq.items():
        T_w_from_f: Dict[int, np.ndarray] = {flist[0]: np.eye(4, dtype=np.float64)}
        for idx in range(1, len(flist)):
            f_prev, f_cur = flist[idx - 1], flist[idx]
            odom = frames[f_cur].get("odom_from_prev")
            if odom and odom.get("valid"):
                T_cur_from_prev = np.asarray(odom["T"], dtype=np.float64)
            else:
                T_cur_from_prev = _ransac_se2_from_dets(frames[f_prev]["dets"], frames[f_cur]["dets"])
            T_w_from_f[f_cur] = T_w_from_f[f_prev] @ np.linalg.inv(T_cur_from_prev)

        active: List[Dict] = []
        for f in flist:
            dets = frames[f]["dets"]
            T_wf = T_w_from_f[f]
            dets_w = [_transform_xy_yaw(d["x"], d["y"], d["rz"], T_wf) for d in dets]

            live = [t for t in active if 1 <= (f - t["last_frame"]) <= gap_limit]
            cost = np.full((len(live), len(dets)), 1e6, dtype=np.float64)

            for i, t in enumerate(live):
                gap = f - t["last_frame"]
                vx, vy = t["vel_w"]
                px = t["last_w"][0] + vx * gap
                py = t["last_w"][1] + vy * gap
                last_grp = CLASS_GROUP.get(t["last"]["className"], "Vehicle")
                for j, d in enumerate(dets):
                    wx, wy, _ = dets_w[j]
                    dist = math.hypot(wx - px, wy - py)
                    same_grp = CLASS_GROUP.get(d["className"], "Vehicle") == last_grp
                    gate = GATE_BASE + 0.45 * (gap - 1) + (0.0 if same_grp else 0.8)
                    if dist < gate:
                        cls_pen = 0.0 if d["className"] == t["last"]["className"] else (0.5 if same_grp else 1.8)
                        cost[i, j] = dist + cls_pen

            assigned = set()
            if live and dets:
                ri, ci = linear_sum_assignment(cost)
                for i, j in zip(ri, ci):
                    if cost[i, j] >= 1e5:
                        continue
                    t = live[i]
                    d = dets[j]
                    wx, wy, wyaw = dets_w[j]
                    gap = f - t["last_frame"]
                    inst_vx = (wx - t["last_w"][0]) / gap
                    inst_vy = (wy - t["last_w"][1]) / gap
                    if t["vel_count"] == 0:
                        t["vel_w"] = (inst_vx, inst_vy)
                    else:
                        t["vel_w"] = (0.6 * inst_vx + 0.4 * t["vel_w"][0], 0.6 * inst_vy + 0.4 * t["vel_w"][1])
                    t["vel_count"] += 1
                    t["obs"].append((f, d, (wx, wy, wyaw)))
                    t["last"] = d
                    t["last_w"] = (wx, wy, wyaw)
                    t["last_frame"] = f
                    assigned.add(j)

            for j, d in enumerate(dets):
                if j not in assigned:
                    wx, wy, wyaw = dets_w[j]
                    active.append({
                        "seq": frames[f]["seq"],
                        "obs": [(f, d, (wx, wy, wyaw))],
                        "last": d,
                        "last_w": (wx, wy, wyaw),
                        "last_frame": f,
                        "vel_w": (0.0, 0.0),
                        "vel_count": 0,
                    })

        seq_end = flist[-1]
        after_seq_frame = next_global_frame.get(seq_end)
        for t in active:
            t["frames_in_seq"] = flist
            t["after_seq_frame"] = after_seq_frame
            t["poses_w"] = T_w_from_f
        raw_tracks.extend(active)

    single_frame_mode = len(all_frames) == 1
    out = []
    for t in raw_tracks:
        res = _finalize(
            t,
            barrier_axis_mode=barrier_axis_mode,
            single_frame_mode=single_frame_mode,
            frame_dt=frame_dt,
            occluded_min_points=occluded_min_points,
            lock_track_size=lock_track_size,
        )
        if res:
            out.append(res)
    for k, t in enumerate(out, 1):
        t["track_id"] = k
    return out


def _finalize(
    t: Dict,
    barrier_axis_mode: str = "length_is_x",
    single_frame_mode: bool = False,
    frame_dt: float = 0.5,
    occluded_min_points: int = 5,
    lock_track_size: bool = True,
) -> Optional[Dict]:
    obs = t["obs"]
    n = len(obs)
    srcs = set(s for _, d, _ in obs for s in d["sources"])
    mean_score = float(np.mean([d["score"] for _, d, _ in obs]))
    only_geo = srcs == {"geo"}

    if not single_frame_mode:
        if n == 1 and (only_geo or (len(obs[0][1]["sources"]) < 2 and mean_score < 0.45)):
            return None
        if only_geo and n < 2:
            return None

    votes: Dict[str, float] = defaultdict(float)
    for _, d, _ in obs:
        mult = 1.0
        if "cp" in d["sources"] and "pp" in d["sources"]:
            mult += 0.6
        if "cam" in d["sources"]:
            mult += 1.0
        votes[d["className"]] += max(d["score"], 0.25) * mult

    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    cls = ranked[0][0]
    margin = ranked[0][1] / max(sum(votes.values()), 1e-6)

    ref_idx = max(
        range(n),
        key=lambda k: (obs[k][1]["pointCount"], -obs[k][1].get("distance", 50.0)),
    )
    ref_f, ref_d, ref_w = obs[ref_idx]
    dense = sorted(
        [d for _, d, _ in obs if d["pointCount"] >= max(12, int(0.5 * ref_d["pointCount"]))],
        key=lambda x: -x["pointCount"],
    )[:3]
    if len(dense) >= 2:
        ref_L = float(np.median([x["dx"] for x in dense]))
        ref_W = float(np.median([x["dy"] for x in dense]))
        ref_H = float(np.median([x["dz"] for x in dense]))
    else:
        ref_L, ref_W, ref_H = ref_d["dx"], ref_d["dy"], ref_d["dz"]

    ref_wyaw = ref_w[2]
    world_disp = np.array([obs[-1][2][0] - obs[0][2][0], obs[-1][2][1] - obs[0][2][1]], dtype=np.float64)
    disp_norm = float(np.linalg.norm(world_disp))
    if n >= 2 and disp_norm > 1.8 and cls not in ("Barrier", "Traffic_cone"):
        move_wyaw = math.atan2(float(world_disp[1]), float(world_disp[0]))
        if math.cos(ref_wyaw - move_wyaw) < -0.15:
            ref_wyaw = wrap_pi(ref_wyaw + math.pi)

    fixed_wyaws = [0.0] * n
    fixed_wyaws[ref_idx] = ref_wyaw
    for k in range(ref_idx + 1, n):
        wy = obs[k][2][2]
        if math.cos(wy - fixed_wyaws[k - 1]) < 0.0:
            wy = wrap_pi(wy + math.pi)
        fixed_wyaws[k] = wy
    for k in range(ref_idx - 1, -1, -1):
        wy = obs[k][2][2]
        if math.cos(wy - fixed_wyaws[k + 1]) < 0.0:
            wy = wrap_pi(wy + math.pi)
        fixed_wyaws[k] = wy

    poses_w = t["poses_w"]
    keyframes = []
    occ_thr = max(0, int(occluded_min_points))
    for k, (f, d, _) in enumerate(obs):
        T_wf = poses_w[f]
        ego_yaw = math.atan2(float(T_wf[1, 0]), float(T_wf[0, 0]))
        local_yaw = wrap_pi(fixed_wyaws[k] - ego_yaw)

        L = ref_L if lock_track_size else float(d["dx"])
        W = ref_W if lock_track_size else float(d["dy"])
        H = ref_H if lock_track_size else float(d["dz"])

        x, y = d["x"], d["y"]
        if (
            lock_track_size
            and cls not in VRU_CLASSES
            and d.get("fit")
            and (abs(L - d["dx"]) > 0.05 or abs(W - d["dy"]) > 0.05)
        ):
            rx, ry = realign(d["fit"], L, W, target_yaw=local_yaw)
            sh = math.hypot(rx - x, ry - y)
            if sh <= 0.35:
                x, y = rx, ry
            elif sh > 1e-6:
                x = x + (rx - x) * (0.35 / sh)
                y = y + (ry - y) * (0.35 / sh)
        z = d["ground"] + H * 0.5 if d.get("ground") is not None else d["z"]

        raw_pts = [
            round(x, 3), round(y, 3), round(z, 3),
            0.0, 0.0, round(local_yaw, 4),
            round(L, 3), round(W, 3), round(H, 3),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]
        pts16 = format_barrier_points(raw_pts, cls, barrier_axis_mode)
        occluded = bool(occ_thr > 0 and d["pointCount"] < occ_thr and n >= 2)
        keyframes.append({
            "frame": f,
            "points": pts16,
            "occluded": occluded,
            "pointCount": d["pointCount"],
            "sources": d["sources"],
            "camera": d.get("camera"),
        })

    seq_frames = t["frames_in_seq"]
    after_seq_frame = t.get("after_seq_frame")
    fs = [o[0] for o in obs]
    outside_frames: List[int] = []
    if fs[-1] < seq_frames[-1]:
        idx_last = seq_frames.index(fs[-1])
        outside_frames.append(seq_frames[idx_last + 1])
    elif after_seq_frame is not None:
        outside_frames.append(after_seq_frame)

    tags = set()
    for _, d, _ in obs:
        tags.update(d["tags"])
    if not only_geo:
        tags.discard("MISSED?")
    if margin < 0.68:
        tags.add("UNCERTAIN_CLASS")
    if any(b - a > 1 for a, b in zip(fs, fs[1:])):
        tags.add("GAP_INTERPOLATED")

    avg_src = float(np.mean([len(d["sources"]) for _, d, _ in obs]))
    level = (
        "HIGH"
        if (avg_src >= 2.0 and "UNCERTAIN_CLASS" not in tags and "DETECTOR_DISAGREE" not in tags and not only_geo)
        else "REVIEW"
    )
    dt = max(0.02, float(frame_dt))
    return {
        "seq": t["seq"],
        "className": cls,
        "votes": {k: round(v, 2) for k, v in ranked},
        "level": level,
        "tags": sorted(tags),
        "sources": sorted(srcs),
        "score": round(mean_score, 3),
        "speed_mps": round(disp_norm / max((fs[-1] - fs[0]) * dt, dt), 2) if n >= 2 else 0.0,
        "length": n,
        "first_frame": fs[0],
        "last_frame": fs[-1],
        "refFrame": ref_f,
        "keyframes": keyframes,
        "outside_frames": outside_frames,
    }
