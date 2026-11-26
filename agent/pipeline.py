"""
Pipeline xu ly 1 frame da nguon tin hieu (Da Guideline / Da Task):
  [A] CenterPoint-Pillar 0.20m
  [B] PointPillars-FPN 0.25m
  [C] Cum hinh hoc LiDAR (GroundGrid + DBSCAN + L-Shape OBB)
  [D] Camera 2D YOLO11s (Batch + High-res Crop)

Ho tro cau hinh dong theo Profile:
  - enabled_classes: loc triet de ngay tu dau (vd: Job khong co Barrier -> 0 Barrier xuat hien)
  - split_two_wheels: tach rieng Bicycle / Motorcycle hoac gop Two_wheels
  - merge_rider: bat/tat gop Pedestrian ngoi tren xe 2 banh
  - enable_size_clamp & class_size_ranges: bang kich thuoc dong theo tung Profile
  - max_distance_by_group & max_distance_by_class: pham vi khoang cach rieng theo tung class/nhom
"""

import math
from typing import Dict, List, Optional, Set

import numpy as np
from sklearn.cluster import DBSCAN

from geometry import (
    CLASS_GROUP,
    GROUP_MAX_DIST,
    VRU_CLASSES,
    GroundGrid,
    bev_iou,
    bev_overlap_metrics,
    canonicalize_barrier,
    classify_by_size,
    l_shape_rect,
    points_in_box,
    refine_box,
    size_fit_score,
    wrap_pi,
)
from odometry import extract_static_structure

CAM_SKIP = {"Trailer", "Construction_vehicle", "Barrier", "Traffic_cone"}
TWO_WHEEL_LIKE = {"Two_wheels", "Bicycle", "Motorcycle"}

SINGLE_MODEL_MIN_SCORE = {
    "Car": 0.30,
    "Truck": 0.36,
    "Bus": 0.38,
    "Trailer": 0.48,
    "Construction_vehicle": 0.48,
    "Pedestrian": 0.28,
    "Two_wheels": 0.28,
    "Bicycle": 0.28,
    "Motorcycle": 0.28,
    "Barrier": 0.30,
    "Traffic_cone": 0.30,
}
TWO_MODEL_MIN_SCORE = {
    "Car": 0.20,
    "Truck": 0.25,
    "Bus": 0.28,
    "Trailer": 0.34,
    "Construction_vehicle": 0.34,
    "Pedestrian": 0.20,
    "Two_wheels": 0.20,
    "Bicycle": 0.20,
    "Motorcycle": 0.20,
    "Barrier": 0.22,
    "Traffic_cone": 0.22,
}
MIN_FG_POINTS = {
    "Car": 8,
    "Truck": 12,
    "Bus": 14,
    "Trailer": 16,
    "Construction_vehicle": 14,
    "Pedestrian": 4,
    "Two_wheels": 5,
    "Bicycle": 4,
    "Motorcycle": 5,
    "Barrier": 5,
    "Traffic_cone": 3,
}


def _max_dist_for_class(
    cls: str,
    dist_limits: Dict[str, float],
    class_dist_limits: Optional[Dict[str, float]] = None,
) -> float:
    if class_dist_limits and cls in class_dist_limits:
        return float(class_dist_limits[cls])
    return float(dist_limits.get(CLASS_GROUP.get(cls, "Vehicle"), 50.0))


def _boxes_conflict(a: Dict, b: Dict) -> bool:
    iou, iomin = bev_overlap_metrics(a, b)
    if iou >= 0.08 or iomin >= 0.18:
        return True
    dist = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
    min_span = min(max(a["dx"], a["dy"]), max(b["dx"], b["dy"]))
    return dist < max(0.95, 0.42 * min_span)


def _merge_rider(dets: List[Dict]) -> List[Dict]:
    drop = set()
    for i, d in enumerate(dets):
        if d["className"] not in TWO_WHEEL_LIKE:
            continue
        for j, o in enumerate(dets):
            if (
                j != i
                and j not in drop
                and o["className"] == "Pedestrian"
                and math.hypot(d["x"] - o["x"], d["y"] - o["y"]) < 0.9
            ):
                top = max(d["z"] + d["dz"] * 0.5, o["z"] + o["dz"] * 0.5)
                bot = min(d["z"] - d["dz"] * 0.5, o["z"] - o["dz"] * 0.5)
                d["z"], d["dz"] = (top + bot) * 0.5, top - bot
                drop.add(j)
    return [d for k, d in enumerate(dets) if k not in drop]


def _dedup_single_source(
    raw: List[Dict],
    src_name: str,
    dist_limits: Optional[Dict[str, float]] = None,
    class_dist_limits: Optional[Dict[str, float]] = None,
    merge_rider: bool = True,
) -> List[Dict]:
    limits = dist_limits or GROUP_MAX_DIST
    ordered = [canonicalize_barrier(d) for d in sorted(raw, key=lambda x: -x["score"])]
    if merge_rider:
        ordered = _merge_rider(ordered)
    kept: List[Dict] = []
    for d in ordered:
        cls = d["className"]
        if math.hypot(d["x"], d["y"]) > _max_dist_for_class(cls, limits, class_dist_limits) + 2.0:
            continue
        dup = next((k for k in kept if _boxes_conflict(k, d)), None)
        if dup is not None:
            if cls != dup["className"]:
                dup.setdefault("altClasses", []).append([cls, round(d["score"], 3), src_name])
            continue
        item = dict(d)
        item["sources"] = [src_name]
        item["tags"] = []
        kept.append(item)
    return kept


def _fuse_two_detectors(cp_dets: List[Dict], pp_dets: List[Dict]) -> List[Dict]:
    used_pp = set()
    fused: List[Dict] = []

    for a in cp_dets:
        best_j, best_m = None, 0.0
        for j, b in enumerate(pp_dets):
            if j in used_pp:
                continue
            iou, iomin = bev_overlap_metrics(a, b)
            dist = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
            same_grp = CLASS_GROUP.get(a["className"], "Vehicle") == CLASS_GROUP.get(b["className"], "Vehicle")
            metric = max(iou, iomin * 0.7)
            if metric < 0.14 and same_grp and dist <= 1.4:
                metric = (1.45 - dist) * 0.2
            if metric > best_m:
                best_m, best_j = metric, j

        if best_j is not None:
            used_pp.add(best_j)
            b = pp_dets[best_j]
            if a["className"] == b["className"]:
                m = dict(a)
                m["sources"] = ["cp", "pp"]
                m["x"] = 0.7 * a["x"] + 0.3 * b["x"]
                m["y"] = 0.7 * a["y"] + 0.3 * b["y"]
                m["z"] = 0.7 * a["z"] + 0.3 * b["z"]
                if abs(wrap_pi(a["rz"] - b["rz"])) < math.radians(35):
                    m["rz"] = wrap_pi(a["rz"] + 0.3 * wrap_pi(b["rz"] - a["rz"]))
                m["score"] = min(0.98, max(a["score"], b["score"]) + 0.10)
            else:
                prefer_pp = (
                    b["score"] > a["score"] + 0.05
                    or (
                        a["className"] in ("Trailer", "Construction_vehicle")
                        and b["className"] in ("Car", "Truck", "Bus")
                        and b["score"] >= a["score"] - 0.12
                    )
                )
                primary, secondary, sec_src = (b, a, "cp") if prefer_pp else (a, b, "pp")
                m = dict(primary)
                m["sources"] = ["cp", "pp"]
                m["altClasses"] = list(a.get("altClasses", [])) + list(b.get("altClasses", []))
                m["altClasses"].append([secondary["className"], round(secondary["score"], 3), sec_src])
                m["tags"] = list(m.get("tags", [])) + ["DETECTOR_DISAGREE"]
                m["score"] = max(a["score"], b["score"])
            fused.append(m)
        else:
            fused.append(a)

    for j, b in enumerate(pp_dets):
        if j not in used_pp:
            if any(_boxes_conflict(k, b) for k in fused):
                continue
            fused.append(b)
    return fused


def _clusters(xyz: np.ndarray, grid: GroundGrid, max_radius: float = 48.0) -> List[Dict]:
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    m = (r > 2.5) & (r < max_radius)
    sub = xyz[m]
    if sub.shape[0] < 20:
        return []
    hag = sub[:, 2] - grid.height(sub[:, 0], sub[:, 1])
    fg = sub[(hag > 0.32) & (hag < 4.2)]
    if fg.shape[0] < 16:
        return []
    labels = DBSCAN(eps=0.65, min_samples=7).fit_predict(fg[:, :2])
    out = []
    for lb in np.unique(labels):
        if lb < 0:
            continue
        pts = fg[labels == lb]
        if pts.shape[0] < 14:
            continue
        cx, cy, L, W, yaw = l_shape_rect(pts[:, :2])
        if L > 18.5 or W > 3.4:
            continue
        g = float(grid.height(np.array([cx]), np.array([cy]))[0])
        H = float(np.percentile(pts[:, 2], 96)) - g
        if H < 0.70 or H > 4.2:
            continue
        out.append({
            "x": cx, "y": cy, "dx": max(L, 0.3), "dy": max(W, 0.3),
            "dz": max(H, 0.3), "z": g + H * 0.5, "rz": yaw,
            "n": int(pts.shape[0]), "used": False,
        })
    return out


def _decide_class(
    d: Dict,
    enabled_set: Optional[Set[str]] = None,
    size_overrides: Optional[Dict] = None,
) -> None:
    cam = d.get("camera")
    lc = d["className"]
    disagree = "DETECTOR_DISAGREE" in d["tags"]

    if not cam or not cam.get("cls"):
        if disagree:
            d["tags"].append("UNCERTAIN_CLASS")
        return

    cc, conf = cam["cls"], cam["conf"]
    if enabled_set is not None and cc not in enabled_set:
        return

    if cc == lc:
        if "cam" not in d["sources"]:
            d["sources"].append("cam")
        if disagree and "DETECTOR_DISAGREE" in d["tags"]:
            d["tags"].remove("DETECTOR_DISAGREE")
        return

    if lc in CAM_SKIP:
        if cc == "Truck" and lc in ("Trailer", "Construction_vehicle"):
            if "cam" not in d["sources"]:
                d["sources"].append("cam")
            return
        if conf >= 0.55:
            d["tags"].append("UNCERTAIN_CLASS")
        return

    if conf < 0.42:
        d["tags"].append("UNCERTAIN_CLASS")
        return

    fit_cam = size_fit_score(cc, d["dx"], d["dy"], d["dz"], size_overrides=size_overrides)
    fit_lid = size_fit_score(lc, d["dx"], d["dy"], d["dz"], size_overrides=size_overrides)
    alt_names = {a[0] for a in d.get("altClasses", [])}

    if fit_cam <= fit_lid + 0.15 or cc in alt_names or conf >= 0.70:
        d["altClass"] = lc
        d["className"] = cc
        if "cam" not in d["sources"]:
            d["sources"].append("cam")
        d["tags"].append("CLASS_FROM_CAMERA")
        if disagree and "DETECTOR_DISAGREE" in d["tags"]:
            d["tags"].remove("DETECTOR_DISAGREE")
    else:
        d["tags"].append("UNCERTAIN_CLASS")


class FramePipeline:
    def __init__(self, cp_engine, pp_engine, cam_checker):
        self.cp = cp_engine
        self.pp = pp_engine
        self.cam = cam_checker

    def process(
        self,
        xyz: np.ndarray,
        intensity: Optional[np.ndarray],
        calib: Optional[Dict],
        images: Optional[Dict],
        use_camera: bool = True,
        use_second_detector: bool = True,
        include_traffic_cone: bool = False,
        enabled_classes: Optional[List[str]] = None,
        split_two_wheels: bool = False,
        merge_rider: bool = True,
        enable_size_clamp: bool = True,
        class_size_ranges: Optional[Dict] = None,
        max_distance_by_group: Optional[Dict[str, float]] = None,
        max_distance_by_class: Optional[Dict[str, float]] = None,
    ) -> Dict:
        dist_limits = {**GROUP_MAX_DIST, **(max_distance_by_group or {})}
        enabled_set: Optional[Set[str]] = set(enabled_classes) if enabled_classes is not None else None

        grid = GroundGrid(xyz)
        static_pts = extract_static_structure(xyz, grid.global_z)

        raw_cp = (
            self.cp.predict(xyz, intensity=intensity, score_threshold=0.18, split_two_wheels=split_two_wheels)
            if self.cp
            else []
        )
        raw_pp = (
            self.pp.predict(xyz, score_threshold=0.28, split_two_wheels=split_two_wheels)
            if (use_second_detector and self.pp is not None)
            else []
        )

        if enabled_set is not None:
            raw_cp = [d for d in raw_cp if d.get("className") in enabled_set]
            raw_pp = [d for d in raw_pp if d.get("className") in enabled_set]
        elif not include_traffic_cone:
            raw_cp = [d for d in raw_cp if d.get("className") != "Traffic_cone"]
            raw_pp = [d for d in raw_pp if d.get("className") != "Traffic_cone"]

        cp_clean = _dedup_single_source(raw_cp, "cp", dist_limits, max_distance_by_class, merge_rider=merge_rider)
        pp_clean = _dedup_single_source(raw_pp, "pp", dist_limits, max_distance_by_class, merge_rider=merge_rider)
        merged = _fuse_two_detectors(cp_clean, pp_clean)

        max_cluster_r = min(65.0, max(list(dist_limits.values()) + [48.0]))
        clusters = _clusters(xyz, grid, max_radius=max_cluster_r)

        candidates: List[Dict] = []
        for m in sorted(merged, key=lambda x: -x["score"]):
            b = refine_box(
                xyz,
                m,
                grid,
                size_overrides=class_size_ranges,
                enable_size_clamp=enable_size_clamp,
            )
            cls = b["className"]
            n, s = b["pointCount"], m["score"]
            if n < max(3, MIN_FG_POINTS.get(cls, 5) // 2):
                continue
            if any(_boxes_conflict(k, b) for k in candidates):
                continue
            b["sources"] = list(m["sources"])
            b["tags"] = list(m.get("tags", []))
            if m.get("altClasses"):
                b["altClasses"] = m["altClasses"]
            b["score"] = float(s)
            candidates.append(b)

        for b in candidates:
            for c in clusters:
                if c["used"]:
                    continue
                inside = points_in_box(np.array([[c["x"], c["y"], b["z"]]]), b, margin_xy=0.35)
                if inside.size or bev_iou(b, c) > 0.18:
                    c["used"] = True
                    if "geo" not in b["sources"]:
                        b["sources"].append("geo")

        excl_geo = {"Trailer", "Construction_vehicle", "Traffic_cone"}
        if split_two_wheels:
            excl_geo.add("Two_wheels")
        else:
            excl_geo.update({"Bicycle", "Motorcycle"})

        for c in clusters:
            if c["used"]:
                continue
            cls, _ = classify_by_size(
                c["dx"],
                c["dy"],
                c["dz"],
                exclude=excl_geo,
                allowed_classes=enabled_set,
                size_overrides=class_size_ranges,
            )
            dist = math.hypot(c["x"], c["y"])
            if not cls or dist > _max_dist_for_class(cls, dist_limits, max_distance_by_class):
                continue
            seed = {
                "className": cls, "x": c["x"], "y": c["y"], "z": c["z"], "rz": c["rz"],
                "dx": c["dx"], "dy": c["dy"], "dz": c["dz"], "score": 0.28,
            }
            b = refine_box(
                xyz,
                seed,
                grid,
                size_overrides=class_size_ranges,
                enable_size_clamp=enable_size_clamp,
            )
            if b["pointCount"] < 16 or any(_boxes_conflict(k, b) for k in candidates):
                continue
            b.update({"sources": ["geo"], "tags": ["MISSED?"], "score": 0.28})
            candidates.append(b)

        cam_info = None
        cam_ran = False
        if use_camera and calib and images:
            cam_info = self.cam.check(
                candidates,
                calib,
                images,
                split_two_wheels=split_two_wheels,
                merge_rider=merge_rider,
            )
            cam_ran = bool(cam_info.get("ok"))
            if cam_ran:
                for d in candidates:
                    _decide_class(d, enabled_set=enabled_set, size_overrides=class_size_ranges)

        filtered: List[Dict] = []
        for d in sorted(candidates, key=lambda x: (-len(x["sources"]), -x["score"])):
            cls = d["className"]
            if enabled_set is not None and cls not in enabled_set:
                continue
            s = d["score"]
            n = d["pointCount"]
            has_cp = "cp" in d["sources"]
            has_pp = "pp" in d["sources"]
            has_cam = "cam" in d["sources"]
            both_det = has_cp and has_pp
            cam = d.get("camera") or {}

            if d["sources"] == ["geo"]:
                if cam_ran and not has_cam:
                    continue
                if not cam_ran and n < 45:
                    continue
            else:
                req_pts = MIN_FG_POINTS.get(cls, 6)
                if not has_cam and n < req_pts:
                    continue

                cam_full = has_cam and cam.get("via") == "full"
                if cam_full:
                    min_s = 0.22 if cls not in ("Trailer", "Construction_vehicle") else 0.32
                elif both_det:
                    min_s = TWO_MODEL_MIN_SCORE.get(cls, 0.25)
                else:
                    min_s = SINGLE_MODEL_MIN_SCORE.get(cls, 0.32)

                if s < min_s:
                    continue

                if (
                    cam_ran
                    and cls != "Barrier"
                    and cam.get("via") == "none"
                    and cam.get("area", 0) >= 1200
                    and cam.get("vis_ratio", 0.0) >= 0.45
                ):
                    if cls in VRU_CLASSES:
                        strict_thr = 0.78
                    elif cls in ("Trailer", "Construction_vehicle", "Bus"):
                        strict_thr = 0.62
                    else:
                        strict_thr = 0.52
                    if not (both_det and s >= strict_thr) and s < strict_thr + 0.10:
                        continue

            dist = math.hypot(d["x"], d["y"])
            d["distance"] = round(dist, 2)
            lim = _max_dist_for_class(cls, dist_limits, max_distance_by_class)
            if dist > lim:
                continue
            if any(_boxes_conflict(k, d) for k in filtered):
                continue

            if dist > lim - 2.0:
                d["tags"].append("UNCERTAIN_SCOPE")
            if enable_size_clamp and size_fit_score(cls, d["dx"], d["dy"], d["dz"], size_overrides=class_size_ranges) > 0.05:
                d["tags"].append("SIZE_OUT_OF_RANGE")
            if n < 5:
                d["tags"].append("LOW_VISIBILITY")
            d["tags"] = sorted(set(d["tags"]))
            filtered.append(d)

        return {
            "dets": filtered,
            "static_pts": static_pts,
            "camera": cam_info,
            "clusters": len(clusters),
            "raw_cp": len(raw_cp),
            "raw_pp": len(raw_pp),
        }
