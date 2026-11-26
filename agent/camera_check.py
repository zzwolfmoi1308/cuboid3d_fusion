"""
Kiem tra class bang Camera 2D (calib K + lidar_to_cam + YOLO11s FP32):
  - Ho tro giai nen context_image cua CVAT (ZIP da camera HOAC anh don JPEG/PNG), khop ten camera trong calib.
  - Ho tro tach rieng Bicycle / Motorcycle (split_two_wheels) va tuy chon gop/tach nguoi lai (merge_rider).
  - 2 tang suy luan: Batch full anh + High-res Crop quanh vung chieu.
"""

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from geometry import box_corners

COCO_IDS = [0, 1, 2, 3, 5, 7]
YOLO_WEIGHTS = Path(__file__).parent / "models" / "yolo11s.pt"

_BOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _coco_to_canonical(cid: int, split_two_wheels: bool = False) -> str:
    if cid == 0:
        return "Pedestrian"
    if cid == 1:
        return "Bicycle" if split_two_wheels else "Two_wheels"
    if cid == 2:
        return "Car"
    if cid == 3:
        return "Motorcycle" if split_two_wheels else "Two_wheels"
    if cid == 5:
        return "Bus"
    if cid == 7:
        return "Truck"
    return "Car"


def decode_context_zip(raw: bytes, calib: Dict) -> Dict[str, np.ndarray]:
    """Giai ma context_image tu CVAT (ZIP nhieu anh camera hoac 1 anh JPEG/PNG don le)."""
    import cv2

    cams_cfg = calib.get("cameras") or calib.get("calib") or {}
    cam_names = list(cams_cfg.keys())
    out: Dict[str, np.ndarray] = {}
    if not raw or len(raw) < 16:
        return out

    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            entries = sorted([n for n in zf.namelist() if not n.endswith("/")])
            stems = {n: Path(n).stem.upper() for n in entries}
            used_entries = set()
            for cam in sorted(cam_names, key=len, reverse=True):
                img_key = str(cams_cfg.get(cam, {}).get("image", cam)).upper()
                targets = {cam.upper(), Path(img_key).stem.upper()}
                hit = [n for n in entries if n not in used_entries and stems[n] in targets]
                if not hit:
                    hit = [
                        n for n in entries
                        if n not in used_entries
                        and any(t in n.upper() for t in targets)
                        and not any(
                            o != cam and o.upper() in n.upper() and len(o) > len(cam)
                            for o in cam_names
                        )
                    ]
                if hit:
                    used_entries.add(hit[0])
                    buf = np.frombuffer(zf.read(hit[0]), dtype=np.uint8)
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if img is not None:
                        out[cam] = img
            if not out and len(entries) == len(cam_names):
                for cam, entry in zip(cam_names, entries):
                    buf = np.frombuffer(zf.read(entry), dtype=np.uint8)
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if img is not None:
                        out[cam] = img
    elif cam_names:
        # Fallback khi CVAT tra ve 1 anh don le (khong nen ZIP)
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            out[cam_names[0]] = img
    return out


def _extract_cam_matrices(cam: Dict) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
    if "lidar_to_cam" in cam and "K" in cam:
        T = np.asarray(cam["lidar_to_cam"], dtype=np.float64)
        K = np.asarray(cam["K"], dtype=np.float64)
        w = float(cam.get("width", 1600))
        h = float(cam.get("height", 900))
        return T, K, w, h
    if "cam_to_velo" in cam and "cam_intrinsic" in cam:
        c2v = np.asarray(cam["cam_to_velo"], dtype=np.float64)
        T = np.linalg.inv(c2v)
        K = np.asarray(cam["cam_intrinsic"], dtype=np.float64)
        sz = cam.get("image_size") or [1920, 1020]
        return T, K, float(sz[0]), float(sz[1])
    if ("lidar2cam" in cam or "extrinsic" in cam) and ("cam2img" in cam or "intrinsic" in cam):
        T = np.asarray(cam.get("lidar2cam") or cam.get("extrinsic"), dtype=np.float64)
        K = np.asarray(cam.get("cam2img") or cam.get("intrinsic"), dtype=np.float64)
        sz = cam.get("image_size") or [cam.get("width", 1600), cam.get("height", 900)]
        return T, K, float(sz[0]), float(sz[1])
    return None


def project_box(b: Dict, cam: Dict, z_near: float = 0.25) -> Optional[Dict]:
    mats = _extract_cam_matrices(cam)
    if mats is None:
        return None
    T, K, w, h = mats

    center_cam = T[:3, :3] @ np.array([b["x"], b["y"], b["z"]], dtype=np.float64) + T[:3, 3]
    if center_cam[2] < 0.45:
        return None

    c = box_corners(b)
    pc = (T[:3, :3] @ c.T + T[:3, 3:4]).T
    if np.sum(pc[:, 2] >= z_near) < 2:
        return None

    pts_cam: List[np.ndarray] = [p for p in pc if p[2] >= z_near]
    if len(pts_cam) < 8:
        for i, j in _BOX_EDGES:
            p1, p2 = pc[i], pc[j]
            if (p1[2] >= z_near) != (p2[2] >= z_near):
                t = (z_near - p1[2]) / (p2[2] - p1[2])
                pts_cam.append(p1 + t * (p2 - p1))

    arr = np.asarray(pts_cam, dtype=np.float64).T
    uv = (K @ arr)[:2] / arr[2]
    x1, y1, x2, y2 = float(uv[0].min()), float(uv[1].min()), float(uv[0].max()), float(uv[1].max())
    cx1, cy1, cx2, cy2 = max(x1, 0.0), max(y1, 0.0), min(x2, w), min(y2, h)
    if cx2 - cx1 < 8.0 or cy2 - cy1 < 8.0:
        return None
    full = (x2 - x1) * (y2 - y1)
    vis = (cx2 - cx1) * (cy2 - cy1)
    if full <= 0.0 or (vis / full < 0.35 and vis < 6400.0):
        return None
    return {
        "bbox": [cx1, cy1, cx2, cy2],
        "area": vis,
        "depth": float(center_cam[2]),
        "vis_ratio": round(vis / full, 2),
    }


def iou2d(a: List[float], b: List[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _install_safe_torchvision_nms() -> None:
    try:
        import torchvision.ops as tv_ops

        if getattr(tv_ops, "_fusion_safe_nms_installed", False):
            return
        orig_nms = tv_ops.nms

        def safe_nms(boxes, scores, iou_threshold: float):
            if boxes.is_cuda:
                try:
                    return orig_nms(boxes, scores, iou_threshold)
                except NotImplementedError:
                    return orig_nms(boxes.cpu(), scores.cpu(), iou_threshold).to(boxes.device)
            return orig_nms(boxes, scores, iou_threshold)

        tv_ops.nms = safe_nms
        tv_ops._fusion_safe_nms_installed = True
    except Exception:
        pass


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


class CameraChecker:
    def __init__(self):
        self.model = None
        self.error: Optional[str] = None

    def _load(self):
        if self.model is not None or self.error:
            return
        try:
            _install_safe_torchvision_nms()
            from ultralytics import YOLO
            YOLO_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            w = str(YOLO_WEIGHTS) if YOLO_WEIGHTS.exists() else "yolo11s.pt"
            self.model = YOLO(w)
            if not YOLO_WEIGHTS.exists() and Path("yolo11s.pt").exists():
                Path("yolo11s.pt").replace(YOLO_WEIGHTS)
        except Exception as exc:
            self.error = f"Khong nap duoc YOLO11s: {exc}"

    def _crop_verify(
        self,
        img: np.ndarray,
        pbox: List[float],
        dev,
        claimed_boxes: Optional[List[List[float]]] = None,
        split_two_wheels: bool = False,
        merge_rider: bool = True,
    ) -> Optional[Tuple[str, float, float]]:
        h, w = img.shape[:2]
        bw, bh = pbox[2] - pbox[0], pbox[3] - pbox[1]
        pad_x = max(bw * 0.35, 36.0)
        pad_y = max(bh * 0.35, 36.0)
        x0 = int(max(0, pbox[0] - pad_x))
        y0 = int(max(0, pbox[1] - pad_y))
        x1 = int(min(w, pbox[2] + pad_x))
        y1 = int(min(h, pbox[3] + pad_y))
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None
        crop = img[y0:y1, x0:x1]
        local_pbox = [pbox[0] - x0, pbox[1] - y0, pbox[2] - x0, pbox[3] - y0]

        res = self.model.predict(
            crop,
            imgsz=384,
            conf=0.22,
            classes=COCO_IDS,
            verbose=False,
            device=dev,
        )[0]
        if len(res.boxes) == 0:
            return None

        dets_c = [
            (res.boxes.xyxy[k].tolist(), int(res.boxes.cls[k]), float(res.boxes.conf[k]))
            for k in range(len(res.boxes))
        ]
        claimed = claimed_boxes or []
        best = None
        for bb, c_id, conf in dets_c:
            ov = iou2d(local_pbox, bb)
            if ov < 0.25:
                continue
            full_bb = [bb[0] + x0, bb[1] + y0, bb[2] + x0, bb[3] + y0]
            if any(iou2d(full_bb, cb) >= 0.40 for cb in claimed):
                continue
            gcls = _coco_to_canonical(c_id, split_two_wheels=split_two_wheels)
            if gcls == "Pedestrian" and merge_rider:
                for bb2, c2, cf2 in dets_c:
                    if c2 in (1, 3) and iou2d(bb, bb2) >= 0.18:
                        gcls = _coco_to_canonical(c2, split_two_wheels=split_two_wheels)
                        conf = max(conf, cf2)
                        break
            cand = (ov * 0.6 + conf * 0.4, gcls, conf, ov)
            if best is None or cand[0] > best[0]:
                best = cand
        return (best[1], best[2], best[3]) if best else None

    def check(
        self,
        dets: List[Dict],
        calib: Dict,
        images: Dict[str, np.ndarray],
        split_two_wheels: bool = False,
        merge_rider: bool = True,
    ) -> Dict:
        self._load()
        if self.model is None:
            return {"ok": False, "error": self.error}

        cams = calib.get("cameras") or calib.get("calib") or {}
        ordered_names = [n for n in cams if n in images]
        if not ordered_names:
            return {"ok": False, "error": "Khong co anh camera khop ten trong calib"}

        dev = 0 if _cuda() else "cpu"
        batch_imgs = [images[n] for n in ordered_names]
        batch_res = self.model.predict(
            batch_imgs,
            imgsz=960,
            conf=0.22,
            classes=COCO_IDS,
            verbose=False,
            device=dev,
        )
        yolo: Dict[str, List] = {}
        for name, r in zip(ordered_names, batch_res):
            yolo[name] = [
                (r.boxes.xyxy[i].tolist(), int(r.boxes.cls[i]), float(r.boxes.conf[i]))
                for i in range(len(r.boxes))
            ]

        proj: List[Optional[Tuple[str, Dict]]] = []
        for d in dets:
            best = None
            for name in ordered_names:
                cam_cfg = dict(cams[name])
                if "width" not in cam_cfg and "image_size" not in cam_cfg:
                    ih, iw = images[name].shape[:2]
                    cam_cfg["width"], cam_cfg["height"] = iw, ih
                p = project_box(d, cam_cfg)
                if p and (best is None or p["area"] > best[1]["area"]):
                    best = (name, p)
            proj.append(best)

        pairs = []
        for i, pb in enumerate(proj):
            if not pb:
                continue
            name, p = pb
            for j, (bb, cls, conf) in enumerate(yolo[name]):
                v = iou2d(p["bbox"], bb)
                if v >= 0.25:
                    pairs.append((v - 0.002 * p["depth"], i, name, j))
        pairs.sort(reverse=True)
        used_d, used_y = set(), set()
        claimed_by_cam: Dict[str, List[Tuple[float, List[float]]]] = {}
        for _, i, name, j in pairs:
            if i in used_d or (name, j) in used_y:
                continue
            used_d.add(i)
            used_y.add((name, j))
            bb, cls, conf = yolo[name][j]
            claimed_by_cam.setdefault(name, []).append((proj[i][1]["depth"], bb))
            gcls = _coco_to_canonical(cls, split_two_wheels=split_two_wheels)
            if gcls == "Pedestrian" and merge_rider:
                for bb2, c2, cf2 in yolo[name]:
                    if c2 in (1, 3) and iou2d(bb, bb2) >= 0.20:
                        gcls = _coco_to_canonical(c2, split_two_wheels=split_two_wheels)
                        conf = max(conf, cf2)
                        break
            dets[i]["camera"] = {
                "cam": name,
                "cls": gcls,
                "conf": round(conf, 3),
                "iou": round(iou2d(proj[i][1]["bbox"], bb), 3),
                "bbox": [round(v) for v in proj[i][1]["bbox"]],
                "area": round(proj[i][1]["area"]),
                "vis_ratio": proj[i][1]["vis_ratio"],
                "via": "full",
            }

        crop_budget = 14
        for i, pb in enumerate(proj):
            if not pb:
                continue
            name, p = pb
            d = dets[i]
            need_crop = (
                "camera" not in d
                or (d["camera"]["cls"] != d["className"] and d["camera"]["conf"] < 0.65)
                or "DETECTOR_DISAGREE" in d.get("tags", [])
                or "MISSED?" in d.get("tags", [])
            )
            if need_crop and crop_budget > 0 and p["area"] >= 500.0:
                crop_budget -= 1
                closer_claimed = [
                    cb for dep, cb in claimed_by_cam.get(name, []) if dep < p["depth"] - 0.5
                ]
                cr = self._crop_verify(
                    images[name],
                    p["bbox"],
                    dev,
                    claimed_boxes=closer_claimed,
                    split_two_wheels=split_two_wheels,
                    merge_rider=merge_rider,
                )
                if cr is not None:
                    gcls, conf, ov = cr
                    prev_conf = d.get("camera", {}).get("conf", 0.0)
                    if "camera" not in d or conf > prev_conf:
                        d["camera"] = {
                            "cam": name,
                            "cls": gcls,
                            "conf": round(conf, 3),
                            "iou": round(ov, 3),
                            "bbox": [round(v) for v in p["bbox"]],
                            "area": round(p["area"]),
                            "vis_ratio": p["vis_ratio"],
                            "via": "crop",
                        }
            if "camera" not in d:
                d["camera"] = {
                    "cam": name,
                    "cls": None,
                    "conf": 0.0,
                    "iou": 0.0,
                    "bbox": [round(v) for v in p["bbox"]],
                    "area": round(p["area"]),
                    "vis_ratio": p["vis_ratio"],
                    "via": "none",
                }

        return {
            "ok": True,
            "images": ordered_names,
            "total_cams": len(cams),
            "yolo_counts": {k: len(v) for k, v in yolo.items()},
        }
