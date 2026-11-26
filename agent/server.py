"""
CVAT 3D Fusion Pre-label Agent (FastAPI · 127.0.0.1:8766) — 100% noi bo, Da Guideline / Da Task.
Ket hop 4 nguon tin hieu doc lap tren Quadro P2000 (FP32) + ho tro cau hinh Profile dong & trich xuat Guideline file (.docx/.pdf/.txt/.md/.csv).
"""

import base64
import io
import json
import re
import time
import xml.etree.ElementTree as ET
import zipfile
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from camera_check import CameraChecker, YOLO_WEIGHTS, decode_context_zip
from odometry import register_frames_icp
from pcd_io import parse_pcd_bytes
from pipeline import FramePipeline
from tracker import build_tracks

PORT = 8766
MODELS_DIR = Path(__file__).parent / "models"

app = FastAPI(title="CVAT 3D Fusion Pre-label Agent", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

cp_engine, cp_info = None, {}
try:
    import torch
    from centerpoint_pillar import CenterPointInferenceEngine

    ckpt_cp = MODELS_DIR / "centerpoint_pillar02_nus.pth"
    cp_engine = CenterPointInferenceEngine(str(ckpt_cp), prefer_cuda=True)
    cp_info = {
        "loaded": True,
        "device": str(cp_engine.device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
except Exception as exc:
    cp_info = {"loaded": False, "error": str(exc)}

pp_engine, pp_info = None, {}
try:
    from pointpillars_fpn import PointPillarsFPNEngine

    ckpt_pp = MODELS_DIR / "pointpillars_fpn_nus.pth"
    if ckpt_pp.exists():
        pp_engine = PointPillarsFPNEngine(str(ckpt_pp), prefer_cuda=True)
        pp_info = {"loaded": True, "device": str(pp_engine.device)}
    else:
        pp_info = {"loaded": False, "error": "Chua co file pointpillars_fpn_nus.pth"}
except Exception as exc:
    pp_info = {"loaded": False, "error": str(exc)}

camera = CameraChecker()
pipeline = FramePipeline(cp_engine, pp_engine, camera)
SESSIONS: Dict[int, Dict[int, Dict]] = {}


class FrameReq(BaseModel):
    job_id: int
    frame: int
    seq: str
    pcd_base64: str
    calib: Optional[dict] = None
    images_zip_base64: Optional[str] = None
    use_camera: bool = True
    use_second_detector: bool = True
    use_odometry: bool = True
    include_traffic_cone: bool = False
    enabled_classes: Optional[List[str]] = None
    split_two_wheels: bool = False
    merge_rider: bool = True
    enable_size_clamp: bool = True
    class_size_ranges: Optional[Dict] = None
    max_distance_by_group: Optional[Dict[str, float]] = None
    max_distance_by_class: Optional[Dict[str, float]] = None


class TracksReq(BaseModel):
    job_id: int
    barrier_axis_mode: str = "length_is_x"
    label_mapping: Optional[Dict[str, str]] = None
    frame_dt: float = 0.5
    max_frame_gap: int = 4
    occluded_min_points: int = 5
    lock_track_size: bool = True


class LocalCalibReq(BaseModel):
    path: str


class GuidelineFileReq(BaseModel):
    filename: str
    file_base64: str


@app.get("/health")
def health():
    vram_mb = None
    try:
        import torch
        if torch.cuda.is_available():
            vram_mb = round(torch.cuda.memory_allocated() / 1e6, 1)
    except Exception:
        pass
    return {
        "ok": True,
        "service": "cvat3d_fusion_agent",
        "version": "1.2.0",
        "port": PORT,
        "vram_allocated_mb": vram_mb,
        "centerpoint": cp_info,
        "pointpillars": pp_info,
        "yolo": "loaded" if camera.model else (camera.error or "ready_lazy"),
    }


@app.post("/reset")
def reset(req: TracksReq):
    SESSIONS.pop(req.job_id, None)
    return {"ok": True}


@app.post("/frame")
def frame(req: FrameReq):
    t0 = time.perf_counter()
    try:
        parsed = parse_pcd_bytes(base64.b64decode(req.pcd_base64))
    except Exception as exc:
        raise HTTPException(400, f"Loi giai ma PCD: {exc}")

    images, img_err = None, None
    total_cams = len((req.calib or {}).get("cameras") or (req.calib or {}).get("calib") or {})
    if req.use_camera and req.calib and req.images_zip_base64:
        try:
            raw_zip = base64.b64decode(req.images_zip_base64)
            images = decode_context_zip(raw_zip, req.calib)
            if not images:
                img_err = f"Khong giai ma duoc context_image (len={len(raw_zip)})"
        except Exception as exc:
            img_err = f"Loi giai nen anh camera: {exc}"

    try:
        res = pipeline.process(
            parsed["xyz"],
            parsed.get("intensity"),
            req.calib,
            images,
            use_camera=req.use_camera,
            use_second_detector=req.use_second_detector,
            include_traffic_cone=req.include_traffic_cone,
            enabled_classes=req.enabled_classes,
            split_two_wheels=req.split_two_wheels,
            merge_rider=req.merge_rider,
            enable_size_clamp=req.enable_size_clamp,
            class_size_ranges=req.class_size_ranges,
            max_distance_by_group=req.max_distance_by_group,
            max_distance_by_class=req.max_distance_by_class,
        )
    except Exception as exc:
        raise HTTPException(500, f"Loi pipeline: {exc}")

    job_store = SESSIONS.setdefault(req.job_id, {})
    prev_entry = job_store.get(req.frame - 1)
    odom = {"valid": False, "fitness": 0.0, "rmse": 99.0, "dx": 0.0, "dy": 0.0, "dyaw_deg": 0.0}
    if (
        req.use_odometry
        and prev_entry is not None
        and prev_entry["seq"] == req.seq
        and prev_entry.get("static_pts") is not None
    ):
        odom = register_frames_icp(prev_entry["static_pts"], res["static_pts"])

    job_store[req.frame] = {
        "seq": req.seq,
        "dets": res["dets"],
        "static_pts": res["static_pts"] if req.use_odometry else None,
        "odom_from_prev": odom,
    }
    old_f = req.frame - 2
    if old_f in job_store and job_store[old_f].get("static_pts") is not None:
        job_store[old_f]["static_pts"] = None

    summary = [
        {
            "className": d["className"],
            "sources": d["sources"],
            "tags": d["tags"],
            "score": round(d["score"], 3),
            "pointCount": d["pointCount"],
            "distance": d["distance"],
        }
        for d in res["dets"]
    ]
    ms = round((time.perf_counter() - t0) * 1000)
    cam_denom = total_cams or (len(images) if images else 0)
    print(
        f"[Fusion] job={req.job_id} f={req.frame} seq={req.seq} pts={parsed['valid_points']} "
        f"cp={res['raw_cp']} pp={res['raw_pp']} clus={res['clusters']} -> dets={len(summary)} "
        f"odom={'OK' if odom['valid'] else 'skip'} imgs={len(images) if images else 0}/{cam_denom} "
        f"cam={(res['camera'] or {}).get('yolo_counts')} err={img_err} ({ms}ms)"
    )
    return {
        "ok": True,
        "frame": req.frame,
        "count": len(summary),
        "raw_cp": res["raw_cp"],
        "raw_pp": res["raw_pp"],
        "dets": summary,
        "odometry": {k: v for k, v in odom.items() if k != "T"},
        "camera": res["camera"],
        "images_found": list(images) if images else [],
        "total_cams": cam_denom,
        "image_error": img_err,
        "elapsed_ms": ms,
    }


@app.post("/load_local_calib")
def load_local_calib(req: LocalCalibReq):
    root = Path(req.path.strip().strip('"').strip("'"))
    if not root.exists():
        raise HTTPException(404, f"Khong tim thay duong dan: {root}")

    calibs: Dict[str, dict] = {}
    if root.is_file():
        calibs["default"] = json.loads(root.read_text(encoding="utf-8"))
    else:
        direct = root / "calib.json"
        if direct.is_file():
            calibs["default"] = json.loads(direct.read_text(encoding="utf-8"))
        for p in sorted(root.rglob("*calib*.json")):
            if p == direct:
                continue
            parent_name = p.parent.name.lower()
            m = re.search(r"sequence[_-]?(\d+)", parent_name)
            key = f"s{int(m.group(1)):02d}" if m else parent_name
            calibs[key] = json.loads(p.read_text(encoding="utf-8"))
            if "default" not in calibs:
                calibs["default"] = calibs[key]

    if not calibs:
        raise HTTPException(404, f"Khong tim thay file calib JSON nao trong {root}")
    return {"ok": True, "count": len(calibs), "keys": sorted(calibs.keys()), "calibs": calibs}


def _extract_docx_text(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        xml_bytes = zf.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    ns_w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    lines: List[str] = []
    for elem in root.iter():
        if elem.tag == f"{ns_w}tr":
            cells = []
            for tc in elem.findall(f".//{ns_w}tc"):
                txt = "".join(t.text or "" for t in tc.findall(f".//{ns_w}t")).strip()
                cells.append(txt)
            if any(cells):
                lines.append("| " + " | ".join(cells) + " |")
        elif elem.tag == f"{ns_w}p":
            # Chi them doan van khong nam trong bang
            txt = "".join(t.text or "" for t in elem.findall(f".//{ns_w}t")).strip()
            if txt and (not lines or txt not in lines[-1]):
                lines.append(txt)
    return "\n".join(lines)


def _extract_pdf_text(raw: bytes) -> str:
    for mod_name in ("pypdf", "PyPDF2"):
        try:
            mod = __import__(mod_name)
            reader = mod.PdfReader(io.BytesIO(raw))
            pages = [p.extract_text() or "" for p in reader.pages]
            joined = "\n".join(pages).strip()
            if joined:
                return joined
        except Exception:
            pass
    try:
        import fitz
        doc = fitz.open(stream=raw, filetype="pdf")
        joined = "\n".join(page.get_text("text") for page in doc).strip()
        if joined:
            return joined
    except Exception:
        pass

    # Pure-Python stdlib PDF FlateDecode stream fallback
    chunks: List[str] = []
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.DOTALL):
        stream_data = m.group(1)
        try:
            dec = zlib.decompress(stream_data)
        except Exception:
            dec = stream_data
        for tm in re.finditer(rb"\(([^()\\]*(?:\\.[^()\\]*)*)\)\s*Tj", dec):
            s = tm.group(1).decode("latin1", errors="ignore")
            chunks.append(s)
        for arr in re.finditer(rb"\[(.*?)\]\s*TJ", dec, re.DOTALL):
            parts = re.findall(rb"\(([^()\\]*(?:\\.[^()\\]*)*)\)", arr.group(1))
            if parts:
                chunks.append("".join(p.decode("latin1", errors="ignore") for p in parts))
    return "\n".join(chunks).strip()


@app.post("/extract_guideline_text")
def extract_guideline_text(req: GuidelineFileReq):
    raw = base64.b64decode(req.file_base64)
    ext = Path(req.filename).suffix.lower()
    if ext == ".docx":
        text = _extract_docx_text(raw)
    elif ext == ".pdf":
        text = _extract_pdf_text(raw)
    else:
        text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        raise HTTPException(
            422,
            "Khong trich xuat duoc chu tu file nay (co the la PDF scan anh). Hay dan noi dung bang/quy tac vao o Text hoac dung JSON/CSV.",
        )
    return {"ok": True, "text": text}


@app.post("/tracks")
def tracks(req: TracksReq):
    frames = SESSIONS.get(req.job_id)
    if not frames:
        raise HTTPException(404, "Chua co frame nao duoc xu ly cho job nay.")
    out = build_tracks(
        frames,
        barrier_axis_mode=req.barrier_axis_mode,
        frame_dt=req.frame_dt,
        max_frame_gap=req.max_frame_gap,
        occluded_min_points=req.occluded_min_points,
        lock_track_size=req.lock_track_size,
    )
    if req.label_mapping:
        remapped = []
        for t in out:
            canon = t["className"]
            t["canonicalClass"] = canon
            if canon in req.label_mapping:
                target = str(req.label_mapping[canon] or "").strip()
                if not target or target == "__IGNORE__":
                    continue
                t["className"] = target
            remapped.append(t)
        out = remapped
    return json.loads(json.dumps({"ok": True, "count": len(out), "tracks": out}, default=float))


if __name__ == "__main__":
    print("=" * 68)
    print(f"  CVAT 3D FUSION PRE-LABEL AGENT v1.2  |  http://127.0.0.1:{PORT}")
    print(f"  Model A (CenterPoint-Pillar) : {cp_info}")
    print(f"  Model B (PointPillars-FPN)   : {pp_info}")
    print(f"  Camera  (YOLO11s FP32)       : {YOLO_WEIGHTS.name} ({'ready' if YOLO_WEIGHTS.exists() else 'auto'})")
    print("=" * 68)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
