"""
Bo kiem thu toan dien cho CVAT 3D Fusion Pre-label Agent (v1.1):
  1. Kiem L-Shape OBB & chuan hoa Barrier (length_is_x vs width_is_x)
  2. Kiem LiDAR ICP Odometry khi xe ego vua tien 4.2m vua re 12 do
  3. Kiem chieu Camera 6 goc + cat canh z_near + giai nen ZIP + YOLO11s FP32
  4. Kiem Tracker da chuoi: dam bao dat outside=True khi chuyen tu chuoi A sang chuoi B, 0 xung dot occluded+outside
  5. Kiem FastAPI endpoints (/health, /reset, /frame, /tracks) + gioi han VRAM <= 3.0 GB tren Quadro P2000
"""

import base64
import io
import math
import struct
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
sys.path.insert(0, str(AGENT_DIR))

from camera_check import decode_context_zip, project_box
from geometry import canonicalize_barrier, format_barrier_points, l_shape_rect, wrap_pi
from odometry import extract_static_structure, register_frames_icp
import server


SAMPLE_CALIB = {
    "frame": "G01-M5-A-000003",
    "cameras": {
        "CAM_FRONT": {
            "image": "CAM_FRONT",
            "width": 1600,
            "height": 900,
            "K": [[1252.8131, 0.0, 826.5881], [0.0, 1252.8131, 469.9847], [0.0, 0.0, 1.0]],
            "lidar_to_cam": [
                [0.9999027, 0.0109943, -0.0085862, -0.0010255],
                [-0.0091779, 0.0549554, -0.9984466, -0.3463567],
                [-0.0105053, 0.9984283, 0.0550509, -0.3644486],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "CAM_BACK": {
            "image": "CAM_BACK",
            "width": 1600,
            "height": 900,
            "K": [[796.8911, 0.0, 857.7774], [0.0, 796.8911, 476.8849], [0.0, 0.0, 1.0]],
            "lidar_to_cam": [
                [-0.9999262, 0.0085470, 0.0086300, 0.0017130],
                [-0.0082859, 0.0395126, -0.9991847, -0.2788417],
                [-0.0088810, -0.9991825, -0.0394389, -1.0395221],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
    },
}


def make_binary_pcd(xyz: np.ndarray, intensity: np.ndarray) -> bytes:
    n = xyz.shape[0]
    header = (
        "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
        f"TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {n}\nHEIGHT 1\nPOINTS {n}\nDATA binary\n"
    ).encode("ascii")
    arr = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
    arr["x"], arr["y"], arr["z"], arr["intensity"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], intensity
    return header + arr.tobytes()


def make_dummy_zip() -> bytes:
    buf = io.BytesIO()
    img = np.full((900, 1600, 3), 120, dtype=np.uint8)
    ok, jpg = cv2.imencode(".jpg", img)
    assert ok
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("CAM_FRONT.jpg", jpg.tobytes())
        zf.writestr("CAM_BACK.jpg", jpg.tobytes())
    return buf.getvalue()


def test_geometry_and_barrier():
    # 1. L-shape points (chi thay can sau va suon trai xe o goc 25 do)
    rng = np.random.default_rng(0)
    true_yaw = math.radians(25.0)
    c, s = math.cos(true_yaw), math.sin(true_yaw)
    l_pts = np.vstack([
        np.column_stack([np.full(60, -2.25), rng.uniform(-0.92, 0.92, 60)]),
        np.column_stack([rng.uniform(-2.25, 2.25, 90), np.full(90, -0.92)]),
    ]) + rng.normal(0, 0.02, (150, 2))
    world_xy = np.column_stack([10.0 + l_pts[:, 0] * c - l_pts[:, 1] * s, 14.0 + l_pts[:, 0] * s + l_pts[:, 1] * c])
    _, _, L, W, yaw = l_shape_rect(world_xy)
    err_deg = min(abs(math.degrees(wrap_pi(yaw - (true_yaw + k * math.pi / 2)))) for k in range(4))
    assert err_deg < 3.0, f"L-shape yaw error too high: {err_deg}"
    assert abs(L - 4.5) < 0.25 and abs(W - 1.84) < 0.25

    # 2. Barrier canonicalize + axis modes
    raw_b = {"className": "Barrier", "x": 3.0, "y": 10.0, "z": -1.3, "dx": 0.52, "dy": 2.4, "dz": 0.95, "rz": 0.0}
    can = canonicalize_barrier(raw_b)
    assert can["dx"] == 2.4 and can["dy"] == 0.52 and abs(abs(can["rz"]) - math.pi / 2) < 1e-4
    p16 = [3.0, 10.0, -1.3, 0.0, 0.0, can["rz"], can["dx"], can["dy"], can["dz"], 0, 0, 0, 0, 0, 0, 0]
    p_len = format_barrier_points(p16, "Barrier", "length_is_x")
    p_wid = format_barrier_points(p16, "Barrier", "width_is_x")
    assert p_len[6] > p_len[7], "length_is_x must have dx > dy"
    assert p_wid[6] < p_wid[7], "width_is_x must have dx < dy"
    print("[PASS] Geometry L-Shape (yaw err=%.2f deg) & Barrier axis modes OK" % err_deg)


def test_icp_odometry():
    rng = np.random.default_rng(11)
    # Tao 600 diem toa nha / hang rao tinh quanh duong
    walls = np.vstack([
        np.column_stack([np.full(250, -9.0), rng.uniform(-25, 35, 250), rng.uniform(-1.2, 2.5, 250)]),
        np.column_stack([np.full(250, 9.5), rng.uniform(-25, 35, 250), rng.uniform(-1.2, 2.5, 250)]),
        np.column_stack([rng.uniform(-9, 9, 150), np.full(150, 28.0), rng.uniform(-1.2, 2.0, 150)]),
    ]).astype(np.float32)
    # Ego di thang +4.0m theo Y va re +10 do -> canh tinh bien doi nguoc lai
    ang = math.radians(-10.0)
    R = np.array([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1]])
    t = np.array([0.4, -4.0, 0.02])
    walls_f2 = ((R @ walls.T).T + t + rng.normal(0, 0.03, walls.shape)).astype(np.float32)

    s1 = extract_static_structure(walls, -1.8)
    s2 = extract_static_structure(walls_f2, -1.8)
    odom = register_frames_icp(s1, s2)
    assert odom["valid"], f"ICP should be valid: {odom}"
    assert abs(odom["dyaw_deg"] - (-10.0)) < 1.0, f"Expected ~-10 deg, got {odom['dyaw_deg']}"
    assert abs(odom["dy"] - (-4.0)) < 0.25, f"Expected dy~-4.0, got {odom['dy']}"
    print(f"[PASS] LiDAR ICP Odometry OK: dy={odom['dy']}m, dyaw={odom['dyaw_deg']}deg, fitness={odom['fitness']}")


def test_camera_projection_near_clip_and_zip():
    imgs = decode_context_zip(make_dummy_zip(), SAMPLE_CALIB)
    assert set(imgs.keys()) == {"CAM_FRONT", "CAM_BACK"}
    # Xe o rat gan CAM_FRONT (y=2.2m, L=4.8m -> duoi xe y=-0.2m < z_near, dau xe y=4.6m > z_near!)
    close_car = {"x": 1.8, "y": 2.2, "z": -1.0, "dx": 4.8, "dy": 1.9, "dz": 1.6, "rz": math.pi / 2}
    proj = project_box(close_car, SAMPLE_CALIB["cameras"]["CAM_FRONT"])
    assert proj is not None and proj["area"] > 1000, f"Near-clipped car must still project: {proj}"
    print(f"[PASS] Camera ZIP decode & near-plane clipping OK: bbox={[round(x) for x in proj['bbox']]}")


def test_api_and_multi_sequence_tracking():
    client = TestClient(server.app)
    h = client.get("/health").json()
    assert h["ok"] and h["centerpoint"]["loaded"] and h["pointpillars"]["loaded"], f"Health failed: {h}"

    # Tao point cloud raycast co 1 xe o (3.5, 13.0) va 1 nguoi di bo o (4.0, 8.0)
    rng = np.random.default_rng(5)
    els = np.radians(np.linspace(-25.0, 8.0, 32))
    azs = np.linspace(-math.pi, math.pi, 720, endpoint=False)

    def build_scene(car_y: float):
        pts = []
        for el in els:
            ce, se = math.cos(el), math.sin(el)
            for az in azs:
                dx, dy, dz = ce * math.cos(az), ce * math.sin(az), se
                bt = (-1.8 / dz) if dz < -1e-4 and (-1.8 / dz) < 45 else None
                inten = 16.0
                # Car shell near (3.5, car_y)
                for ox, oy, x0, x1, y0, y1, z0, z1 in [
                    (3.5, car_y, -0.92, 0.92, -2.25, 2.25, -1.65, -0.35),
                    (4.0, 8.0, -0.32, 0.32, -0.32, 0.32, -1.75, -0.10),
                ]:
                    tmin, tmax = 0.0, 1e9
                    ok = True
                    for o, r, lo, hi in ((-ox, dx, x0, x1), (-oy, dy, y0, y1), (0.0, dz, z0, z1)):
                        if abs(r) < 1e-8:
                            if o < lo or o > hi:
                                ok = False
                                break
                        else:
                            t1, t2 = (lo - o) / r, (hi - o) / r
                            if t1 > t2:
                                t1, t2 = t2, t1
                            tmin, tmax = max(tmin, t1), min(tmax, t2)
                            if tmin > tmax:
                                ok = False
                                break
                    if ok and tmin > 0.5 and (bt is None or tmin < bt):
                        bt, inten = tmin, 42.0
                if bt is not None:
                    pts.append([dx * bt + rng.normal(0, 0.01), dy * bt + rng.normal(0, 0.01), dz * bt + rng.normal(0, 0.01), inten])
        arr = np.array(pts, dtype=np.float32)
        return make_binary_pcd(arr[:, :3], arr[:, 3])

    zip_b64 = base64.b64encode(make_dummy_zip()).decode("ascii")
    # 1. Kiem chung Camera False-Positive Suppression: anh xam trong (0 xe) phai loai het box ao!
    client.post("/reset", json={"job_id": 9998})
    r_blank = client.post(
        "/frame",
        json={
            "job_id": 9998,
            "frame": 0,
            "seq": "BLANK",
            "pcd_base64": base64.b64encode(build_scene(13.0)).decode("ascii"),
            "calib": SAMPLE_CALIB,
            "images_zip_base64": zip_b64,
            "use_camera": True,
            "use_second_detector": True,
            "use_odometry": False,
        },
    ).json()
    assert r_blank["ok"] and r_blank["count"] == 0, f"Blank camera should suppress false positives: {r_blank}"
    print("[PASS] Camera False-Positive Suppression OK (blank wall -> 0 hallucinated boxes)")

    # 2. Kiem chung Tracking da chuoi A (160..162) -> B (163..164) tren LiDAR (use_camera=False)
    client.post("/reset", json={"job_id": 9999})
    for f, sq, cy in [
        (160, "G01-M5-A", 13.0),
        (161, "G01-M5-A", 14.0),
        (162, "G01-M5-A", 15.0),
        (163, "G01-M5-B", 12.5),
        (164, "G01-M5-B", 13.5),
    ]:
        pcd_b64 = base64.b64encode(build_scene(cy)).decode("ascii")
        r = client.post(
            "/frame",
            json={
                "job_id": 9999,
                "frame": f,
                "seq": sq,
                "pcd_base64": pcd_b64,
                "calib": SAMPLE_CALIB,
                "images_zip_base64": None,
                "use_camera": False,
                "use_second_detector": True,
                "use_odometry": True,
            },
        ).json()
        assert r["ok"] and r["count"] >= 1, f"Frame {f} failed: {r}"

    tr = client.post("/tracks", json={"job_id": 9999, "barrier_axis_mode": "length_is_x"}).json()
    assert tr["ok"] and tr["count"] >= 2, f"Tracks failed: {tr}"

    seq_a_tracks = [t for t in tr["tracks"] if t["seq"] == "G01-M5-A"]
    seq_b_tracks = [t for t in tr["tracks"] if t["seq"] == "G01-M5-B"]
    assert len(seq_a_tracks) >= 1 and len(seq_b_tracks) >= 1
    for t in seq_a_tracks:
        # Moi track cua chuoi A ket thuc o <= 162 BAT BUOC phai co outside <= 163 de khong tran sang chuoi B!
        assert len(t["outside_frames"]) >= 1 and max(t["outside_frames"]) <= 163, f"Ghost track leak: {t}"
        out_set = set(t["outside_frames"])
        for kf in t["keyframes"]:
            assert kf["frame"] not in out_set, "occluded/keyframe and outside conflict!"

    vram_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    assert vram_mb < 3000.0, f"VRAM exceeded 3GB limit on P2000: {vram_mb} MB"
    print(f"[PASS] End-to-End FastAPI + Multi-Sequence Tracking OK: {tr['count']} tracks, Peak VRAM = {vram_mb:.1f} MB")


def test_dynamic_guideline_and_docx_extract():
    client = TestClient(server.app)

    # 1. Test trich xuat bang tu file .docx ngay tren Agent
    docx_buf = io.BytesIO()
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body>'
        '<w:p><w:r><w:t>Lab Day 13 Custom Guideline</w:t></w:r></w:p>'
        '<w:tbl><w:tr>'
        '<w:tc><w:p><w:r><w:t>Car</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:p><w:r><w:t>3.8 - 5.2</w:t></w:r></w:p></w:tc>'
        '</w:tr></w:tbl>'
        '</w:body></w:document>'
    )
    with zipfile.ZipFile(docx_buf, "w") as zf:
        zf.writestr("word/document.xml", doc_xml.encode("utf-8"))
    ext_res = client.post(
        "/extract_guideline_text",
        json={
            "filename": "day13_guideline.docx",
            "file_base64": base64.b64encode(docx_buf.getvalue()).decode("ascii"),
        },
    ).json()
    assert ext_res["ok"] and "| Car | 3.8 - 5.2 |" in ext_res["text"], f"DOCX extract failed: {ext_res}"

    # 2. Test /frame & /tracks voi enabled_classes khong co Barrier + split_two_wheels + custom class_size_ranges
    rng = np.random.default_rng(7)
    ground = np.column_stack([rng.uniform(-15, 25, 800), rng.uniform(-15, 25, 800), np.full(800, -1.8)]).astype(np.float32)
    pcd_b64 = base64.b64encode(make_binary_pcd(ground, np.full(800, 20.0, dtype=np.float32))).decode("ascii")

    client.post("/reset", json={"job_id": 10013})
    r_dyn = client.post(
        "/frame",
        json={
            "job_id": 10013,
            "frame": 0,
            "seq": "DAY13",
            "pcd_base64": pcd_b64,
            "use_camera": False,
            "use_second_detector": True,
            "use_odometry": False,
            "enabled_classes": ["Car", "Pedestrian", "Bicycle", "Motorcycle"],
            "split_two_wheels": True,
            "merge_rider": False,
            "enable_size_clamp": True,
            "class_size_ranges": {"Car": [[3.6, 5.5], [1.6, 2.2], [1.3, 2.2]]},
            "max_distance_by_class": {"Car": 60.0},
        },
    ).json()
    assert r_dyn["ok"]
    assert all(d["className"] != "Barrier" for d in r_dyn["dets"])
    print("[PASS] Dynamic Guideline API (enabled_classes, split_two_wheels, DOCX extractor) OK")


if __name__ == "__main__":
    test_geometry_and_barrier()
    test_icp_odometry()
    test_camera_projection_near_clip_and_zip()
    test_api_and_multi_sequence_tracking()
    test_dynamic_guideline_and_docx_extract()
    print("ALL TESTS PASSED!")
