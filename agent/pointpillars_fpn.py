"""
Pure-PyTorch PointPillars-FPN (nuScenes) Inference Engine — Detector phu doc lap (Model B).
Nap truc tiep 100% state_dict (strict=True) tu checkpoint chinh thuc OpenMMLab:
  hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d (mmdet3d 0.16.0)

Dac diem doc lap voi CenterPoint-Pillar (Model A):
  - Luoi Pillar 0.25m x 0.25m (400x400) + HardVFE 2 tang (time_lag=0.0, zero-pad floor chinh xac)
  - Neck FPN da ty le (200x200, 100x100, 50x50, scales 1/2/4)
  - Anchor-based Anchor3DHead + Direction Classifier (khac han heatmap CenterHead)
  - Chay thuan PyTorch FP32 tren Quadro P2000 (~200 MB VRAM, khong can mmcv/mmdet3d/spconv)
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PC_RANGE = (-50.0, -50.0, -5.0, 50.0, 50.0, 3.0)
VOXEL_SIZE = (0.25, 0.25, 8.0)
GRID_SIZE = (400, 400)
MAX_POINTS_PER_PILLAR = 64

CLASS_NAMES = [
    "car", "truck", "trailer", "bus", "construction_vehicle",
    "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier",
]

NUSCENES_TO_GUIDELINE = {
    "car": "Car",
    "truck": "Truck",
    "trailer": "Trailer",
    "bus": "Bus",
    "construction_vehicle": "Construction_vehicle",
    "bicycle": "Two_wheels",
    "motorcycle": "Two_wheels",
    "pedestrian": "Pedestrian",
    "barrier": "Barrier",
    "traffic_cone": "Traffic_cone",
}

ANCHOR_SIZES = [
    [2.5981, 0.866, 1.0],
    [1.7321, 0.5774, 1.0],
    [1.0, 1.0, 1.0],
    [0.4, 0.4, 1.0],
]
ANCHOR_ROTS = [0.0, 1.57]
LEVEL_SCALES = [1.0, 2.0, 4.0]
DIR_OFFSET = -0.7854


def _wrap_pi(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


class VFELayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.linear(x)), inplace=True)


class HardVFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe_layers = nn.ModuleList([
            VFELayer(10, 64),
            VFELayer(128, 64),
        ])


class SECOND(nn.Module):
    def __init__(self):
        super().__init__()
        in_filters = [64, 64, 128]
        out_channels = [64, 128, 256]
        layer_nums = [3, 5, 5]
        layer_strides = [2, 2, 2]
        blocks = []
        for idx, num in enumerate(layer_nums):
            layers = [
                nn.Conv2d(in_filters[idx], out_channels[idx], 3, stride=layer_strides[idx], padding=1, bias=False),
                nn.BatchNorm2d(out_channels[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            for _ in range(num):
                layers.extend([
                    nn.Conv2d(out_channels[idx], out_channels[idx], 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                ])
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        return outs


class ConvBNReLU(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int, p: int):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class FPNNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            ConvBNReLU(64, 256, k=1, p=0),
            ConvBNReLU(128, 256, k=1, p=0),
            ConvBNReLU(256, 256, k=1, p=0),
        ])
        self.fpn_convs = nn.ModuleList([
            ConvBNReLU(256, 256, k=3, p=1),
            ConvBNReLU(256, 256, k=3, p=1),
            ConvBNReLU(256, 256, k=3, p=1),
        ])

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="nearest"
            )
        return [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]


class Anchor3DHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_cls = nn.Conv2d(256, 80, kernel_size=1, bias=True)
        self.conv_reg = nn.Conv2d(256, 72, kernel_size=1, bias=True)
        self.conv_dir_cls = nn.Conv2d(256, 16, kernel_size=1, bias=True)

    def forward(self, feats: List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return [(self.conv_cls(f), self.conv_reg(f), self.conv_dir_cls(f)) for f in feats]


class PointPillarsFPNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pts_voxel_encoder = HardVFE()
        self.pts_backbone = SECOND()
        self.pts_neck = FPNNeck()
        self.pts_bbox_head = Anchor3DHead()

    @torch.no_grad()
    def voxelize_and_encode(self, pts4: torch.Tensor) -> torch.Tensor:
        """
        HardVFE khop chinh xac mmdet3d v0.16.0:
          - pts4: [x, y, z, time_lag] (trong nus-3d default use_dim=[0,1,2,4], channel 3 la time_lag = 0.0)
          - Xu ly chinh xac gia tri zero-padding tren cac pillar co < 64 diem.
        """
        device = pts4.device
        xmin, ymin, zmin, xmax, ymax, zmax = PC_RANGE
        vx, vy, _ = VOXEL_SIZE
        W, H = GRID_SIZE

        mask = (
            (pts4[:, 0] >= xmin) & (pts4[:, 0] < xmax) &
            (pts4[:, 1] >= ymin) & (pts4[:, 1] < ymax) &
            (pts4[:, 2] >= zmin) & (pts4[:, 2] < zmax)
        )
        pts = pts4[mask]
        canvas = torch.zeros((1, 64, H, W), dtype=torch.float32, device=device)
        if pts.shape[0] == 0:
            return canvas

        x_idx = torch.floor((pts[:, 0] - xmin) / vx).long().clamp(0, W - 1)
        y_idx = torch.floor((pts[:, 1] - ymin) / vy).long().clamp(0, H - 1)
        pillar_id = y_idx * W + x_idx
        unique_ids, inverse = torch.unique(pillar_id, return_inverse=True)
        num_pillars = unique_ids.shape[0]

        counts = torch.zeros((num_pillars, 1), dtype=torch.float32, device=device)
        counts.index_add_(0, inverse, torch.ones((pts.shape[0], 1), dtype=torch.float32, device=device))
        sums = torch.zeros((num_pillars, 3), dtype=torch.float32, device=device)
        sums.index_add_(0, inverse, pts[:, :3])
        means = sums / counts.clamp(min=1.0)
        f_cluster = pts[:, :3] - means[inverse]

        x_c = x_idx.float() * vx + (xmin + vx * 0.5)
        y_c = y_idx.float() * vy + (ymin + vy * 0.5)
        z_c = torch.full_like(x_c, (zmin + zmax) * 0.5)
        f_center = torch.stack([pts[:, 0] - x_c, pts[:, 1] - y_c, pts[:, 2] - z_c], dim=1)

        feats10 = torch.cat([pts, f_cluster, f_center], dim=1)
        l0 = self.pts_voxel_encoder.vfe_layers[0]
        l1 = self.pts_voxel_encoder.vfe_layers[1]

        z0 = l0(torch.zeros((1, 10), dtype=torch.float32, device=device))
        h0 = l0(feats10)
        has_pad = (counts.squeeze(1) < MAX_POINTS_PER_PILLAR)

        p_max0 = torch.where(has_pad.unsqueeze(1), z0.expand(num_pillars, 64), torch.full((num_pillars, 64), -1e9, device=device))
        p_max0.scatter_reduce_(0, inverse.unsqueeze(1).expand(-1, 64), h0, reduce="amax", include_self=True)

        h1_pad = l1(torch.cat([z0.expand(num_pillars, 64), p_max0], dim=1))
        h1 = l1(torch.cat([h0, p_max0[inverse]], dim=1))
        p_max1 = torch.where(has_pad.unsqueeze(1), h1_pad, torch.full((num_pillars, 64), -1e9, device=device))
        p_max1.scatter_reduce_(0, inverse.unsqueeze(1).expand(-1, 64), h1, reduce="amax", include_self=True)

        py = torch.div(unique_ids, W, rounding_mode="floor")
        px = unique_ids % W
        canvas[0, :, py, px] = p_max1.t()
        return canvas

    @torch.no_grad()
    def forward_points(self, pts4: torch.Tensor):
        bev = self.voxelize_and_encode(pts4)
        feats = self.pts_backbone(bev)
        neck = self.pts_neck(feats)
        return self.pts_bbox_head(neck)


def _circle_nms_per_class(dets: List[Dict[str, Any]], radii: Dict[str, float]) -> List[Dict[str, Any]]:
    by_cls: Dict[str, List[Dict[str, Any]]] = {}
    for d in dets:
        by_cls.setdefault(d["className"], []).append(d)
    kept: List[Dict[str, Any]] = []
    for cls, items in by_cls.items():
        items.sort(key=lambda x: -x["score"])
        r2 = radii.get(cls, 1.5) ** 2
        sel: List[Dict[str, Any]] = []
        for it in items:
            if all((it["x"] - s["x"]) ** 2 + (it["y"] - s["y"]) ** 2 > r2 for s in sel):
                sel.append(it)
        kept.extend(sel)
    kept.sort(key=lambda x: -x["score"])
    return kept


class PointPillarsFPNEngine:
    def __init__(self, checkpoint_path: str, prefer_cuda: bool = True):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda:0" if (prefer_cuda and torch.cuda.is_available()) else "cpu")
        self.model = PointPillarsFPNNet()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device)
        self.model.eval()
        self._anchors_cache: Dict[Tuple[int, int, float], torch.Tensor] = {}

    def _level_anchors(self, H: int, W: int, scale: float) -> torch.Tensor:
        key = (H, W, scale)
        if key in self._anchors_cache:
            return self._anchors_cache[key]
        xmin, ymin, _, xmax, ymax, _ = PC_RANGE
        step_x = (xmax - xmin) / W
        step_y = (ymax - ymin) / H
        xs = torch.linspace(xmin + step_x * 0.5, xmax - step_x * 0.5, W, device=self.device)
        ys = torch.linspace(ymin + step_y * 0.5, ymax - step_y * 0.5, H, device=self.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        sizes = torch.tensor(ANCHOR_SIZES, dtype=torch.float32, device=self.device) * scale
        rots = torch.tensor(ANCHOR_ROTS, dtype=torch.float32, device=self.device)
        anc = torch.zeros((H, W, 4, 2, 7), dtype=torch.float32, device=self.device)
        anc[..., 0] = xx[:, :, None, None]
        anc[..., 1] = yy[:, :, None, None]
        anc[..., 2] = -1.8
        anc[..., 3] = sizes[None, None, :, None, 0]
        anc[..., 4] = sizes[None, None, :, None, 1]
        anc[..., 5] = sizes[None, None, :, None, 2]
        anc[..., 6] = rots[None, None, None, :]
        flat = anc.reshape(-1, 7)
        self._anchors_cache[key] = flat
        return flat

    @torch.no_grad()
    def predict(
        self,
        xyz: np.ndarray,
        time_lag: Optional[np.ndarray] = None,
        score_threshold: float = 0.22,
        nms_pre: int = 600,
        split_two_wheels: bool = False,
    ) -> List[Dict[str, Any]]:
        if xyz.shape[0] == 0:
            return []
        n = xyz.shape[0]
        tl = np.zeros((n, 1), dtype=np.float32) if time_lag is None else time_lag.reshape(n, 1).astype(np.float32)
        pts4 = torch.from_numpy(np.hstack([xyz.astype(np.float32), tl])).to(self.device)
        level_outs = self.model.forward_points(pts4)

        raw_dets: List[Dict[str, Any]] = []
        for lvl_idx, (cls_score, bbox_pred, dir_cls) in enumerate(level_outs):
            _, _, H, W = cls_score.shape
            anchors = self._level_anchors(H, W, LEVEL_SCALES[lvl_idx])

            scores = torch.sigmoid(cls_score[0].permute(1, 2, 0).reshape(-1, 10))
            deltas = bbox_pred[0].permute(1, 2, 0).reshape(-1, 9)
            dirs = dir_cls[0].permute(1, 2, 0).reshape(-1, 2)

            max_scores, cls_ids = scores.max(dim=1)
            keep = max_scores >= score_threshold
            if not keep.any():
                continue
            idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
            if idx.numel() > nms_pre:
                _, topk_i = torch.topk(max_scores[idx], k=nms_pre)
                idx = idx[topk_i]

            sc = max_scores[idx]
            cid = cls_ids[idx]
            a = anchors[idx]
            d = deltas[idx]
            dir_label = dirs[idx].argmax(dim=1)

            diag = torch.sqrt(a[:, 3] ** 2 + a[:, 4] ** 2)
            xg = d[:, 0] * diag + a[:, 0]
            yg = d[:, 1] * diag + a[:, 1]
            wg = torch.exp(d[:, 3].clamp(-4.0, 4.0)) * a[:, 3]
            lg = torch.exp(d[:, 4].clamp(-4.0, 4.0)) * a[:, 4]
            hg = torch.exp(d[:, 5].clamp(-4.0, 4.0)) * a[:, 5]
            zg_center = d[:, 2] * a[:, 5] + a[:, 2] + a[:, 5] * 0.5
            rg = d[:, 6] + a[:, 6]

            dir_rot = torch.remainder(rg - DIR_OFFSET, math.pi)
            rg = dir_rot + DIR_OFFSET + math.pi * dir_label.float()

            xg_np = xg.cpu().numpy()
            yg_np = yg.cpu().numpy()
            zg_np = zg_center.cpu().numpy()
            wg_np = wg.cpu().numpy()
            lg_np = lg.cpu().numpy()
            hg_np = hg.cpu().numpy()
            rg_np = rg.cpu().numpy()
            sc_np = sc.cpu().numpy()
            cid_np = cid.cpu().numpy()

            for k in range(len(sc_np)):
                raw_cls = CLASS_NAMES[int(cid_np[k])]
                if split_two_wheels and raw_cls in ("bicycle", "motorcycle"):
                    mapped = "Bicycle" if raw_cls == "bicycle" else "Motorcycle"
                else:
                    mapped = NUSCENES_TO_GUIDELINE.get(raw_cls)
                if not mapped:
                    continue
                raw_dets.append({
                    "className": mapped,
                    "rawClass": raw_cls,
                    "score": float(sc_np[k]),
                    "x": float(xg_np[k]),
                    "y": float(yg_np[k]),
                    "z": float(zg_np[k]),
                    "rz": _wrap_pi(float(rg_np[k])),
                    "dx": float(wg_np[k]),
                    "dy": float(lg_np[k]),
                    "dz": float(hg_np[k]),
                    "pointCount": 0,
                })

        radii = {
            "Car": 1.8, "Truck": 2.4, "Bus": 2.8, "Trailer": 2.8,
            "Construction_vehicle": 2.4, "Barrier": 0.7, "Two_wheels": 0.65,
            "Bicycle": 0.6, "Motorcycle": 0.65, "Pedestrian": 0.45,
            "Traffic_cone": 0.4,
        }
        return _circle_nms_per_class(raw_dets, radii)
