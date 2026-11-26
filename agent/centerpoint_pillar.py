"""
Pure-PyTorch CenterPoint-Pillar (nuScenes) Inference Engine
Khong can mmcv / mmdet3d / spconv / bien dich CUDA ops tren Windows!
Nap truc tiep 100% trong so (strict=True) tu checkpoint chinh thuc cua OpenMMLab:
  centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus

Toi uu cho Quadro P2000 (Pascal sm_61, 4GB VRAM):
  - Chay FP32 on dinh, su dung ~0.8 - 1.2 GB VRAM
  - Du doan 10 class nuScenes -> map chuan ve 8 class Guideline V1
  - Du doan du kich thuoc that (L, W, H) va goc quay 360° (yaw) ke ca khi xe bi che khuat mot phan!
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Cau hinh chuan cua centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus
PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
VOXEL_SIZE = (0.2, 0.2, 8.0)
GRID_SIZE = (512, 512)  # (W_x, H_y)
OUT_SIZE_FACTOR = 4     # 512 / 4 = 128
MAX_POINTS_PER_PILLAR = 20

TASKS = [
    ["car"],
    ["truck", "construction_vehicle"],
    ["bus", "trailer"],
    ["barrier"],
    ["motorcycle", "bicycle"],
    ["pedestrian", "traffic_cone"],
]

# Ban kinh CircleNMS (met) cho tung nhom task (tinh chinh de khong gop nham xe do sat nhau)
TASK_NMS_RADIUS = [1.8, 2.5, 3.0, 0.7, 0.65, 0.4]

NUSCENES_TO_GUIDELINE = {
    "car": "Car",
    "truck": "Truck",
    "construction_vehicle": "Construction_vehicle",
    "bus": "Bus",
    "trailer": "Trailer",
    "barrier": "Barrier",
    "motorcycle": "Two_wheels",
    "bicycle": "Two_wheels",
    "pedestrian": "Pedestrian",
    "traffic_cone": "Traffic_cone",
}


class PFNLayer(nn.Module):
    def __init__(self, in_channels: int = 11, out_channels: int = 64):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (N_pts, 11)
        x = self.linear(features)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        return x


class PillarFeatureNet(nn.Module):
    def __init__(self, in_channels: int = 11, feat_channels: int = 64):
        super().__init__()
        self.pfn_layers = nn.ModuleList([PFNLayer(in_channels, feat_channels)])


class SECOND(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: Tuple[int, ...] = (64, 128, 256),
        layer_nums: Tuple[int, ...] = (3, 5, 5),
        layer_strides: Tuple[int, ...] = (2, 2, 2),
    ):
        super().__init__()
        blocks = []
        in_filters = [in_channels, *out_channels[:-1]]
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


class SECONDFPN(nn.Module):
    def __init__(self):
        super().__init__()
        # upsample_strides = [0.5, 1, 2], in_channels = [64, 128, 256], out_channels = [128, 128, 128]
        self.deblocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ),
            # QUAN TRONG: use_conv_for_no_stride=True -> deblocks.1 la Conv2d (KHONG phai ConvTranspose2d,
            # neu dung ConvTranspose2d se bi chuyen vi ma tran trong so 128x128 lam nhieu feature map!)
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ),
        ])

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        ups = [deblock(x) for deblock, x in zip(self.deblocks, xs)]
        return torch.cat(ups, dim=1)


class ConvModule(nn.Module):
    """Khop chinh xac ten key '.conv' va '.bn' cua mmcv ConvModule trong checkpoint."""
    def __init__(self, in_c: int, out_c: int, k: int = 3, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class SeparateHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        heads = {
            "reg": 2,
            "height": 1,
            "dim": 3,
            "rot": 2,
            "vel": 2,
            "heatmap": num_classes,
        }
        for name, out_c in heads.items():
            seq = nn.Sequential(
                ConvModule(in_channels, in_channels, k=3, p=1),
                nn.Conv2d(in_channels, out_c, kernel_size=3, padding=1, bias=True),
            )
            setattr(self, name, seq)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "reg": self.reg(x),
            "height": self.height(x),
            "dim": self.dim(x),
            "rot": self.rot(x),
            "vel": self.vel(x),
            "heatmap": self.heatmap(x),
        }


class CenterHead(nn.Module):
    def __init__(self, in_channels: int = 384, share_conv_channel: int = 64):
        super().__init__()
        self.shared_conv = ConvModule(in_channels, share_conv_channel, k=3, p=1)
        self.task_heads = nn.ModuleList([
            SeparateHead(share_conv_channel, len(t)) for t in TASKS
        ])

    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        x = self.shared_conv(x)
        return [head(x) for head in self.task_heads]


class CenterPointPillarNet(nn.Module):
    """Mo hinh CenterPoint-Pillar thuan PyTorch khop 100% state_dict OpenMMLab."""

    def __init__(self):
        super().__init__()
        self.pts_voxel_encoder = PillarFeatureNet(in_channels=11, feat_channels=64)
        self.pts_backbone = SECOND(in_channels=64, out_channels=(64, 128, 256), layer_nums=(3, 5, 5), layer_strides=(2, 2, 2))
        self.pts_neck = SECONDFPN()
        self.pts_bbox_head = CenterHead(in_channels=384, share_conv_channel=64)

    @torch.no_grad()
    def voxelize_and_encode(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Dynamic Pillarization thuan PyTorch (khong can custom CUDA ops).
        pts: (N, 5) float32 [x, y, z, intensity, timestamp]
        Tra ve BEV feature map: (1, 64, 512, 512)
        """
        device = pts.device
        xmin, ymin, zmin, xmax, ymax, zmax = PC_RANGE
        vx, vy, vz = VOXEL_SIZE
        W, H = GRID_SIZE

        mask = (
            (pts[:, 0] >= xmin) & (pts[:, 0] < xmax) &
            (pts[:, 1] >= ymin) & (pts[:, 1] < ymax) &
            (pts[:, 2] >= zmin) & (pts[:, 2] < zmax)
        )
        pts = pts[mask]
        canvas = torch.zeros((1, 64, H, W), dtype=torch.float32, device=device)
        if pts.shape[0] == 0:
            return canvas

        x_idx = torch.floor((pts[:, 0] - xmin) / vx).long().clamp(0, W - 1)
        y_idx = torch.floor((pts[:, 1] - ymin) / vy).long().clamp(0, H - 1)
        pillar_id = y_idx * W + x_idx

        unique_ids, inverse = torch.unique(pillar_id, return_inverse=True)
        num_pillars = unique_ids.shape[0]

        # Tinh tam cum (x_mean, y_mean, z_mean) cho moi pillar
        counts = torch.zeros((num_pillars, 1), dtype=torch.float32, device=device)
        counts.index_add_(0, inverse, torch.ones((pts.shape[0], 1), dtype=torch.float32, device=device))

        sums = torch.zeros((num_pillars, 3), dtype=torch.float32, device=device)
        sums.index_add_(0, inverse, pts[:, :3])
        means = sums / counts.clamp(min=1.0)

        f_cluster = pts[:, :3] - means[inverse]

        # Tinh lech so voi tam hinh hoc cua pillar (x_c, y_c, z_c)
        x_center = x_idx.float() * vx + (xmin + vx * 0.5)
        y_center = y_idx.float() * vy + (ymin + vy * 0.5)
        z_center = torch.full_like(x_center, (zmin + zmax) * 0.5)
        f_center = torch.stack([
            pts[:, 0] - x_center,
            pts[:, 1] - y_center,
            pts[:, 2] - z_center,
        ], dim=1)

        # Ghep du 11 chieu: [x, y, z, intensity, timestamp, dx_c, dy_c, dz_c, dx_p, dy_p, dz_p]
        feats = torch.cat([pts, f_cluster, f_center], dim=1)
        pfn = self.pts_voxel_encoder.pfn_layers[0]
        pt_feats = pfn(feats)  # (N_pts, 64)

        # HardVoxelization (max_num_points=20): pillar co < 20 diem duoc zero-pad truoc PFNLayer ->
        # cac kenh co ReLU(BN(0)) > 0 phai giu san z0 khi lay max theo pillar (khop chinh xac mmdet3d!)
        z0 = pfn(torch.zeros((1, 11), dtype=torch.float32, device=device))
        has_pad = counts.squeeze(1) < MAX_POINTS_PER_PILLAR
        pillar_feats = torch.where(
            has_pad.unsqueeze(1),
            z0.expand(num_pillars, 64),
            torch.full((num_pillars, 64), -1e9, dtype=torch.float32, device=device),
        )
        idx_exp = inverse.unsqueeze(1).expand(-1, 64)
        pillar_feats.scatter_reduce_(0, idx_exp, pt_feats, reduce="amax", include_self=True)

        # Dat len luoi BEV (1, 64, H, W)
        py = torch.div(unique_ids, W, rounding_mode="floor")
        px = unique_ids % W
        canvas[0, :, py, px] = pillar_feats.t()
        return canvas

    @torch.no_grad()
    def forward_points(self, pts: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        bev = self.voxelize_and_encode(pts)
        feats = self.pts_backbone(bev)
        neck_feat = self.pts_neck(feats)
        return self.pts_bbox_head(neck_feat)


def _circle_nms_np(centers_xy: np.ndarray, scores: np.ndarray, min_radius: float) -> List[int]:
    order = np.argsort(-scores)
    keep = []
    thresh2 = min_radius * min_radius
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        dx = centers_xy[rest, 0] - centers_xy[i, 0]
        dy = centers_xy[rest, 1] - centers_xy[i, 1]
        dist2 = dx * dx + dy * dy
        order = rest[dist2 > thresh2]
    return keep


class CenterPointInferenceEngine:
    def __init__(self, checkpoint_path: str, prefer_cuda: bool = True):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda:0" if (prefer_cuda and torch.cuda.is_available()) else "cpu")
        self.model = CenterPointPillarNet()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        xyz: np.ndarray,
        intensity: Optional[np.ndarray] = None,
        score_threshold: float = 0.16,
        max_per_task: int = 200,
        split_two_wheels: bool = False,
    ) -> List[Dict[str, Any]]:
        if xyz.shape[0] == 0:
            return []

        n = xyz.shape[0]
        if intensity is None:
            inten = np.zeros((n, 1), dtype=np.float32)
        else:
            inten = intensity.reshape(n, 1).astype(np.float32)
            # Chuan hoa intensity ve dai 0..255 giong nuScenes neu dang la 0..1
            if inten.max() <= 1.05 and inten.max() > 0:
                inten = inten * 255.0

        ts = np.zeros((n, 1), dtype=np.float32)
        pts_np = np.hstack([xyz.astype(np.float32), inten, ts])
        pts_t = torch.from_numpy(pts_np).to(self.device)

        task_outs = self.model.forward_points(pts_t)

        xmin, ymin, _, _, _, _ = PC_RANGE
        vx, vy, _ = VOXEL_SIZE
        detections = []

        for task_idx, out in enumerate(task_outs):
            cls_names = TASKS[task_idx]
            hm = torch.sigmoid(out["heatmap"][0])  # (C, 128, 128)
            # Theo test_cfg cua checkpoint: max_pool_nms=False, loc truc tiep top-K roi CircleNMS
            C, H_out, W_out = hm.shape
            scores_flat = hm.reshape(-1)
            k = min(max_per_task, scores_flat.numel())
            top_scores, top_inds = torch.topk(scores_flat, k=k)

            keep_mask = top_scores >= score_threshold
            if not keep_mask.any():
                continue
            top_scores = top_scores[keep_mask]
            top_inds = top_inds[keep_mask]

            cls_ids = torch.div(top_inds, H_out * W_out, rounding_mode="floor")
            rem = top_inds % (H_out * W_out)
            ys = torch.div(rem, W_out, rounding_mode="floor")
            xs = rem % W_out

            reg = out["reg"][0]       # (2, H, W)
            hei = out["height"][0]    # (1, H, W)
            dim = out["dim"][0]       # (3, H, W)
            rot = out["rot"][0]       # (2, H, W)

            rx = reg[0, ys, xs]
            ry = reg[1, ys, xs]
            cx = (xs.float() + rx) * OUT_SIZE_FACTOR * vx + xmin
            cy = (ys.float() + ry) * OUT_SIZE_FACTOR * vy + ymin
            cz = hei[0, ys, xs]

            dims = torch.exp(dim[:, ys, xs]).t()  # (M, 3): [dx, dy, dz]
            rot_sin = rot[0, ys, xs]
            rot_cos = rot[1, ys, xs]
            yaw = torch.atan2(rot_sin, rot_cos)

            cx_np = cx.cpu().numpy()
            cy_np = cy.cpu().numpy()
            cz_np = cz.cpu().numpy()
            dims_np = dims.cpu().numpy()
            yaw_np = yaw.cpu().numpy()
            sc_np = top_scores.cpu().numpy()
            cid_np = cls_ids.cpu().numpy()

            centers = np.stack([cx_np, cy_np], axis=1)
            keep_nms = _circle_nms_np(centers, sc_np, TASK_NMS_RADIUS[task_idx])

            for idx in keep_nms:
                raw_cls = cls_names[int(cid_np[idx])]
                if split_two_wheels and raw_cls in ("bicycle", "motorcycle"):
                    mapped = "Bicycle" if raw_cls == "bicycle" else "Motorcycle"
                else:
                    mapped = NUSCENES_TO_GUIDELINE.get(raw_cls)
                if not mapped:
                    continue
                # mmdet3d 1.0.0rc3 dung he LiDARInstance3DBoxes chuan (l, w, h, yaw CCW voi 0 = +X) — khop CVAT 3D
                yaw_cvat = float(yaw_np[idx])
                while yaw_cvat > math.pi:
                    yaw_cvat -= 2.0 * math.pi
                while yaw_cvat < -math.pi:
                    yaw_cvat += 2.0 * math.pi

                detections.append({
                    "className": mapped,
                    "rawClass": raw_cls,
                    "score": float(sc_np[idx]),
                    "x": float(cx_np[idx]),
                    "y": float(cy_np[idx]),
                    "z": float(cz_np[idx]),
                    "rz": yaw_cvat,
                    "dx": float(dims_np[idx, 0]),
                    "dy": float(dims_np[idx, 1]),
                    "dz": float(dims_np[idx, 2]),
                    "pointCount": 0,
                })

        return detections
