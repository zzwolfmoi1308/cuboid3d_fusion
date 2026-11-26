/**
 * Bộ 5 Kiểm Tra Point Cloud (V1.5 — 100% Chỉ Đọc, Đa Guideline / Đa Profile)
 * Mọi công tắc kiểm tra (PCD1..PCD5) và bán kính quét đều đọc từ cấu hình hiệu lực của Profile hiện hành.
 */

import { DEFAULT_RULE_TOGGLES, DEFAULT_SETTINGS } from "./guideline_config.js";
import { parseCuboidPoints, percentile } from "./rules_metadata.js";

function toBoxLocal(px, py, pz, box) {
  const dx = px - box.x;
  const dy = py - box.y;
  const dz = pz - box.z;
  const cos = Math.cos(-box.rz);
  const sin = Math.sin(-box.rz);
  return {
    lx: dx * cos - dy * sin,
    ly: dx * sin + dy * cos,
    lz: dz
  };
}

export function runPcdFrameQA(xyz, frameBoxes, userSettings = {}) {
  const settings = { ...DEFAULT_SETTINGS, ...userSettings };
  const rules = {
    ...DEFAULT_RULE_TOGGLES,
    ...(settings.rules || {})
  };
  const nPts = Math.floor((xyz?.length || 0) / 3);
  const findings = [];
  const boxMetrics = [];
  const activeBoxes = [];

  let counter = 0;
  const pushFinding = (f) => {
    findings.push({ id: `pcd_${++counter}`, ...f });
  };

  const insideAnyBox = new Uint8Array(nPts);

  for (const b of frameBoxes) {
    if (b.outside) continue;
    const p = parseCuboidPoints(b.points);
    if (!p) continue;
    activeBoxes.push({ ...b, parsed: p });
  }

  for (const item of activeBoxes) {
    const box = item.parsed;
    const halfX = box.dx / 2;
    const halfY = box.dy / 2;
    const halfZ = box.dz / 2;
    const boxRadius = Math.hypot(halfX, halfY);
    const innerRing = boxRadius + 0.5;
    const outerRing = boxRadius + 3.0;
    const boxBottomZ = box.z - halfZ;

    let insideCount = 0;
    let minLx = Infinity, maxLx = -Infinity;
    let minLy = Infinity, maxLy = -Infinity;
    let minLz = Infinity, maxLz = -Infinity;
    const annulusZ = [];
    const occupiedVoxels = new Set();

    const vxBins = 4;
    const vyBins = 4;
    const vzBins = 3;

    for (let i = 0; i < nPts; i++) {
      const px = xyz[i * 3];
      const py = xyz[i * 3 + 1];
      const pz = xyz[i * 3 + 2];

      const d2 = (px - box.x) * (px - box.x) + (py - box.y) * (py - box.y);
      if (d2 > outerRing * outerRing) continue;

      const loc = toBoxLocal(px, py, pz, box);
      const isInside =
        Math.abs(loc.lx) <= halfX &&
        Math.abs(loc.ly) <= halfY &&
        Math.abs(loc.lz) <= halfZ;

      if (isInside) {
        insideAnyBox[i] = 1;
        insideCount++;
        if (loc.lx < minLx) minLx = loc.lx;
        if (loc.lx > maxLx) maxLx = loc.lx;
        if (loc.ly < minLy) minLy = loc.ly;
        if (loc.ly > maxLy) maxLy = loc.ly;
        if (loc.lz < minLz) minLz = loc.lz;
        if (loc.lz > maxLz) maxLz = loc.lz;

        const bx = Math.min(vxBins - 1, Math.max(0, Math.floor(((loc.lx + halfX) / (box.dx || 1e-6)) * vxBins)));
        const by = Math.min(vyBins - 1, Math.max(0, Math.floor(((loc.ly + halfY) / (box.dy || 1e-6)) * vyBins)));
        const bz = Math.min(vzBins - 1, Math.max(0, Math.floor(((loc.lz + halfZ) / (box.dz || 1e-6)) * vzBins)));
        occupiedVoxels.add(`${bx}_${by}_${bz}`);
      } else {
        const distXY = Math.sqrt(d2);
        if (distXY >= innerRing && distXY <= outerRing && Math.abs(pz - boxBottomZ) <= 1.2) {
          annulusZ.push(pz);
        }
      }
    }

    // PCD 1: Lơ lửng / Chìm so với mặt đất cục bộ
    let zGround = null;
    let groundConfidence = "low";
    let groundDelta = null;

    if (annulusZ.length >= 12) {
      zGround = percentile(annulusZ, 0.15);
      const zSpread = percentile(annulusZ, 0.5) - zGround;
      groundConfidence = annulusZ.length >= 25 && zSpread < 0.35 ? "high" : "medium";
      groundDelta = boxBottomZ - zGround;

      if (rules.PCD1_GROUND && groundDelta > settings.pcdGroundFloatThresholdM) {
        pushFinding({
          ruleId: "PCD1_FLOATING",
          severity: "Warning",
          frame: item.frame,
          trackId: item.trackId,
          objectType: "track",
          className: item.className,
          title: `Box lơ lửng +${(groundDelta * 100).toFixed(0)} cm so với mặt đất`,
          message: `${item.className} #${item.trackId} tại frame ${item.frame}: đáy box z=${boxBottomZ.toFixed(2)}m, mặt đất cục bộ z_ground=${zGround.toFixed(2)}m (độ tin cậy: ${groundConfidence}).`,
          suggestion: "Hạ đáy box chạm mặt đường trên Side/Front view (hoặc dùng nút Snap to Ground khi V2 bật).",
          meta: { zGround, groundDelta, groundConfidence, canSnapGround: groundConfidence === "high" }
        });
      } else if (rules.PCD1_GROUND && groundDelta < -settings.pcdGroundSinkThresholdM) {
        pushFinding({
          ruleId: "PCD1_SINKING",
          severity: "Warning",
          frame: item.frame,
          trackId: item.trackId,
          objectType: "track",
          className: item.className,
          title: `Box chìm ${(Math.abs(groundDelta) * 100).toFixed(0)} cm dưới mặt đất`,
          message: `${item.className} #${item.trackId} tại frame ${item.frame}: đáy box z=${boxBottomZ.toFixed(2)}m thấp hơn mặt đất cục bộ z_ground=${zGround.toFixed(2)}m.`,
          suggestion: "Nâng đáy box lên đúng mặt đường trên Side/Front view.",
          meta: { zGround, groundDelta, groundConfidence, canSnapGround: groundConfidence === "high" }
        });
      }
    }

    // PCD 2: Box 0 điểm nhưng không bật occluded
    if (rules.PCD2_ZERO_POINTS && insideCount === 0 && !item.occluded) {
      pushFinding({
        ruleId: "PCD2_ZERO_POINTS_UNOCCLUDED",
        severity: "Warning",
        frame: item.frame,
        trackId: item.trackId,
        objectType: "track",
        className: item.className,
        title: "Box có 0 điểm LiDAR nhưng chưa bật occluded",
        message: `${item.className} #${item.trackId} tại frame ${item.frame} không chứa điểm LiDAR nào và occluded=false.`,
        suggestion: "Kiểm tra camera: nếu bị che tạm thời thì bật occluded=true; nếu đã rời scene thì switch outside=true."
      });
    }

    // PCD 3: Quá rộng (Empty Margin)
    if (rules.PCD3_EMPTY_MARGIN && insideCount >= 8) {
      const spanX = maxLx - minLx;
      const spanY = maxLy - minLy;
      const spanZ = maxLz - minLz;
      const marginX = box.dx - spanX;
      const marginY = box.dy - spanY;
      const marginZ = box.dz - spanZ;
      const thr = settings.pcdEmptyMarginThresholdM ?? 0.45;

      if (!item.occluded && (marginX > thr * 2 || marginY > thr * 2 || marginZ > thr * 1.5)) {
        pushFinding({
          ruleId: "PCD3_EMPTY_MARGIN",
          severity: "Info",
          frame: item.frame,
          trackId: item.trackId,
          objectType: "track",
          className: item.className,
          title: "Nghi ngờ box quá rộng so với cụm điểm",
          message: `${item.className} #${item.trackId} tại frame ${item.frame}: khoảng trống dư (ΔX=${marginX.toFixed(2)}m, ΔY=${marginY.toFixed(2)}m, ΔZ=${marginZ.toFixed(2)}m) với ${insideCount} điểm bên trong.`,
          suggestion: "Kiểm tra Top/Side/Front xem có bị dư biên quá mức không (bỏ qua nếu vật bị tự che một mặt)."
        });
      }
    }

    // PCD 4: Điểm thưa
    const voxelCoverage = occupiedVoxels.size / (vxBins * vyBins * vzBins);
    if (
      rules.PCD4_SPARSE_POINTS &&
      insideCount > 0 &&
      insideCount < 15 &&
      voxelCoverage < (settings.pcdMinSparseCoverage ?? 0.15) &&
      !item.occluded
    ) {
      pushFinding({
        ruleId: "PCD4_SPARSE_POINTS",
        severity: "Info",
        frame: item.frame,
        trackId: item.trackId,
        objectType: "track",
        className: item.className,
        title: `Điểm rất thưa (${insideCount} điểm, phủ ${(voxelCoverage * 100).toFixed(0)}%)`,
        message: `${item.className} #${item.trackId} tại frame ${item.frame} chỉ có ${insideCount} điểm LiDAR (độ phủ voxel thô ${(voxelCoverage * 100).toFixed(0)}% < 15%).`,
        suggestion: "Chỉ giữ box khi có đủ cơ sở nội suy từ frame lân cận hoặc tạo mẫu LOW_VISIBILITY.",
        meta: { insideCount, voxelCoverage, suggestedIssueType: "LOW_VISIBILITY" }
      });
    }

    boxMetrics.push({
      trackId: item.trackId,
      frame: item.frame,
      className: item.className,
      insideCount,
      voxelCoverage: Number(voxelCoverage.toFixed(3)),
      zGround: zGround != null ? Number(zGround.toFixed(3)) : null,
      groundConfidence
    });
  }

  // PCD 5: Gợi ý cụm điểm nghi bỏ sót (đọc bán kính từ Profile thay vì fix cứng 45m/50m)
  if (rules.PCD5_MISSED_HINTS) {
    const maxVehicleDist = settings.maxDistanceByGroup?.Vehicle || settings.pcdMaxHintDistanceM || 45;
    const hintRadius = settings.enableDistanceRule === false ? 55 : Math.min(60, Math.max(15, maxVehicleDist - 3));
    const missedHints = detectMissedObjectHints(
      xyz,
      nPts,
      insideAnyBox,
      frameBoxes[0]?.frame ?? 0,
      hintRadius
    );
    for (const hint of missedHints) {
      pushFinding(hint);
    }
  }

  return { findings, boxMetrics, totalPoints: nPts };
}

function detectMissedObjectHints(xyz, nPts, insideAnyBox, frame, maxClusterRadius = 45.0) {
  const groundCellSize = 4.0;
  const groundMinZ = new Map();
  const outerGroundR = maxClusterRadius + 5.0;

  for (let i = 0; i < nPts; i++) {
    const x = xyz[i * 3];
    const y = xyz[i * 3 + 1];
    const z = xyz[i * 3 + 2];
    const r2 = x * x + y * y;
    if (r2 < 2.5 * 2.5 || r2 > outerGroundR * outerGroundR) continue;
    const key = `${Math.floor(x / groundCellSize)}_${Math.floor(y / groundCellSize)}`;
    const prev = groundMinZ.get(key);
    if (prev === undefined || z < prev) {
      groundMinZ.set(key, z);
    }
  }

  const clusterCell = 1.2;
  const cells = new Map();
  for (let i = 0; i < nPts; i++) {
    if (insideAnyBox[i]) continue;
    const x = xyz[i * 3];
    const y = xyz[i * 3 + 1];
    const z = xyz[i * 3 + 2];
    const r2 = x * x + y * y;
    if (r2 < 3.0 * 3.0 || r2 > maxClusterRadius * maxClusterRadius) continue;

    const gKey = `${Math.floor(x / groundCellSize)}_${Math.floor(y / groundCellSize)}`;
    const zg = groundMinZ.get(gKey);
    if (zg === undefined) continue;

    const h = z - zg;
    if (h < 0.3 || h > 4.2) continue;

    const cx = Math.floor(x / clusterCell);
    const cy = Math.floor(y / clusterCell);
    const cKey = `${cx}_${cy}`;
    let cell = cells.get(cKey);
    if (!cell) {
      cell = { cx, cy, count: 0, sx: 0, sy: 0, sz: 0, minZ: z, maxZ: z, zg };
      cells.set(cKey, cell);
    }
    cell.count++;
    cell.sx += x;
    cell.sy += y;
    cell.sz += z;
    if (z < cell.minZ) cell.minZ = z;
    if (z > cell.maxZ) cell.maxZ = z;
  }

  const visited = new Set();
  const clusters = [];

  for (const [key, cell] of cells.entries()) {
    if (visited.has(key) || cell.count < 6) continue;
    const queue = [cell];
    visited.add(key);

    let totalPts = 0;
    let sx = 0, sy = 0, sz = 0;
    let minCx = cell.cx, maxCx = cell.cx;
    let minCy = cell.cy, maxCy = cell.cy;
    let maxH = 0;

    while (queue.length > 0) {
      const curr = queue.shift();
      totalPts += curr.count;
      sx += curr.sx;
      sy += curr.sy;
      sz += curr.sz;
      if (curr.cx < minCx) minCx = curr.cx;
      if (curr.cx > maxCx) maxCx = curr.cx;
      if (curr.cy < minCy) minCy = curr.cy;
      if (curr.cy > maxCy) maxCy = curr.cy;
      const ch = curr.maxZ - curr.zg;
      if (ch > maxH) maxH = ch;

      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          if (dx === 0 && dy === 0) continue;
          const nKey = `${curr.cx + dx}_${curr.cy + dy}`;
          if (!visited.has(nKey)) {
            const nb = cells.get(nKey);
            if (nb && nb.count >= 4) {
              visited.add(nKey);
              queue.push(nb);
            }
          }
        }
      }
    }

    const spanX = (maxCx - minCx + 1) * clusterCell;
    const spanY = (maxCy - minCy + 1) * clusterCell;
    const maxSpan = Math.max(spanX, spanY);

    if (totalPts >= 18 && maxSpan <= 16.0 && maxH >= 0.6 && maxH <= 4.0) {
      const mx = sx / totalPts;
      const my = sy / totalPts;
      const mz = sz / totalPts;
      clusters.push({
        x: mx,
        y: my,
        z: mz,
        dist: Math.hypot(mx, my),
        totalPts,
        maxH
      });
    }
  }

  clusters.sort((a, b) => b.totalPts - a.totalPts);
  return clusters.slice(0, 8).map((c, idx) => ({
    ruleId: "PCD5_MISSED_OBJECT_HINT",
    severity: "Info",
    frame,
    trackId: null,
    objectType: "hint",
    className: "Unknown",
    title: `Gợi ý cụm điểm chưa có box (#${idx + 1}, ${c.totalPts} điểm)`,
    message: `Tại frame ${frame}, tọa độ (${c.x.toFixed(1)}, ${c.y.toFixed(1)}, ${c.z.toFixed(1)}), cách LiDAR ${c.dist.toFixed(1)} m, cao ~${c.maxH.toFixed(1)} m.`,
    suggestion: "Kiểm tra camera tại vị trí này xem có phải đối tượng bị bỏ sót hay chỉ là cây/biển báo/cột đèn."
  }));
}
