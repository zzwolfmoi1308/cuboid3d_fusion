/**
 * Bộ Luật QA Metadata & Tọa độ 3D (V1 — 100% Chỉ Đọc, Đa Guideline / Đa Profile)
 * Đọc trực tiếp payload từ GET /api/jobs/{id}/annotations (chứa đầy đủ tracks, shapes, keyframe & outside: true).
 * Mọi công tắc luật (R1..R9), khoảng cách tối đa, bảng kích thước, nhóm class, chu kỳ frame đều cấu hình động qua Profile.
 */

import {
  VALID_CLASSES,
  CLASS_GROUPS,
  GROUP_MAX_DISTANCE,
  GROUP_SIZE_TOLERANCE,
  DEFAULT_RULE_TOGGLES,
  DEFAULT_SETTINGS,
  findClassSpec,
  getExpectedDxDyDzRange
} from "./guideline_config.js";

const RAD2DEG = 180 / Math.PI;

export function wrapAngleRad(rad) {
  let a = rad;
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

export function shortestAngleDiffDeg(aRad, bRad) {
  return wrapAngleRad(bRad - aRad) * RAD2DEG;
}

export function median(values) {
  if (!values || values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

export function percentile(values, p) {
  if (!values || values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * p)));
  return sorted[idx];
}

export function parseCuboidPoints(points) {
  if (!Array.isArray(points) || points.length < 9) return null;
  const [x, y, z, rx, ry, rz, dx, dy, dz] = points;
  if (![x, y, z, rx, ry, rz, dx, dy, dz].every(Number.isFinite)) return null;
  return { x, y, z, rx, ry, rz, dx, dy, dz };
}

/**
 * Nội suy tuyến tính trạng thái của track cho mọi frame từ keyframe đầu đến frame cuối,
 * đồng thời giữ nguyên cờ `isKeyframe` để không trộn lẫn keyframe và frame nội suy.
 */
export function expandTrackFrames(track, stopFrame = null) {
  const rawShapes = [...(track.shapes || [])].sort((a, b) => a.frame - b.frame);
  if (rawShapes.length === 0) return [];

  const expanded = [];
  for (let i = 0; i < rawShapes.length; i++) {
    const curr = rawShapes[i];
    const next = rawShapes[i + 1];

    expanded.push({
      frame: curr.frame,
      points: curr.points,
      occluded: Boolean(curr.occluded),
      outside: Boolean(curr.outside),
      isKeyframe: true,
      attributes: curr.attributes || []
    });

    if (curr.outside) {
      continue;
    }

    if (next && next.frame > curr.frame + 1) {
      const p0 = parseCuboidPoints(curr.points);
      const p1 = parseCuboidPoints(next.points);
      const span = next.frame - curr.frame;
      for (let f = curr.frame + 1; f < next.frame; f++) {
        const t = (f - curr.frame) / span;
        let interpPts = curr.points;
        if (p0 && p1 && !next.outside) {
          const dyaw = wrapAngleRad(p1.rz - p0.rz);
          interpPts = [
            p0.x + (p1.x - p0.x) * t,
            p0.y + (p1.y - p0.y) * t,
            p0.z + (p1.z - p0.z) * t,
            p0.rx + (p1.rx - p0.rx) * t,
            p0.ry + (p1.ry - p0.ry) * t,
            wrapAngleRad(p0.rz + dyaw * t),
            p0.dx + (p1.dx - p0.dx) * t,
            p0.dy + (p1.dy - p0.dy) * t,
            p0.dz + (p1.dz - p0.dz) * t,
            0, 0, 0, 0, 0, 0, 0
          ];
        }
        expanded.push({
          frame: f,
          points: interpPts,
          occluded: Boolean(curr.occluded),
          outside: false,
          isKeyframe: false,
          attributes: curr.attributes || []
        });
      }
    } else if (!next && typeof stopFrame === "number" && stopFrame > curr.frame) {
      for (let f = curr.frame + 1; f <= stopFrame; f++) {
        expanded.push({
          frame: f,
          points: curr.points,
          occluded: Boolean(curr.occluded),
          outside: false,
          isKeyframe: false,
          attributes: curr.attributes || []
        });
      }
    }
  }

  return expanded;
}

export function selectReferenceKeyframe(keyframes, pcdBestFrame = null) {
  const visible = keyframes.filter((k) => !k.outside && parseCuboidPoints(k.points));
  if (visible.length === 0) return null;

  if (typeof pcdBestFrame === "number") {
    const found = visible.find((k) => k.frame === pcdBestFrame);
    if (found) return found;
  }

  const unoccluded = visible.filter((k) => !k.occluded);
  const pool = unoccluded.length > 0 ? unoccluded : visible;

  let best = pool[0];
  let bestDist = Infinity;
  for (const kf of pool) {
    const p = parseCuboidPoints(kf.points);
    const d = Math.hypot(p.x, p.y);
    if (d < bestDist) {
      bestDist = d;
      best = kf;
    }
  }
  return best;
}

function resolveEffectiveRules(settings) {
  const r = {
    ...DEFAULT_RULE_TOGGLES,
    ...(settings.rules || {})
  };
  if (settings.enableDistanceRule === false) r.R2_DISTANCE = false;
  if (settings.enableSizeRefRule === false) r.R3_SIZE_REF = false;
  if (settings.enableBarrierAxisRule === false) r.R4_BARRIER_AXIS = false;
  if (settings.warnShapeInsteadOfTrack === false) r.R9_WARN_SHAPE = false;
  if (settings.allowPitchRoll === true && settings.rules?.R5_PITCH_ROLL === undefined) {
    r.R5_PITCH_ROLL = false;
  }
  return r;
}

function resolveClassGroup(className, settings) {
  const spec = findClassSpec(className, settings);
  if (spec?.group) return spec.group;
  if (CLASS_GROUPS[className]) return CLASS_GROUPS[className];
  const low = String(className || "").toLowerCase();
  for (const [k, v] of Object.entries(CLASS_GROUPS)) {
    if (k.toLowerCase() === low) return v;
  }
  return null;
}

function isPedestrianLikeClass(className, settings) {
  const spec = findClassSpec(className, settings);
  if (typeof spec?.ignoreYaw === "boolean") return spec.ignoreYaw;
  return /^(pedestrian|person|human|people|walker|child|nguoi)$/i.test(String(className || "").trim());
}

/**
 * Chạy các Luật QA V1 đang bật trong Profile trên raw annotations của 1 job CVAT 3D.
 */
export function runMetadataQA(rawAnnotations, labelIdToName = {}, userSettings = {}, options = {}) {
  const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
  const settings = { ...DEFAULT_SETTINGS, ...userSettings };
  const rules = resolveEffectiveRules(settings);
  const findings = [];
  const tracks = Array.isArray(rawAnnotations?.tracks) ? rawAnnotations.tracks : [];
  const standaloneShapes = Array.isArray(rawAnnotations?.shapes) ? rawAnnotations.shapes : [];

  const getLabelName = (item) => {
    if (item.label_name) return item.label_name;
    if (item.label && typeof item.label === "string") return item.label;
    return labelIdToName[item.label_id] || `Unknown(${item.label_id})`;
  };

  let findingCounter = 0;
  const pushFinding = (f) => {
    findings.push({
      id: `qa_${++findingCounter}`,
      ...f
    });
  };

  // --- A. Kiểm tra các Shape đơn lẻ ---
  const allowedClasses =
    Array.isArray(settings.validClasses) && settings.validClasses.length > 0
      ? settings.validClasses
      : settings.taxonomyMode === "job_labels"
        ? null
        : VALID_CLASSES;

  const isClassAllowed = (cls) => {
    if (!allowedClasses) return true;
    return allowedClasses.some((c) => c.toLowerCase() === String(cls || "").toLowerCase());
  };

  for (const sh of standaloneShapes) {
    const className = getLabelName(sh);
    const trackId = sh.id ?? null;
    if (rules.R9_WARN_SHAPE) {
      pushFinding({
        ruleId: "R9_SHAPE_INSTEAD_OF_TRACK",
        severity: "Info",
        frame: sh.frame,
        trackId,
        objectType: "shape",
        className,
        title: "Đối tượng dạng Shape lẻ (không phải Track)",
        message: `Shape #${trackId ?? "?"} (${className}) tại frame ${sh.frame}. Profile hiện tại khuyến nghị vẽ Cuboid 3D dạng Track.`,
        suggestion: "Chuyển sang dạng Track hoặc tắt luật R9s trong Settings nếu task dùng Shape độc lập."
      });
    }

    if (rules.R1_TAXONOMY && !isClassAllowed(className)) {
      pushFinding({
        ruleId: "R1_TAXONOMY",
        severity: "Error",
        frame: sh.frame,
        trackId,
        objectType: "shape",
        className,
        title: `Sai Taxonomy: "${className}"`,
        message: `Class "${className}" không nằm trong danh sách nhãn cho phép của profile hiện tại.`,
        suggestion: `Đổi về đúng 1 trong các nhãn hợp lệ: ${allowedClasses.join(", ")}.`
      });
    }
    const p = parseCuboidPoints(sh.points);
    if (p && !sh.outside) {
      checkSingleBoxRules({
        p,
        className,
        frame: sh.frame,
        trackId,
        objectType: "shape",
        occluded: Boolean(sh.occluded),
        outside: Boolean(sh.outside),
        settings,
        rules,
        pushFinding
      });
    }
  }

  // --- B. Thu thập dữ liệu theo frame để khử góc quay ego cho Luật 8 ---
  const trackRecords = [];
  const frameDeltaYaws = new Map();

  for (let idx = 0; idx < tracks.length; idx++) {
    const tr = tracks[idx];
    const trackId = tr.id ?? idx + 1;
    const className = getLabelName(tr);
    const rawKeyframes = [...(tr.shapes || [])].sort((a, b) => a.frame - b.frame);
    const expanded = expandTrackFrames(tr, options.stopFrame);
    trackRecords.push({ tr, trackId, className, rawKeyframes, expanded });

    const grp = resolveClassGroup(className, settings);
    const isPedLike = isPedestrianLikeClass(className, settings);
    if (grp === "Vehicle" || grp === "Static" || (!grp && !isPedLike)) {
      for (let i = 1; i < expanded.length; i++) {
        const prev = expanded[i - 1];
        const curr = expanded[i];
        if (prev.outside || curr.outside || curr.frame !== prev.frame + 1) continue;
        const p0 = parseCuboidPoints(prev.points);
        const p1 = parseCuboidPoints(curr.points);
        if (!p0 || !p1) continue;
        const dyaw = shortestAngleDiffDeg(p0.rz, p1.rz);
        if (!frameDeltaYaws.has(curr.frame)) frameDeltaYaws.set(curr.frame, []);
        frameDeltaYaws.get(curr.frame).push(dyaw);
      }
    }
  }

  const egoYawCompensationByFrame = new Map();
  for (const [frame, dyaws] of frameDeltaYaws.entries()) {
    egoYawCompensationByFrame.set(frame, dyaws.length >= 3 ? median(dyaws) : 0);
  }

  const allCompensatedDyaws = [];
  const allSecondOrderAccels = [];
  const frameDt = Math.max(0.02, Number(settings.frameIntervalSec) || 0.5);

  // --- C. Chạy các Luật đang bật trên từng Track ---
  for (const rec of trackRecords) {
    const { trackId, className, rawKeyframes, expanded } = rec;
    const firstFrame = rawKeyframes[0]?.frame ?? 0;

    // LUẬT 1: Taxonomy
    if (rules.R1_TAXONOMY && !isClassAllowed(className)) {
      pushFinding({
        ruleId: "R1_TAXONOMY",
        severity: "Error",
        frame: firstFrame,
        trackId,
        objectType: "track",
        className,
        title: `Sai Taxonomy: "${className}"`,
        message: `Track #${trackId} mang class "${className}" không nằm trong danh sách nhãn cho phép của profile hiện tại.`,
        suggestion: `Đổi về đúng 1 trong các nhãn hợp lệ: ${allowedClasses.join(", ")}.`
      });
    }

    // LUẬT 9a & 9b: Tính liên tục Track
    if (rules.R9_TRACK_CONTINUITY) {
      let sawOutside = false;
      let firstOutsideFrame = null;
      for (const kf of rawKeyframes) {
        if (kf.outside) {
          if (!sawOutside) {
            sawOutside = true;
            firstOutsideFrame = kf.frame;
          }
        } else if (sawOutside) {
          pushFinding({
            ruleId: "R9_TRACK_REAPPEAR_AFTER_OUTSIDE",
            severity: "Warning",
            frame: kf.frame,
            trackId,
            objectType: "track",
            className,
            title: "Nghi ngờ tái sử dụng Track ID sau outside",
            message: `Track #${trackId} (${className}) đã bật outside=true tại frame ${firstOutsideFrame} nhưng lại xuất hiện tiếp tại frame ${kf.frame}.`,
            suggestion: "Nếu vật đã rời vùng quan sát hoặc bị che quá ngưỡng, hãy tạo Track ID mới thay vì tái sử dụng."
          });
          break;
        }
      }

      const activeFrames = expanded.filter((f) => !f.outside);
      if (activeFrames.length === 1) {
        const fOnly = activeFrames[0].frame;
        const isLastFrame = typeof options.stopFrame === "number" && fOnly === options.stopFrame;
        pushFinding({
          ruleId: "R9_SINGLE_FRAME_TRACK",
          severity: "Info",
          frame: fOnly,
          trackId,
          objectType: "track",
          className,
          title: "Track chỉ xuất hiện 1 frame lẻ loi",
          message: `Track #${trackId} (${className}) chỉ có 1 frame hiển thị tại frame ${fOnly}${isLastFrame ? " (frame cuối — có thể hợp lệ)" : ""}.`,
          suggestion: "Kiểm tra lại xem vật có xuất hiện ở các frame lân cận hay không."
        });
      }
    }

    // LUẬT 7a & 7b: Occluded & Outside
    if (rules.R7_OCCLUDED_OUTSIDE) {
      for (const kf of rawKeyframes) {
        if (kf.occluded && kf.outside) {
          pushFinding({
            ruleId: "R7_OCCLUDED_AND_OUTSIDE",
            severity: "Error",
            frame: kf.frame,
            trackId,
            objectType: "track",
            className,
            title: "Xung đột: occluded và outside cùng bật",
            message: `Track #${trackId} (${className}) tại frame ${kf.frame} đang bật đồng thời cả occluded=true và outside=true.`,
            suggestion: "Chọn tắt 1 trong 2 cờ (dùng V2 Controlled Assist hoặc chỉnh tay trên CVAT).",
            meta: { canFixConflict: true }
          });
        }
      }

      const seqForOcc = settings.occludedCountMode === "keyframe" ? rawKeyframes : expanded;
      let occRunStart = null;
      let occRunCount = 0;
      const flushOccRun = () => {
        if (occRunCount === 0) return;
        const exceeded =
          settings.occludedOperator === ">="
            ? occRunCount >= settings.occludedThreshold
            : occRunCount > settings.occludedThreshold;
        if (exceeded) {
          const unit = settings.occludedCountMode === "keyframe" ? "keyframe" : "frame";
          pushFinding({
            ruleId: "R7_OCCLUDED_STREAK",
            severity: "Warning",
            frame: occRunStart,
            trackId,
            objectType: "track",
            className,
            title: `Che khuất liên tiếp ${occRunCount} ${unit}`,
            message: `Track #${trackId} (${className}) bị occluded liên tiếp ${occRunCount} ${unit} bắt đầu từ frame ${occRunStart} (ngưỡng hiện tại: ${settings.occludedOperator} ${settings.occludedThreshold} ${unit}).`,
            suggestion: "Nếu bị che quá lâu hoặc không chắc cùng vật, hãy switch outside kết thúc track cũ và tạo track mới khi hiện lại.",
            meta: { occRunCount, unit }
          });
        }
        occRunStart = null;
        occRunCount = 0;
      };

      for (const item of seqForOcc) {
        if (item.occluded && !item.outside) {
          if (occRunCount === 0) occRunStart = item.frame;
          occRunCount++;
        } else {
          flushOccRun();
        }
      }
      flushOccRun();
    }

    // LUẬT 2, 3, 4, 5 trên từng keyframe thật
    for (const kf of rawKeyframes) {
      if (kf.outside) continue;
      const p = parseCuboidPoints(kf.points);
      if (!p) continue;
      checkSingleBoxRules({
        p,
        className,
        frame: kf.frame,
        trackId,
        objectType: "track",
        occluded: Boolean(kf.occluded),
        outside: Boolean(kf.outside),
        settings,
        rules,
        pushFinding
      });
    }

    // LUẬT 6: Tính nhất quán kích thước trong cùng Track
    if (rules.R6_SIZE_CONSISTENCY) {
      const visibleKeyframes = rawKeyframes.filter((k) => !k.outside && parseCuboidPoints(k.points));
      if (visibleKeyframes.length >= 2) {
        const pcdBestFrame = options.pcdBestKeyframeByTrack?.[String(trackId)] ?? null;
        const refKf = selectReferenceKeyframe(visibleKeyframes, pcdBestFrame);
        const refP = refKf ? parseCuboidPoints(refKf.points) : null;

        if (refP) {
          const tolFrac = (settings.trackSizeTolerancePct ?? 5) / 100;
          const deviatedKeyframes = [];

          for (const kf of visibleKeyframes) {
            if (kf.frame === refKf.frame) continue;
            const p = parseCuboidPoints(kf.points);
            const dL = Math.abs(p.dx - refP.dx) / (refP.dx || 1e-6);
            const dW = Math.abs(p.dy - refP.dy) / (refP.dy || 1e-6);
            const dH = Math.abs(p.dz - refP.dz) / (refP.dz || 1e-6);
            const maxDev = Math.max(dL, dW, dH);

            if (maxDev > tolFrac) {
              deviatedKeyframes.push({
                frame: kf.frame,
                dx: p.dx,
                dy: p.dy,
                dz: p.dz,
                dLPct: Number((dL * 100).toFixed(1)),
                dWPct: Number((dW * 100).toFixed(1)),
                dHPct: Number((dH * 100).toFixed(1)),
                maxDevPct: Number((maxDev * 100).toFixed(1)),
                suspectedException: maxDev > 0.25
              });
            }
          }

          if (deviatedKeyframes.length > 0) {
            const worst = deviatedKeyframes.reduce((a, b) => (b.maxDevPct > a.maxDevPct ? b : a));
            pushFinding({
              ruleId: "R6_SIZE_CONSISTENCY",
              severity: "Warning",
              frame: worst.frame,
              trackId,
              objectType: "track",
              className,
              title: `Lệch kích thước trong Track tới ${worst.maxDevPct}%`,
              message: `Track #${trackId} (${className}) có ${deviatedKeyframes.length} keyframe lệch > ±${settings.trackSizeTolerancePct}% so với keyframe chuẩn #${refKf.frame} (${refP.dx.toFixed(2)}×${refP.dy.toFixed(2)}×${refP.dz.toFixed(2)} m). Lệch lớn nhất tại frame ${worst.frame} (L:${worst.dLPct}%, W:${worst.dWPct}%, H:${worst.dHPct}%).`,
              suggestion: "Kiểm tra ngoại lệ (mở cửa, hạ sàn, bốc dỡ hàng, người ngồi/cúi, ghép rơ-moóc). Nếu không phải ngoại lệ, có thể dùng V2 Khóa Size lên keyframe.",
              meta: {
                refFrame: refKf.frame,
                refDims: { dx: refP.dx, dy: refP.dy, dz: refP.dz },
                deviatedKeyframes
              }
            });
          }
        }
      }
    }

    // LUẬT 8: Chuyển động & Yaw
    for (let i = 1; i < expanded.length; i++) {
      const prev = expanded[i - 1];
      const curr = expanded[i];
      if (prev.outside || curr.outside || curr.frame !== prev.frame + 1) continue;

      const p0 = parseCuboidPoints(prev.points);
      const p1 = parseCuboidPoints(curr.points);
      if (!p0 || !p1) continue;

      const rawDyaw = shortestAngleDiffDeg(p0.rz, p1.rz);
      const egoComp = egoYawCompensationByFrame.get(curr.frame) || 0;
      const netDyaw = wrapAngleRad(((rawDyaw - egoComp) * Math.PI) / 180) * RAD2DEG;
      const absNetDyaw = Math.abs(netDyaw);
      allCompensatedDyaws.push(absNetDyaw);

      if (rules.R8_MOTION_YAW) {
        const isPed = isPedestrianLikeClass(className, settings);
        if (!(isPed && settings.ignorePedestrianYaw)) {
          const yawThreshold = isPed
            ? settings.maxYawJumpPedestrianDeg
            : settings.maxYawJumpVehicleDeg;

          if (absNetDyaw > yawThreshold) {
            const near180 = Math.abs(absNetDyaw - 180) <= 25;
            pushFinding({
              ruleId: "R8_YAW_JUMP",
              severity: "Warning",
              frame: curr.frame,
              trackId,
              objectType: "track",
              className,
              title: near180
                ? `Nghi ngờ ngược đầu 180° (Δyaw = ${netDyaw.toFixed(1)}°)`
                : `Nhảy góc Yaw (Δyaw = ${netDyaw.toFixed(1)}°)`,
              message: `Track #${trackId} (${className}) đổi góc yaw từ frame ${prev.frame} → ${curr.frame}: Δyaw sau khử ego = ${netDyaw.toFixed(1)}° (thô: ${rawDyaw.toFixed(1)}°, ego median: ${egoComp.toFixed(1)}°).`,
              suggestion: near180
                ? "Kiểm tra mũi tên đỏ trên Camera/LiDAR; nếu bị ngược đầu, dùng Flip Yaw 180°."
                : "Kiểm tra lại độ mượt yaw giữa các keyframe khúc rẽ hoặc hiện tượng wrap ±π.",
              meta: {
                rawDyaw: Number(rawDyaw.toFixed(1)),
                egoComp: Number(egoComp.toFixed(1)),
                netDyaw: Number(netDyaw.toFixed(1)),
                near180
              }
            });
          }
        }

        if (settings.coordinateFrame === "world_centric") {
          const dist = Math.hypot(p1.x - p0.x, p1.y - p0.y);
          const speedMps = dist / frameDt;
          if (speedMps > settings.maxWorldSpeedMps) {
            pushFinding({
              ruleId: "R8_WORLD_SPEED",
              severity: "Warning",
              frame: curr.frame,
              trackId,
              objectType: "track",
              className,
              title: `Tốc độ bất thường (${speedMps.toFixed(1)} m/s)`,
              message: `Track #${trackId} (${className}) dịch chuyển ${dist.toFixed(2)} m trong ${frameDt} s (${speedMps.toFixed(1)} m/s ≈ ${(speedMps * 3.6).toFixed(0)} km/h) giữa frame ${prev.frame} → ${curr.frame}.`,
              suggestion: "Kiểm tra xem có bị nhầm Track ID giữa hai vật khác nhau hay dịch nhầm box không."
            });
          }
        }
      }

      if (i + 1 < expanded.length) {
        const next = expanded[i + 1];
        if (!next.outside && next.frame === curr.frame + 1) {
          const p2 = parseCuboidPoints(next.points);
          if (p2) {
            const ax = p2.x - 2 * p1.x + p0.x;
            const ay = p2.y - 2 * p1.y + p0.y;
            const az = p2.z - 2 * p1.z + p0.z;
            const accelMag = Math.hypot(ax, ay, az);
            allSecondOrderAccels.push(accelMag);

            if (rules.R8_MOTION_YAW && accelMag > settings.maxSecondOrderAccelM) {
              pushFinding({
                ruleId: "R8_ACCEL_SPIKE",
                severity: "Warning",
                frame: curr.frame,
                trackId,
                objectType: "track",
                className,
                title: `Đột biến quỹ đạo (Δ²p = ${accelMag.toFixed(2)} m)`,
                message: `Track #${trackId} (${className}) có đỉnh sai phân bậc hai Δ²p = ${accelMag.toFixed(2)} m tại frame ${curr.frame} (ngưỡng: ${settings.maxSecondOrderAccelM} m).`,
                suggestion: "Kiểm tra xem box có bị giật vị trí, nhầm đối tượng, hay thiếu keyframe giữa khúc rẽ không.",
                meta: { accelMag: Number(accelMag.toFixed(2)) }
              });
            }
          }
        }
      }
    }
  }

  const elapsedMs = Math.round(
    (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0
  );

  const sevOrder = { Error: 0, Warning: 1, Info: 2 };
  findings.sort((a, b) => {
    const s = (sevOrder[a.severity] ?? 9) - (sevOrder[b.severity] ?? 9);
    if (s !== 0) return s;
    return (a.frame ?? 0) - (b.frame ?? 0);
  });

  const enabledV1Count = [
    "R1_TAXONOMY",
    "R2_DISTANCE",
    "R3_SIZE_REF",
    "R4_BARRIER_AXIS",
    "R5_PITCH_ROLL",
    "R6_SIZE_CONSISTENCY",
    "R7_OCCLUDED_OUTSIDE",
    "R8_MOTION_YAW",
    "R9_TRACK_CONTINUITY"
  ].filter((k) => rules[k]).length;

  const summary = {
    totalTracks: tracks.length,
    totalShapes: standaloneShapes.length,
    totalFindings: findings.length,
    errors: findings.filter((f) => f.severity === "Error").length,
    warnings: findings.filter((f) => f.severity === "Warning").length,
    infos: findings.filter((f) => f.severity === "Info").length,
    enabledRuleCount: enabledV1Count,
    elapsedMs,
    noiseStats: {
      yawSamples: allCompensatedDyaws.length,
      yawP95Deg: Number(percentile(allCompensatedDyaws, 0.95).toFixed(1)),
      yawMaxDeg: Number((allCompensatedDyaws.length ? Math.max(...allCompensatedDyaws) : 0).toFixed(1)),
      accelSamples: allSecondOrderAccels.length,
      accelP95M: Number(percentile(allSecondOrderAccels, 0.95).toFixed(2)),
      accelMaxM: Number((allSecondOrderAccels.length ? Math.max(...allSecondOrderAccels) : 0).toFixed(2))
    }
  };

  return { findings, summary };
}

function checkSingleBoxRules({
  p,
  className,
  frame,
  trackId,
  objectType,
  settings,
  rules,
  pushFinding
}) {
  const customSpec = findClassSpec(className, settings);
  const group = resolveClassGroup(className, settings);
  const dist = Math.hypot(p.x, p.y);

  // LUẬT 2: Phạm vi khoảng cách (ưu tiên maxDistance riêng của class trong Profile, rồi mới đến nhóm)
  if (rules.R2_DISTANCE) {
    const distMap = settings.maxDistanceByGroup || GROUP_MAX_DISTANCE;
    const maxDist =
      customSpec && Number.isFinite(customSpec.maxDistance) && customSpec.maxDistance > 0
        ? Number(customSpec.maxDistance)
        : group
          ? (distMap[group] ?? GROUP_MAX_DISTANCE[group])
          : settings.maxDistanceDefault;
    const margin = settings.distanceBoundaryMarginM ?? 2.0;
    if (Number.isFinite(maxDist) && dist > maxDist) {
      const nearBoundary = dist <= maxDist + margin;
      pushFinding({
        ruleId: "R2_DISTANCE_RANGE",
        severity: "Warning",
        frame,
        trackId,
        objectType,
        className,
        title: `Vượt phạm vi ${maxDist} m (${dist.toFixed(1)} m)`,
        message: `${className} #${trackId} tại frame ${frame} cách LiDAR ${dist.toFixed(1)} m (giới hạn${group ? ` nhóm ${group}` : ""}: ≤ ${maxDist} m).`,
        suggestion: nearBoundary
          ? "Sát biên khoảng cách (±2 m): KHÔNG tự xóa vội — cân nhắc tạo mẫu UNCERTAIN_SCOPE để nhóm vote / hỏi mentor."
          : "Kiểm tra lại phạm vi; nếu đã rời phạm vi thì switch outside thay vì vẽ tiếp.",
        meta: {
          distance: Number(dist.toFixed(2)),
          maxDist,
          nearBoundary,
          suggestedIssueType: nearBoundary ? "UNCERTAIN_SCOPE" : null
        }
      });
    }
  }

  // LUẬT 3: Bảng kích thước tham chiếu (động theo classSpecs của Profile hoặc fallback Guideline V1)
  const expected = rules.R3_SIZE_REF ? getExpectedDxDyDzRange(className, settings) : null;
  if (expected) {
    const effGroup = expected.group || group;
    const tol =
      Number.isFinite(expected.tolerance)
        ? expected.tolerance
        : (effGroup && GROUP_SIZE_TOLERANCE[effGroup]) ?? 0.1;
    const checks = [
      { axis: expected.swapped ? `dx (W ${className})` : "dx (L)", val: p.dx, range: expected.dx },
      { axis: expected.swapped ? `dy (L ${className})` : "dy (W)", val: p.dy, range: expected.dy },
      { axis: "dz (H)", val: p.dz, range: expected.dz }
    ];

    const violations = [];
    for (const c of checks) {
      const lo = c.range[0] * (1 - tol);
      const hi = c.range[1] * (1 + tol);
      if (c.val < lo || c.val > hi) {
        violations.push(`${c.axis}=${c.val.toFixed(2)}m (chuẩn ${c.range[0]}–${c.range[1]}m ±${Math.round(tol * 100)}%)`);
      }
    }

    if (violations.length > 0) {
      pushFinding({
        ruleId: "R3_SIZE_REFERENCE",
        severity: "Warning",
        frame,
        trackId,
        objectType,
        className,
        title: `Kích thước ngoài bảng tham chiếu`,
        message: `${className} #${trackId} tại frame ${frame}: ${violations.join("; ")}.`,
        suggestion: "Đối chiếu lại point cloud + camera hoặc cập nhật khoảng kích thước class trong Profile."
      });
    }
  }

  // LUẬT 4: Trục hướng Barrier (chỉ chạy khi R4 bật VÀ đúng nhãn Barrier của Profile)
  const barrierName = String(settings.barrierLabelName ?? "Barrier").trim().toLowerCase();
  if (rules.R4_BARRIER_AXIS && barrierName && String(className || "").toLowerCase() === barrierName) {
    const wrongOrientation =
      settings.barrierAxisMode === "width_is_x"
        ? p.dx > p.dy * 1.15
        : p.dy > p.dx * 1.15;

    if (wrongOrientation) {
      pushFinding({
        ruleId: "R4_BARRIER_AXIS",
        severity: "Warning",
        frame,
        trackId,
        objectType,
        className,
        title: `Nghi ngờ xoay nhầm trục 90° của ${className}`,
        message: `${className} #${trackId} tại frame ${frame} có dx=${p.dx.toFixed(2)}m, dy=${p.dy.toFixed(2)}m — trái với quy ước trục hiện tại (${settings.barrierAxisMode === "width_is_x" ? "dy > dx" : "dx > dy"}).`,
        suggestion: "Kiểm tra hướng mũi tên đỏ; có thể đổi quy ước trục trong Settings hoặc xoay 90° + hoán L/W.",
        meta: { canRotateBarrier90: true }
      });
    }
  }

  // LUẬT 5: Pitch / Roll khác 0
  if (rules.R5_PITCH_ROLL) {
    const rxDeg = Math.abs(wrapAngleRad(p.rx) * RAD2DEG);
    const ryDeg = Math.abs(wrapAngleRad(p.ry) * RAD2DEG);
    const maxPR = settings.maxPitchRollDeg ?? 5.0;
    if (rxDeg > maxPR || ryDeg > maxPR) {
      pushFinding({
        ruleId: "R5_PITCH_ROLL_NONZERO",
        severity: "Warning",
        frame,
        trackId,
        objectType,
        className,
        title: `Pitch/Roll khác 0 (rx=${rxDeg.toFixed(1)}°, ry=${ryDeg.toFixed(1)}°)`,
        message: `${className} #${trackId} tại frame ${frame} bị nghiêng trục rx=${rxDeg.toFixed(1)}°, ry=${ryDeg.toFixed(1)}° (ngưỡng ≤ ${maxPR}°).`,
        suggestion: "Đưa pitch/roll về 0° trên Side/Front view (hoặc tắt luật R5 nếu task cho phép đường dốc 9-DoF)."
      });
    }
  }
}
