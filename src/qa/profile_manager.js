/**
 * Multi-Guideline Profile & Dynamic Rule/Schema Manager — CVAT 3D Fusion Suite
 * Hỗ trợ:
 *  - 10 class AI gốc (tách riêng Motorcycle & Bicycle, tự gộp thành Two_wheels nếu cùng map vào 1 nhãn)
 *  - Bật/tắt độc lập toàn bộ 15 Luật QA (R1..R9 + PCD1..PCD5)
 *  - Cấu hình riêng từng class (classSpecs: group, maxDistance, L/W/H, sizeTolerance, ignoreYaw)
 *  - Checklist, Bảng kích thước tham chiếu, Mẫu Issue và Thông số Sensor riêng theo từng Profile
 *  - Tự động nhận diện xem Barrier có liên quan tới Job/Profile hiện tại hay không để ẩn sạch UI thừa
 */

import {
  VALID_CLASSES,
  CLASS_GROUPS,
  GROUP_MAX_DISTANCE,
  GROUP_SIZE_TOLERANCE,
  SIZE_REFERENCE_SUBTYPES,
  CLASS_SIZE_UNION,
  DEFAULT_RULE_TOGGLES,
  SUBMIT_CHECKLIST_ITEMS,
  GENERIC_CHECKLIST_ITEMS,
  MENTOR_ISSUE_TYPES,
  DEFAULT_SETTINGS
} from "./guideline_config.js";

export const CANONICAL_AI_CLASSES = [
  {
    key: "Car",
    label: "Car (Ô tô con / SUV / Van)",
    defaultGroup: "Vehicle",
    synonyms: ["car", "sedan", "suv", "van", "vehicle", "auto", "automobile", "hatchback", "minivan", "passenger_car", "o_to"]
  },
  {
    key: "Truck",
    label: "Truck (Xe tải / Bán tải)",
    defaultGroup: "Vehicle",
    synonyms: ["truck", "pickup", "lorry", "cargo", "heavy_vehicle", "xe_tai", "van_truck"]
  },
  {
    key: "Trailer",
    label: "Trailer (Rơ-moóc)",
    defaultGroup: "Vehicle",
    synonyms: ["trailer", "semi_trailer", "container", "ro_mooc"]
  },
  {
    key: "Bus",
    label: "Bus (Xe buýt)",
    defaultGroup: "Vehicle",
    synonyms: ["bus", "coach", "shuttle", "xe_buyt"]
  },
  {
    key: "Construction_vehicle",
    label: "Construction_vehicle (Xe công trình)",
    defaultGroup: "Vehicle",
    synonyms: ["construction_vehicle", "construction", "excavator", "crane", "special_vehicle", "engineering_vehicle", "machinery"]
  },
  {
    key: "Pedestrian",
    label: "Pedestrian (Người đi bộ)",
    defaultGroup: "VRU",
    synonyms: ["pedestrian", "person", "human", "people", "walker", "nguoi", "ped"]
  },
  {
    key: "Motorcycle",
    label: "Motorcycle (Xe máy / Mô tô)",
    defaultGroup: "VRU",
    synonyms: ["motorcycle", "motorbike", "scooter", "moto", "xe_may", "motorcyclist", "two_wheels", "two_wheeler"]
  },
  {
    key: "Bicycle",
    label: "Bicycle (Xe đạp / Người đi xe đạp)",
    defaultGroup: "VRU",
    synonyms: ["bicycle", "bike", "cyclist", "bicyclist", "xe_dap", "two_wheels", "two_wheeler"]
  },
  {
    key: "Barrier",
    label: "Barrier (Rào chắn)",
    defaultGroup: "Static",
    synonyms: ["barrier", "barricade", "guardrail", "fence", "jersey_barrier", "rao_chan"]
  },
  {
    key: "Traffic_cone",
    label: "Traffic_cone (Cọc tiêu giao thông)",
    defaultGroup: "Static",
    synonyms: ["traffic_cone", "cone", "pylon", "coc_tieu", "bollard"]
  }
];

function buildDefaultV1ClassSpecs() {
  const specs = {};
  for (const cls of VALID_CLASSES) {
    const grp = CLASS_GROUPS[cls] || "Vehicle";
    const u = CLASS_SIZE_UNION[cls];
    specs[cls] = {
      className: cls,
      group: grp,
      maxDistance: GROUP_MAX_DISTANCE[grp] ?? 50,
      L: u ? [...u.L] : null,
      W: u ? [...u.W] : null,
      H: u ? [...u.H] : null,
      sizeTolerance: GROUP_SIZE_TOLERANCE[grp] ?? 0.1,
      ignoreYaw: cls === "Pedestrian",
      axisMode: cls === "Barrier" ? "barrier_sync" : "length_is_x"
    };
  }
  return specs;
}

export const BUILTIN_PROFILES = {
  guideline_v1_ai20k: {
    id: "guideline_v1_ai20k",
    name: "Guideline V1 (AI20K · 8 class chuẩn)",
    builtin: true,
    taxonomyMode: "strict_profile",
    validClasses: [...VALID_CLASSES],
    labelMapping: {
      Car: "Car",
      Truck: "Truck",
      Trailer: "Trailer",
      Bus: "Bus",
      Construction_vehicle: "Construction_vehicle",
      Pedestrian: "Pedestrian",
      Motorcycle: "Two_wheels",
      Bicycle: "Two_wheels",
      Two_wheels: "Two_wheels",
      Barrier: "Barrier",
      Traffic_cone: ""
    },
    rules: { ...DEFAULT_RULE_TOGGLES },
    enableDistanceRule: true,
    maxDistanceVehicle: GROUP_MAX_DISTANCE.Vehicle,
    maxDistanceVru: GROUP_MAX_DISTANCE.VRU,
    maxDistanceStatic: GROUP_MAX_DISTANCE.Static,
    maxDistanceDefault: 50,
    enableSizeRefRule: true,
    enableBarrierAxisRule: true,
    barrierLabelName: "Barrier",
    defaultObjectType: "track",
    warnShapeInsteadOfTrack: true,
    classSpecs: buildDefaultV1ClassSpecs(),
    sizeReferenceRows: JSON.parse(JSON.stringify(SIZE_REFERENCE_SUBTYPES)),
    checklistItems: JSON.parse(JSON.stringify(SUBMIT_CHECKLIST_ITEMS)),
    issueTypes: JSON.parse(JSON.stringify(MENTOR_ISSUE_TYPES)),
    sensorConfig: {
      frameIntervalSec: 0.5,
      allowPitchRoll: false,
      mergeRiderWithTwoWheels: true,
      occludedMinPoints: 5,
      maxFrameGap: 4
    },
    qaThresholds: {
      barrierAxisMode: DEFAULT_SETTINGS.barrierAxisMode,
      occludedOperator: DEFAULT_SETTINGS.occludedOperator,
      occludedThreshold: DEFAULT_SETTINGS.occludedThreshold,
      occludedCountMode: DEFAULT_SETTINGS.occludedCountMode,
      coordinateFrame: DEFAULT_SETTINGS.coordinateFrame,
      maxPitchRollDeg: DEFAULT_SETTINGS.maxPitchRollDeg,
      maxYawJumpVehicleDeg: DEFAULT_SETTINGS.maxYawJumpVehicleDeg,
      maxSecondOrderAccelM: DEFAULT_SETTINGS.maxSecondOrderAccelM,
      trackSizeTolerancePct: DEFAULT_SETTINGS.trackSizeTolerancePct,
      ignorePedestrianYaw: DEFAULT_SETTINGS.ignorePedestrianYaw
    }
  },

  generic_auto: {
    id: "generic_auto",
    name: "Generic CVAT 3D (Tự động theo Job)",
    builtin: true,
    taxonomyMode: "job_labels",
    validClasses: [],
    labelMapping: {
      Car: "Car",
      Truck: "Truck",
      Trailer: "Trailer",
      Bus: "Bus",
      Construction_vehicle: "Construction_vehicle",
      Pedestrian: "Pedestrian",
      Motorcycle: "Two_wheels",
      Bicycle: "Two_wheels",
      Two_wheels: "Two_wheels",
      Barrier: "",
      Traffic_cone: ""
    },
    rules: {
      ...DEFAULT_RULE_TOGGLES,
      R2_DISTANCE: false,
      R3_SIZE_REF: false,
      R4_BARRIER_AXIS: false,
      R9_WARN_SHAPE: false
    },
    enableDistanceRule: false,
    maxDistanceVehicle: 75,
    maxDistanceVru: 60,
    maxDistanceStatic: 50,
    maxDistanceDefault: 75,
    enableSizeRefRule: false,
    enableBarrierAxisRule: false,
    barrierLabelName: "",
    defaultObjectType: "track",
    warnShapeInsteadOfTrack: false,
    classSpecs: {},
    sizeReferenceRows: [],
    checklistItems: JSON.parse(JSON.stringify(GENERIC_CHECKLIST_ITEMS)),
    issueTypes: JSON.parse(JSON.stringify(MENTOR_ISSUE_TYPES)),
    sensorConfig: {
      frameIntervalSec: 0.5,
      allowPitchRoll: false,
      mergeRiderWithTwoWheels: true,
      occludedMinPoints: 5,
      maxFrameGap: 4
    },
    qaThresholds: {
      barrierAxisMode: DEFAULT_SETTINGS.barrierAxisMode,
      occludedOperator: DEFAULT_SETTINGS.occludedOperator,
      occludedThreshold: DEFAULT_SETTINGS.occludedThreshold,
      occludedCountMode: DEFAULT_SETTINGS.occludedCountMode,
      coordinateFrame: DEFAULT_SETTINGS.coordinateFrame,
      maxPitchRollDeg: DEFAULT_SETTINGS.maxPitchRollDeg,
      maxYawJumpVehicleDeg: DEFAULT_SETTINGS.maxYawJumpVehicleDeg,
      maxSecondOrderAccelM: DEFAULT_SETTINGS.maxSecondOrderAccelM,
      trackSizeTolerancePct: DEFAULT_SETTINGS.trackSizeTolerancePct,
      ignorePedestrianYaw: DEFAULT_SETTINGS.ignorePedestrianYaw
    }
  }
};

export function cloneProfile(profile) {
  return JSON.parse(JSON.stringify(profile));
}

export function normalizeToken(s) {
  return String(s || "")
    .trim()
    .toLowerCase()
    .replace(/[\s\-./]+/g, "_");
}

/**
 * Suy luận thông số mặc định thông minh cho 1 tên nhãn bất kỳ khi tạo Profile từ Job CVAT
 */
export function inferClassSpecFromLabelName(labelName) {
  const orig = String(labelName || "").trim();
  const norm = normalizeToken(orig);

  let matchedCanon = null;
  for (const item of CANONICAL_AI_CLASSES) {
    if (norm === normalizeToken(item.key) || item.synonyms.includes(norm)) {
      matchedCanon = item.key;
      break;
    }
  }
  if (!matchedCanon) {
    if (norm === "two_wheels" || norm === "two_wheeler") matchedCanon = "Two_wheels";
    else {
      for (const item of CANONICAL_AI_CLASSES) {
        if (item.synonyms.some((syn) => norm.includes(syn))) {
          matchedCanon = item.key;
          break;
        }
      }
    }
  }

  const group =
    (matchedCanon && CLASS_GROUPS[matchedCanon]) ||
    (/ped|person|human|walker|child|rider|cyclist|bike|moto/i.test(norm)
      ? "VRU"
      : /barrier|cone|pole|sign|bollard|static|wall|fence/i.test(norm)
        ? "Static"
        : "Vehicle");

  const refKey = matchedCanon || (CLASS_SIZE_UNION[orig] ? orig : null);
  const union = refKey ? CLASS_SIZE_UNION[refKey] : null;
  const isPedLike = /ped|person|human|walker|child/i.test(norm);
  const isBarrierLike = /barrier|barricade|guardrail|rao_chan/i.test(norm);

  return {
    className: orig,
    group,
    maxDistance: GROUP_MAX_DISTANCE[group] ?? 50,
    L: union ? [...union.L] : null,
    W: union ? [...union.W] : null,
    H: union ? [...union.H] : null,
    sizeTolerance: GROUP_SIZE_TOLERANCE[group] ?? 0.1,
    ignoreYaw: isPedLike,
    axisMode: isBarrierLike ? "barrier_sync" : "length_is_x"
  };
}

/**
 * Tự động ánh xạ 10 class AI sang danh sách nhãn thực tế của Job CVAT hiện tại.
 * Nếu Job có nhãn gộp (vd "Two_wheels") thì cả Motorcycle và Bicycle cùng trỏ về "Two_wheels".
 * Nếu Job tách riêng "Motorcycle" và "Bicycle" thì từng loại trỏ đúng nhãn riêng của nó!
 */
export function autoMapLabelsToJob(jobLabels = []) {
  const cleanJobLabels = (jobLabels || []).map((l) => String(l || "").trim()).filter(Boolean);
  const normMap = cleanJobLabels.map((orig) => ({ orig, norm: normalizeToken(orig) }));
  const mapping = {};

  for (const item of CANONICAL_AI_CLASSES) {
    // 1. Exact match (case-insensitive)
    let matched = normMap.find((j) => j.norm === normalizeToken(item.key));
    // 2. Exact synonym match (ưu tiên các từ khóa đặc trưng trước từ chung two_wheels)
    if (!matched) {
      for (const syn of item.synonyms) {
        const hit = normMap.find((j) => j.norm === syn);
        if (hit) {
          matched = hit;
          break;
        }
      }
    }
    // 3. Partial substring match
    if (!matched) {
      for (const syn of item.synonyms) {
        const hit = normMap.find(
          (j) => j.norm.includes(syn) || (j.norm.length >= 3 && syn.includes(j.norm))
        );
        if (hit) {
          matched = hit;
          break;
        }
      }
    }
    mapping[item.key] = matched ? matched.orig : "";
  }

  // Giữ khóa tương thích ngược Two_wheels
  if (mapping.Motorcycle && mapping.Motorcycle === mapping.Bicycle) {
    mapping.Two_wheels = mapping.Motorcycle;
  } else {
    mapping.Two_wheels = mapping.Motorcycle || mapping.Bicycle || "";
  }

  return mapping;
}

/**
 * Kiểm tra xem Profile hiện tại có yêu cầu tách riêng Xe máy (Motorcycle) và Xe đạp (Bicycle) không
 */
export function shouldSplitTwoWheels(labelMapping = {}) {
  if (!labelMapping || typeof labelMapping !== "object") return false;
  const hasExplicitMoto = Object.prototype.hasOwnProperty.call(labelMapping, "Motorcycle");
  const hasExplicitBike = Object.prototype.hasOwnProperty.call(labelMapping, "Bicycle");
  if (!hasExplicitMoto && !hasExplicitBike) return false;
  const m = String(labelMapping.Motorcycle ?? "").trim();
  const b = String(labelMapping.Bicycle ?? "").trim();
  // Nếu cả 2 cùng bằng nhau (vd cùng "Two_wheels" hoặc cùng rỗng), không cần tách
  return m.toLowerCase() !== b.toLowerCase();
}

/**
 * Trả về danh sách canonical classes cần bật cho Python Agent dựa trên labelMapping
 */
export function getEnabledCanonicalClasses(labelMapping = {}) {
  const split2w = shouldSplitTwoWheels(labelMapping);
  const enabled = new Set();
  for (const item of CANONICAL_AI_CLASSES) {
    const val = String(labelMapping?.[item.key] ?? "").trim();
    if (!val || val === "__IGNORE__") continue;
    if (!split2w && (item.key === "Motorcycle" || item.key === "Bicycle")) {
      enabled.add("Two_wheels");
    } else {
      enabled.add(item.key);
    }
  }
  // Fallback nếu mapping cũ chỉ có Two_wheels
  if (!split2w && labelMapping?.Two_wheels && String(labelMapping.Two_wheels).trim()) {
    enabled.add("Two_wheels");
  }
  return Array.from(enabled);
}

/**
 * Kiểm tra xem các tùy chọn & nút Barrier có thực sự cần thiết cho Profile + Job đang mở hay không
 */
export function isBarrierRelevantForContext(profile, jobLabels = []) {
  const p = profile || BUILTIN_PROFILES.generic_auto;
  const barrierTarget = String(p.barrierLabelName || p.labelMapping?.Barrier || "").trim().toLowerCase();

  // Nếu đang mở 1 Job CVAT thật có danh sách nhãn: kiểm tra Job có nhãn Barrier nào không
  if (Array.isArray(jobLabels) && jobLabels.length > 0) {
    const hasBarrierInJob = jobLabels.some((lbl) => {
      const low = String(lbl || "").trim().toLowerCase();
      if (!low) return false;
      if (barrierTarget && low === barrierTarget) return true;
      return /barrier|barricade|guardrail|rao_chan|rào/i.test(low);
    });
    if (!hasBarrierInJob) return false;
  }

  const ruleOn = p.rules ? Boolean(p.rules.R4_BARRIER_AXIS) : Boolean(p.enableBarrierAxisRule);
  return Boolean(ruleOn || barrierTarget);
}

function normalizeClassSpecEntry(name, rawSpec = {}) {
  const inferred = inferClassSpecFromLabelName(name);
  const parsePair = (arr, fallback) => {
    if (Array.isArray(arr) && arr.length >= 2) {
      const a = Number(arr[0]);
      const b = Number(arr[1]);
      if (Number.isFinite(a) && Number.isFinite(b) && a > 0 && b >= a) {
        return [Number(a.toFixed(3)), Number(b.toFixed(3))];
      }
    }
    return fallback;
  };
  const grp = rawSpec.group || inferred.group || "Vehicle";
  const maxD = Number(rawSpec.maxDistance);
  const tol = Number(rawSpec.sizeTolerance);
  return {
    className: String(rawSpec.className || name).trim(),
    group: grp,
    maxDistance: Number.isFinite(maxD) && maxD > 0 ? maxD : inferred.maxDistance,
    L: rawSpec.L === null ? null : parsePair(rawSpec.L, inferred.L),
    W: rawSpec.W === null ? null : parsePair(rawSpec.W, inferred.W),
    H: rawSpec.H === null ? null : parsePair(rawSpec.H, inferred.H),
    noSizeRef: Boolean(rawSpec.noSizeRef),
    sizeTolerance: Number.isFinite(tol) && tol > 0 ? tol : inferred.sizeTolerance,
    ignoreYaw: typeof rawSpec.ignoreYaw === "boolean" ? rawSpec.ignoreYaw : inferred.ignoreYaw,
    axisMode: rawSpec.axisMode || inferred.axisMode || "length_is_x"
  };
}

/**
 * Chuẩn hóa đối tượng Profile (tương thích 100% với cả JSON cũ lẫn JSON mới đa guideline)
 */
export function normalizeProfile(raw, fallbackId = "custom_profile") {
  const isV1 = (raw?.id || fallbackId) === "guideline_v1_ai20k";
  const base = cloneProfile(isV1 ? BUILTIN_PROFILES.guideline_v1_ai20k : BUILTIN_PROFILES.generic_auto);
  const p = raw && typeof raw === "object" ? raw : {};

  // Đồng bộ labelMapping cũ (chỉ có Two_wheels) sang Motorcycle + Bicycle
  const rawMap = { ...(p.labelMapping || {}) };
  if (
    rawMap.Two_wheels !== undefined &&
    rawMap.Motorcycle === undefined &&
    rawMap.Bicycle === undefined
  ) {
    rawMap.Motorcycle = rawMap.Two_wheels;
    rawMap.Bicycle = rawMap.Two_wheels;
  }
  const mergedMapping = {
    ...base.labelMapping,
    ...rawMap
  };
  if (mergedMapping.Motorcycle && mergedMapping.Motorcycle === mergedMapping.Bicycle) {
    mergedMapping.Two_wheels = mergedMapping.Motorcycle;
  } else if (!mergedMapping.Two_wheels) {
    mergedMapping.Two_wheels = mergedMapping.Motorcycle || mergedMapping.Bicycle || "";
  }

  // Đồng bộ công tắc luật (hỗ trợ cả cờ legacy lẫn object p.rules mới)
  const rules = {
    ...base.rules,
    ...(p.rules && typeof p.rules === "object" ? p.rules : {})
  };
  if (typeof p.enableDistanceRule === "boolean" && p.rules?.R2_DISTANCE === undefined) {
    rules.R2_DISTANCE = p.enableDistanceRule;
  }
  if (typeof p.enableSizeRefRule === "boolean" && p.rules?.R3_SIZE_REF === undefined) {
    rules.R3_SIZE_REF = p.enableSizeRefRule;
  }
  if (typeof p.enableBarrierAxisRule === "boolean" && p.rules?.R4_BARRIER_AXIS === undefined) {
    rules.R4_BARRIER_AXIS = p.enableBarrierAxisRule;
  }
  if (typeof p.warnShapeInsteadOfTrack === "boolean" && p.rules?.R9_WARN_SHAPE === undefined) {
    rules.R9_WARN_SHAPE = p.warnShapeInsteadOfTrack;
  }

  // Chuẩn hóa classSpecs
  const normSpecs = {};
  const rawSpecs = p.classSpecs;
  if (Array.isArray(rawSpecs)) {
    for (const item of rawSpecs) {
      if (item?.className) {
        normSpecs[item.className] = normalizeClassSpecEntry(item.className, item);
      }
    }
  } else if (rawSpecs && typeof rawSpecs === "object") {
    for (const [k, v] of Object.entries(rawSpecs)) {
      if (v && typeof v === "object") {
        const key = String(v.className || k).trim();
        normSpecs[key] = normalizeClassSpecEntry(key, v);
      }
    }
  } else if (isV1) {
    Object.assign(normSpecs, buildDefaultV1ClassSpecs());
  }

  const sensorConfig = {
    ...base.sensorConfig,
    ...(p.sensorConfig && typeof p.sensorConfig === "object" ? p.sensorConfig : {})
  };
  if (sensorConfig.allowPitchRoll && p.rules?.R5_PITCH_ROLL === undefined) {
    rules.R5_PITCH_ROLL = false;
  }

  const sizeReferenceRows = Array.isArray(p.sizeReferenceRows)
    ? p.sizeReferenceRows
    : isV1
      ? base.sizeReferenceRows
      : [];

  const checklistItems =
    Array.isArray(p.checklistItems) && p.checklistItems.length > 0
      ? p.checklistItems.map((it, idx) =>
          typeof it === "string"
            ? { id: `chk_custom_${idx + 1}`, text: it }
            : { id: String(it.id || `chk_custom_${idx + 1}`), text: String(it.text || "") }
        )
      : base.checklistItems;

  const issueTypes =
    Array.isArray(p.issueTypes) && p.issueTypes.length > 0
      ? p.issueTypes.map((it) => ({
          code: String(it.code || "ISSUE").trim(),
          label: String(it.label || it.code || "ISSUE").trim(),
          example: String(it.example || "").trim()
        }))
      : base.issueTypes;

  return {
    ...base,
    ...p,
    id: String(p.id || fallbackId),
    name: String(p.name || "Custom Profile"),
    builtin: Boolean(p.builtin),
    taxonomyMode: p.taxonomyMode === "strict_profile" ? "strict_profile" : "job_labels",
    validClasses: Array.isArray(p.validClasses) ? p.validClasses.map(String).filter(Boolean) : [],
    labelMapping: mergedMapping,
    rules,
    enableDistanceRule: Boolean(rules.R2_DISTANCE),
    enableSizeRefRule: Boolean(rules.R3_SIZE_REF),
    enableBarrierAxisRule: Boolean(rules.R4_BARRIER_AXIS),
    warnShapeInsteadOfTrack: Boolean(rules.R9_WARN_SHAPE),
    barrierLabelName:
      p.barrierLabelName !== undefined
        ? String(p.barrierLabelName || "")
        : mergedMapping.Barrier || "",
    classSpecs: normSpecs,
    sizeReferenceRows,
    checklistItems,
    issueTypes,
    sensorConfig,
    qaThresholds: {
      ...base.qaThresholds,
      ...(p.qaThresholds || {})
    }
  };
}

/**
 * Tạo mới một Profile khớp hoàn toàn với danh sách nhãn thực tế của Job CVAT đang mở
 */
export function createProfileFromJobLabels(name, jobLabels = []) {
  const cleanLabels = (jobLabels || []).map((l) => String(l || "").trim()).filter(Boolean);
  const mapping = autoMapLabelsToJob(cleanLabels);
  const classSpecs = {};
  const sizeRows = [];
  let hasBarrier = false;

  for (const lbl of cleanLabels) {
    const spec = inferClassSpecFromLabelName(lbl);
    classSpecs[lbl] = spec;
    if (/barrier|barricade|guardrail|rao_chan/i.test(lbl)) {
      hasBarrier = true;
    }
    if (spec.L && spec.W && spec.H) {
      sizeRows.push({
        subtype: `${lbl} (${spec.group})`,
        className: lbl,
        L: [...spec.L],
        W: [...spec.W],
        H: [...spec.H]
      });
    }
  }

  const id = `job_profile_${Date.now()}`;
  return normalizeProfile(
    {
      id,
      name: name || `Task Profile (${cleanLabels.length} class)`,
      builtin: false,
      taxonomyMode: "strict_profile",
      validClasses: cleanLabels,
      labelMapping: mapping,
      barrierLabelName: hasBarrier ? mapping.Barrier || "Barrier" : "",
      rules: {
        ...DEFAULT_RULE_TOGGLES,
        R2_DISTANCE: true,
        R3_SIZE_REF: sizeRows.length > 0,
        R4_BARRIER_AXIS: hasBarrier,
        R9_WARN_SHAPE: true
      },
      classSpecs,
      sizeReferenceRows: sizeRows,
      checklistItems: [
        {
          id: "chk_job_1",
          text: `Kiểm tra đúng ${cleanLabels.length} class của Task (${cleanLabels.join(", ")}); không gán nhầm class.`
        },
        ...GENERIC_CHECKLIST_ITEMS.slice(1)
      ]
    },
    id
  );
}

/**
 * Chuyển bảng tham chiếu kích thước hiển thị ở Tab 3 theo Profile + Job hiện tại
 */
export function getActiveSizeReferenceRows(profile, jobLabels = [], showAllFallback = false) {
  const p = normalizeProfile(profile || BUILTIN_PROFILES.guideline_v1_ai20k);
  const barrierRelevant = isBarrierRelevantForContext(p, jobLabels);

  let rows = [];
  if (Array.isArray(p.sizeReferenceRows) && p.sizeReferenceRows.length > 0) {
    rows = p.sizeReferenceRows;
  } else if (p.classSpecs && Object.keys(p.classSpecs).length > 0) {
    rows = Object.values(p.classSpecs)
      .filter((s) => s.L && s.W && s.H && !s.noSizeRef)
      .map((s) => ({
        subtype: `${s.className} (${s.group || "Custom"})`,
        className: s.className,
        L: s.L,
        W: s.W,
        H: s.H
      }));
  } else if (showAllFallback) {
    rows = SIZE_REFERENCE_SUBTYPES;
  } else if (Array.isArray(jobLabels) && jobLabels.length > 0) {
    rows = jobLabels
      .map((lbl) => inferClassSpecFromLabelName(lbl))
      .filter((s) => s.L && s.W && s.H)
      .map((s) => ({
        subtype: `${s.className} (Gợi ý theo tên nhãn)`,
        className: s.className,
        L: s.L,
        W: s.W,
        H: s.H
      }));
  } else {
    rows = SIZE_REFERENCE_SUBTYPES;
  }

  if (!showAllFallback && !barrierRelevant) {
    const bName = String(p.barrierLabelName || "Barrier").toLowerCase();
    rows = rows.filter(
      (r) =>
        String(r.className || "").toLowerCase() !== bName &&
        String(r.className || "").toLowerCase() !== "barrier"
    );
  }

  if (!showAllFallback && Array.isArray(jobLabels) && jobLabels.length > 0 && p.id === "guideline_v1_ai20k") {
    const jobSet = new Set(jobLabels.map((l) => String(l || "").toLowerCase()));
    const filtered = rows.filter((r) => jobSet.has(String(r.className || "").toLowerCase()));
    if (filtered.length > 0) rows = filtered;
  }

  return rows;
}

/**
 * Tạo cấu hình settings hoàn chỉnh truyền vào runMetadataQA() & runPcdFrameQA() từ Profile + Job hiện tại
 */
export function buildEffectiveQaSettings(baseSettings, profile, jobLabels = []) {
  const p = normalizeProfile(profile || BUILTIN_PROFILES.guideline_v1_ai20k);
  const barrierRelevant = isBarrierRelevantForContext(p, jobLabels);

  const effectiveSpecs = { ...(p.classSpecs || {}) };
  // Bổ sung thông số suy luận nhẹ cho các nhãn mới của Job chưa khai báo trong classSpecs
  for (const lbl of jobLabels || []) {
    const key = String(lbl || "").trim();
    if (key && !effectiveSpecs[key]) {
      const lower = key.toLowerCase();
      const existsCaseInsensitive = Object.keys(effectiveSpecs).some((k) => k.toLowerCase() === lower);
      if (!existsCaseInsensitive) {
        effectiveSpecs[key] = inferClassSpecFromLabelName(key);
      }
    }
  }

  const effectiveRules = {
    ...DEFAULT_RULE_TOGGLES,
    ...(baseSettings?.rules || {}),
    ...(p.rules || {}),
    R4_BARRIER_AXIS: Boolean(p.rules?.R4_BARRIER_AXIS && barrierRelevant)
  };

  return {
    ...DEFAULT_SETTINGS,
    ...baseSettings,
    ...p.qaThresholds,
    ...(p.sensorConfig || {}),
    profileId: p.id,
    profileName: p.name,
    taxonomyMode: p.taxonomyMode,
    validClasses:
      p.taxonomyMode === "job_labels"
        ? [...(jobLabels || [])]
        : p.validClasses.length > 0
          ? [...p.validClasses]
          : [...(jobLabels || [])],
    rules: effectiveRules,
    enableDistanceRule: Boolean(effectiveRules.R2_DISTANCE),
    maxDistanceByGroup: {
      Vehicle: Number(p.maxDistanceVehicle) || 50,
      VRU: Number(p.maxDistanceVru) || 40,
      Static: Number(p.maxDistanceStatic) || 30
    },
    maxDistanceDefault: Number(p.maxDistanceDefault) || 50,
    enableSizeRefRule: Boolean(effectiveRules.R3_SIZE_REF),
    enableBarrierAxisRule: Boolean(effectiveRules.R4_BARRIER_AXIS),
    barrierLabelName: p.barrierLabelName || "",
    barrierRelevant,
    defaultObjectType: p.defaultObjectType === "shape" ? "shape" : "track",
    warnShapeInsteadOfTrack: Boolean(effectiveRules.R9_WARN_SHAPE),
    classSpecs: effectiveSpecs,
    labelMapping: { ...p.labelMapping }
  };
}

/**
 * Tạo gói cấu hình gửi xuống Python Agent (/frame & /tracks) theo Profile hiện tại
 */
export function buildAgentGuidelinePayload(profile, jobLabels = []) {
  const p = normalizeProfile(profile || BUILTIN_PROFILES.guideline_v1_ai20k);
  const splitTwoWheels = shouldSplitTwoWheels(p.labelMapping);
  const enabledClasses = getEnabledCanonicalClasses(p.labelMapping);

  // Xây bảng kích thước và khoảng cách theo canonical class gửi xuống Agent
  const classSizeRanges = {};
  const maxDistanceByClass = {};
  for (const item of CANONICAL_AI_CLASSES) {
    const mappedLabel = String(p.labelMapping?.[item.key] || "").trim();
    if (!mappedLabel) continue;
    const spec =
      p.classSpecs?.[mappedLabel] ||
      Object.values(p.classSpecs || {}).find(
        (s) => String(s.className || "").toLowerCase() === mappedLabel.toLowerCase()
      );
    const canonKey =
      !splitTwoWheels && (item.key === "Motorcycle" || item.key === "Bicycle")
        ? "Two_wheels"
        : item.key;
    if (spec) {
      if (spec.L && spec.W && spec.H && !spec.noSizeRef) {
        classSizeRanges[canonKey] = [spec.L, spec.W, spec.H];
      }
      if (Number.isFinite(spec.maxDistance) && spec.maxDistance > 0) {
        maxDistanceByClass[canonKey] = Number(spec.maxDistance);
      }
    }
  }

  const maxDistanceByGroup = p.rules?.R2_DISTANCE
    ? {
        Vehicle: Number(p.maxDistanceVehicle) || 50,
        VRU: Number(p.maxDistanceVru) || 40,
        Static: Number(p.maxDistanceStatic) || 30
      }
    : { Vehicle: 120, VRU: 100, Static: 100 };

  return {
    enabled_classes: enabledClasses,
    split_two_wheels: splitTwoWheels,
    merge_rider: Boolean(p.sensorConfig?.mergeRiderWithTwoWheels !== false),
    enable_size_clamp: Boolean(p.rules?.R3_SIZE_REF),
    class_size_ranges: Object.keys(classSizeRanges).length > 0 ? classSizeRanges : null,
    max_distance_by_group: maxDistanceByGroup,
    max_distance_by_class: Object.keys(maxDistanceByClass).length > 0 ? maxDistanceByClass : null,
    frame_dt: Number(p.sensorConfig?.frameIntervalSec) || 0.5,
    max_frame_gap: Number(p.sensorConfig?.maxFrameGap) || 4,
    occluded_min_points:
      p.sensorConfig?.occludedMinPoints != null ? Number(p.sensorConfig.occludedMinPoints) : 5,
    lock_track_size: Boolean(p.rules?.R6_SIZE_CONSISTENCY !== false),
    barrier_relevant: isBarrierRelevantForContext(p, jobLabels)
  };
}

/**
 * Áp dụng Label Mapping lên danh sách Track từ Fusion Agent trước khi hiển thị & chèn vào CVAT
 */
export function applyLabelMappingToTracks(tracks, labelMapping) {
  if (!Array.isArray(tracks) || !labelMapping) return tracks || [];
  const mapped = [];
  for (const t of tracks) {
    const canon = t.canonicalClass || t.className;
    let target = null;
    if (Object.prototype.hasOwnProperty.call(labelMapping, canon)) {
      target = String(labelMapping[canon] || "").trim();
    } else if (canon === "Two_wheels") {
      target = String(labelMapping.Motorcycle || labelMapping.Bicycle || "").trim();
    }
    if (target !== null) {
      if (!target || target === "__IGNORE__") continue;
      mapped.push({
        ...t,
        canonicalClass: canon,
        className: target
      });
    } else {
      mapped.push({ ...t, canonicalClass: canon });
    }
  }
  return mapped;
}
