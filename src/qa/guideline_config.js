/**
 * Cấu hình mặc định & Preset Guideline V1 (3D_Cuboid_Annotation_Guideline_V1.pdf)
 * Tất cả các module QA Engine (V1/V1.5), Pre-label, Safe Actions (V2) và Issue Formatter (V2.5)
 * đều đọc qua cấu hình hiệu lực của Profile hiện hành, chỉ dùng các hằng số dưới đây làm preset/fallback.
 */

export const VALID_CLASSES = [
  "Car",
  "Truck",
  "Trailer",
  "Bus",
  "Construction_vehicle",
  "Pedestrian",
  "Two_wheels",
  "Barrier"
];

export const CLASS_GROUPS = {
  Car: "Vehicle",
  Truck: "Vehicle",
  Trailer: "Vehicle",
  Bus: "Vehicle",
  Construction_vehicle: "Vehicle",
  Pedestrian: "VRU",
  Two_wheels: "VRU",
  Bicycle: "VRU",
  Motorcycle: "VRU",
  Barrier: "Static",
  Traffic_cone: "Static"
};

/**
 * Phạm vi khoảng cách tối đa mặc định (m) theo mục 2 trang 2 Guideline V1
 */
export const GROUP_MAX_DISTANCE = {
  Vehicle: 50,
  VRU: 40,
  Static: 30
};

/**
 * Sai số kích thước cho phép mặc định theo nhóm
 */
export const GROUP_SIZE_TOLERANCE = {
  Vehicle: 0.05, // ±5%
  VRU: 0.10,     // ±10%
  Static: 0.10   // ±10%
};

/**
 * Bảng tham chiếu kích thước chi tiết Preset Guideline V1 (m)
 */
export const SIZE_REFERENCE_SUBTYPES = [
  { subtype: "Sedan, hatchback", className: "Car", L: [3.8, 5.0], W: [1.7, 1.9], H: [1.4, 1.6] },
  { subtype: "SUV, minivan, van chở người", className: "Car", L: [4.4, 5.3], W: [1.8, 2.1], H: [1.65, 2.1] },
  { subtype: "Bán tải", className: "Truck", L: [5.2, 6.2], W: [1.85, 2.1], H: [1.8, 2.0] },
  { subtype: "Xe tải hộp, xe tải nhẹ", className: "Truck", L: [5.5, 9.0], W: [2.0, 2.55], H: [2.4, 3.5] },
  { subtype: "Xe tải nặng, đầu kéo", className: "Truck", L: [6.0, 12.0], W: [2.4, 2.6], H: [3.0, 4.0] },
  { subtype: "Rơ-moóc sau đầu kéo", className: "Trailer", L: [8.0, 16.0], W: [2.5, 2.6], H: [3.5, 4.1] },
  { subtype: "Rơ-moóc nhỏ (sau xe con)", className: "Trailer", L: [2.0, 5.0], W: [1.5, 2.2], H: [0.8, 2.5] },
  { subtype: "Xe buýt thường", className: "Bus", L: [10.0, 12.5], W: [2.5, 2.6], H: [3.0, 3.5] },
  { subtype: "Xe buýt nối toa", className: "Bus", L: [17.0, 18.5], W: [2.5, 2.6], H: [3.0, 3.4] },
  { subtype: "Máy xúc, cần cẩu, xe lu", className: "Construction_vehicle", L: [4.0, 12.0], W: [2.0, 3.2], H: [2.5, 4.0] },
  { subtype: "Người lớn", className: "Pedestrian", L: [0.4, 0.9], W: [0.5, 0.9], H: [1.5, 1.95] },
  { subtype: "Trẻ em", className: "Pedestrian", L: [0.3, 0.6], W: [0.3, 0.6], H: [0.9, 1.4] },
  { subtype: "Xe đạp có người", className: "Two_wheels", L: [1.6, 1.9], W: [0.5, 0.7], H: [1.6, 1.9] },
  { subtype: "Xe máy có người", className: "Two_wheels", L: [1.8, 2.3], W: [0.7, 0.95], H: [1.4, 1.8] },
  { subtype: "Xe máy / xe đạp dựng", className: "Two_wheels", L: [1.6, 2.2], W: [0.5, 0.9], H: [1.0, 1.3] },
  { subtype: "Rào bê tông, rào nước", className: "Barrier", L: [1.0, 3.0], W: [0.4, 0.7], H: [0.8, 1.1] }
];

/**
 * Hợp (Union) khoảng kích thước mặc định cho từng class phổ biến (bao gồm cả Bicycle/Motorcycle/Traffic_cone tách lẻ)
 */
export const CLASS_SIZE_UNION = {
  ...VALID_CLASSES.reduce((acc, cls) => {
    const rows = SIZE_REFERENCE_SUBTYPES.filter((r) => r.className === cls);
    acc[cls] = {
      L: [Math.min(...rows.map((r) => r.L[0])), Math.max(...rows.map((r) => r.L[1]))],
      W: [Math.min(...rows.map((r) => r.W[0])), Math.max(...rows.map((r) => r.W[1]))],
      H: [Math.min(...rows.map((r) => r.H[0])), Math.max(...rows.map((r) => r.H[1]))]
    };
    return acc;
  }, {}),
  Bicycle: { L: [1.5, 2.0], W: [0.45, 0.8], H: [1.0, 1.9] },
  Motorcycle: { L: [1.7, 2.4], W: [0.6, 1.0], H: [1.0, 1.85] },
  Traffic_cone: { L: [0.25, 0.65], W: [0.25, 0.65], H: [0.45, 1.15] }
};

/**
 * Danh mục đầy đủ 15 công tắc Luật QA (9 nhóm Metadata V1 + 5 nhóm Point Cloud V1.5)
 */
export const QA_RULE_CATALOG = [
  { key: "R1_TAXONOMY", stage: "V1", label: "R1 · Kiểm tra đúng bộ nhãn (Taxonomy)" },
  { key: "R2_DISTANCE", stage: "V1", label: "R2 · Giới hạn khoảng cách tối đa (m)" },
  { key: "R3_SIZE_REF", stage: "V1", label: "R3 · Đối chiếu bảng kích thước chuẩn (L×W×H)" },
  { key: "R4_BARRIER_AXIS", stage: "V1", label: "R4 · Kiểm tra trục hướng Barrier / vật dẹt" },
  { key: "R5_PITCH_ROLL", stage: "V1", label: "R5 · Cấm nghiêng Pitch / Roll (Cuboid 7-DoF)" },
  { key: "R6_SIZE_CONSISTENCY", stage: "V1", label: "R6 · Đồng nhất kích thước trong Track (±%)" },
  { key: "R7_OCCLUDED_OUTSIDE", stage: "V1", label: "R7 · Xung đột Occluded+Outside & chuỗi che khuất" },
  { key: "R8_MOTION_YAW", stage: "V1", label: "R8 · Nhảy góc Yaw, tốc độ & đột biến quỹ đạo" },
  { key: "R9_TRACK_CONTINUITY", stage: "V1", label: "R9 · Tái sử dụng Track sau outside & Track 1 frame" },
  { key: "R9_WARN_SHAPE", stage: "V1", label: "R9s · Cảnh báo vẽ Shape lẻ thay vì Track" },
  { key: "PCD1_GROUND", stage: "PCD", label: "PCD1 · Box lơ lửng / chìm dưới mặt đất" },
  { key: "PCD2_ZERO_POINTS", stage: "PCD", label: "PCD2 · Box 0 điểm LiDAR nhưng chưa bật Occluded" },
  { key: "PCD3_EMPTY_MARGIN", stage: "PCD", label: "PCD3 · Box dư khoảng trống quá rộng" },
  { key: "PCD4_SPARSE_POINTS", stage: "PCD", label: "PCD4 · Cụm điểm quá thưa (<15% thể tích)" },
  { key: "PCD5_MISSED_HINTS", stage: "PCD", label: "PCD5 · Gợi ý cụm điểm nghi bỏ sót đối tượng" }
];

export const DEFAULT_RULE_TOGGLES = QA_RULE_CATALOG.reduce((acc, item) => {
  acc[item.key] = true;
  return acc;
}, {});

/**
 * 5 loại Mentor Issue chuẩn mặc định
 */
export const MENTOR_ISSUE_TYPES = [
  {
    code: "UNCERTAIN_CLASS",
    label: "UNCERTAIN_CLASS — Không chắc class",
    example: "Car hay Truck (van chở hàng); Truck hay Construction_vehicle"
  },
  {
    code: "UNCERTAIN_SCOPE",
    label: "UNCERTAIN_SCOPE — Không chắc phạm vi / quá thưa điểm",
    example: "Vật quanh ngưỡng khoảng cách giới hạn; vật quá thưa điểm"
  },
  {
    code: "ATTRIBUTE_CHECK",
    label: "ATTRIBUTE_CHECK — Không chắc thuộc tính",
    example: "Không chắc occluded / truncated / outside"
  },
  {
    code: "UNCERTAIN_TRACK",
    label: "UNCERTAIN_TRACK — Không chắc danh tính Track ID",
    example: "Không chắc cùng một vật sau khi bị che khuất"
  },
  {
    code: "LOW_VISIBILITY",
    label: "LOW_VISIBILITY — Khó quan sát",
    example: "Vật khó quan sát do đêm, mưa, ngược sáng, quá xa hoặc quá ít điểm LiDAR"
  }
];

/**
 * 7 bước Checklist chất lượng mặc định (Preset Guideline V1)
 */
export const SUBMIT_CHECKLIST_ITEMS = [
  {
    id: "chk_1_taxonomy",
    text: "Chỉ 8 class, gán đúng class theo taxonomy; không tự tạo hoặc đổi tên class."
  },
  {
    id: "chk_2_cuboid",
    text: "Cuboid bao trọn điểm, đáy chạm đất, mũi tên đỏ chỉ đầu xe; đã kiểm ở Top, Side, Front và camera. Sai số trong phạm vi cho phép."
  },
  {
    id: "chk_3_occluded",
    text: "Occluded gán đúng; occluded và outside không cùng bật."
  },
  {
    id: "chk_4_consistency",
    text: "Kích thước lấy từ keyframe dày điểm nhất, khóa ±5%; tâm và yaw đổi mượt."
  },
  {
    id: "chk_5_track_id",
    text: "track_id không tái sử dụng; nghi ngờ danh tính thì kết thúc track cũ, tạo track mới."
  },
  {
    id: "chk_6_mentor",
    text: "Các trường hợp không chắc chắn đã tạo issue / comment và raise lên mentor."
  },
  {
    id: "chk_7_double_check",
    text: "Đã check 02 lần toàn bộ dữ liệu trước khi save."
  }
];

/**
 * Checklist trung tính dùng cho các Profile đa task / tự động theo Job
 */
export const GENERIC_CHECKLIST_ITEMS = [
  {
    id: "gen_1_taxonomy",
    text: "Gán đúng nhãn và thuộc tính theo Guideline của Task hiện tại."
  },
  {
    id: "gen_2_cuboid_fit",
    text: "Cuboid ôm sát điểm LiDAR, đúng hướng mũi tên đầu vật thể và chạm đúng mặt đất trên Top/Side/Front + Camera."
  },
  {
    id: "gen_3_continuity",
    text: "Kiểm tra tính liên tục giữa các frame (kích thước, quỹ đạo, cờ occluded / outside hợp lệ)."
  },
  {
    id: "gen_4_edge_cases",
    text: "Ghi chú / tạo Issue cho các tình huống mờ, bị che khuất nặng hoặc chưa rõ quy tắc."
  },
  {
    id: "gen_5_final_review",
    text: "Quét lại QA Audit và rà soát toàn bộ chuỗi frame trước khi bấm Save / Submit."
  }
];

/**
 * Cấu hình mặc định toàn cục
 */
export const DEFAULT_SETTINGS = {
  barrierAxisMode: "length_is_x",

  occludedThreshold: 6,
  occludedOperator: ">",
  occludedCountMode: "frame",

  maxPitchRollDeg: 5.0,
  allowPitchRoll: false,

  trackSizeTolerancePct: 5.0,

  coordinateFrame: "lidar_centric",
  frameIntervalSec: 0.5,
  maxYawJumpVehicleDeg: 45,
  maxYawJumpPedestrianDeg: 135,
  ignorePedestrianYaw: true,
  maxSecondOrderAccelM: 3.5,
  maxWorldSpeedMps: 38.0,

  distanceBoundaryMarginM: 2.0,

  mergeRiderWithTwoWheels: true,
  occludedMinPoints: 5,
  maxFrameGap: 4,

  pcdUrlTemplate: "",
  pcdGroundFloatThresholdM: 0.22,
  pcdGroundSinkThresholdM: 0.20,
  pcdEmptyMarginThresholdM: 0.45,
  pcdMinSparseCoverage: 0.15,
  pcdMaxHintDistanceM: 45.0,

  rules: { ...DEFAULT_RULE_TOGGLES },

  enableV2Assist: false,
  enableDirectCvatIssuePost: false,

  enableV3Agent: false,
  v3AgentUrl: "http://127.0.0.1:8765"
};

/**
 * Tìm thông số classSpec theo tên class (không phân biệt hoa thường)
 */
export function findClassSpec(className, settings = {}) {
  const specs = settings?.classSpecs;
  if (!specs || typeof specs !== "object") return null;
  if (specs[className]) return specs[className];
  const lower = String(className || "").trim().toLowerCase();
  for (const [k, v] of Object.entries(specs)) {
    if (k.toLowerCase() === lower) return v;
  }
  return null;
}

/**
 * Lấy khoảng kích thước (dx, dy, dz) áp dụng cho 1 class,
 * ưu tiên đọc từ settings.classSpecs (của Profile hiện tại), sau đó mới fallback về CLASS_SIZE_UNION.
 */
export function getExpectedDxDyDzRange(className, settings = DEFAULT_SETTINGS) {
  const customSpec = findClassSpec(className, settings);
  let base = null;
  let tolerance = null;
  let group = null;

  if (customSpec) {
    group = customSpec.group || null;
    if (Number.isFinite(customSpec.sizeTolerance)) {
      tolerance = Number(customSpec.sizeTolerance);
    }
    if (
      Array.isArray(customSpec.L) && customSpec.L.length === 2 &&
      Array.isArray(customSpec.W) && customSpec.W.length === 2 &&
      Array.isArray(customSpec.H) && customSpec.H.length === 2
    ) {
      base = { L: customSpec.L, W: customSpec.W, H: customSpec.H };
    } else if (customSpec.noSizeRef) {
      return null;
    }
  }

  if (!base) {
    base = CLASS_SIZE_UNION[className] || null;
  }
  if (!base) return null;

  const barrierName = String(settings?.barrierLabelName || "Barrier").toLowerCase();
  const isBarrierLike =
    String(className || "").toLowerCase() === barrierName ||
    customSpec?.axisMode === "barrier_sync";
  const effectiveAxisMode =
    customSpec?.axisMode && customSpec.axisMode !== "barrier_sync"
      ? customSpec.axisMode
      : isBarrierLike
        ? settings?.barrierAxisMode || "length_is_x"
        : "length_is_x";

  if (effectiveAxisMode === "width_is_x") {
    return {
      dx: base.W,
      dy: base.L,
      dz: base.H,
      swapped: true,
      tolerance,
      group
    };
  }
  return {
    dx: base.L,
    dy: base.W,
    dz: base.H,
    swapped: false,
    tolerance,
    group
  };
}
