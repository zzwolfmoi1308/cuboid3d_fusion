/**
 * Smart Guideline Importer & Template Generator — CVAT 3D Fusion Suite
 * Cho phép người dùng upload / dán guideline bất kỳ lúc nào theo từng Profile mà không cần sửa code:
 *  1. File JSON (cả Profile JSON đầy đủ lẫn Guideline JSON rút gọn dễ viết)
 *  2. File CSV / TSV (bảng nhãn, nhóm, khoảng cách, L/W/H, dung sai)
 *  3. File Markdown (.md) / Text (.txt) hoặc văn bản copy-paste trực tiếp từ PDF/Word/Notion
 */

import {
  DEFAULT_RULE_TOGGLES,
  GENERIC_CHECKLIST_ITEMS,
  MENTOR_ISSUE_TYPES
} from "./guideline_config.js";
import {
  autoMapLabelsToJob,
  inferClassSpecFromLabelName,
  normalizeProfile
} from "./profile_manager.js";

function parseRangePair(str) {
  if (Array.isArray(str) && str.length >= 2) {
    const a = Number(str[0]);
    const b = Number(str[1]);
    if (Number.isFinite(a) && Number.isFinite(b) && a > 0 && b >= a) return [a, b];
  }
  const s = String(str ?? "").trim().replace(/,/g, ".");
  if (!s) return null;
  const m = s.match(/(\d+(?:\.\d+)?)\s*(?:[-–—~]|to|đến|->)\s*(\d+(?:\.\d+)?)/i);
  if (m) {
    const a = parseFloat(m[1]);
    const b = parseFloat(m[2]);
    if (Number.isFinite(a) && Number.isFinite(b) && a > 0 && b >= a) return [a, b];
  }
  const single = s.match(/^(\d+(?:\.\d+)?)\s*m?$/i);
  if (single) {
    const v = parseFloat(single[1]);
    if (Number.isFinite(v) && v > 0) return [Number((v * 0.85).toFixed(2)), Number((v * 1.15).toFixed(2))];
  }
  return null;
}

/**
 * Phân tích Guideline từ định dạng JSON (hỗ trợ cả Full Profile JSON và Simplified Guideline JSON)
 */
function parseGuidelineJson(parsed, fallbackName = "Imported Guideline") {
  // Trường hợp 1: Là Full Profile JSON đã chuẩn hóa sẵn
  if (parsed.labelMapping && (parsed.qaThresholds || parsed.rules || parsed.classSpecs)) {
    const id = `imported_${Date.now()}`;
    const profile = normalizeProfile({ ...parsed, id, builtin: false }, id);
    return {
      profile,
      summary: `Đã nạp Profile JSON "${profile.name}" (${Object.keys(profile.classSpecs || {}).length} class specs).`
    };
  }

  // Trường hợp 2: Simplified Guideline JSON thân thiện cho người dùng / LLM
  // Ví dụ: { name, classes: [{ name, group, maxDistance, L: [3.8,5.2], W: [1.7,2.1], H: [1.4,2.0], tolerancePct: 5, ignoreYaw: false }], rules: {...}, checklist: [...] }
  const rawClasses = Array.isArray(parsed.classes)
    ? parsed.classes
    : parsed.classes && typeof parsed.classes === "object"
      ? Object.entries(parsed.classes).map(([k, v]) => ({ name: k, ...(typeof v === "object" ? v : {}) }))
      : [];

  const distGroup = parsed.maxDistanceByGroup || parsed.distances || {};
  const validClasses = [];
  const classSpecs = {};
  const sizeRows = [];

  for (const item of rawClasses) {
    const clsName = String(item.name || item.className || item.label || "").trim();
    if (!clsName) continue;
    if (!validClasses.includes(clsName)) validClasses.push(clsName);

    const base = inferClassSpecFromLabelName(clsName);
    const grp = item.group || base.group;
    const groupDefaultDist = Number(distGroup[grp]) || base.maxDistance;
    const L = parseRangePair(item.L ?? item.length) || base.L;
    const W = parseRangePair(item.W ?? item.width) || base.W;
    const H = parseRangePair(item.H ?? item.height) || base.H;
    const tol =
      item.sizeTolerance != null
        ? Number(item.sizeTolerance)
        : item.tolerancePct != null
          ? Number(item.tolerancePct) / 100
          : base.sizeTolerance;
    const maxDist = Number(item.maxDistance ?? item.distance ?? groupDefaultDist);

    classSpecs[clsName] = {
      className: clsName,
      group: grp,
      maxDistance: Number.isFinite(maxDist) && maxDist > 0 ? maxDist : groupDefaultDist,
      L,
      W,
      H,
      noSizeRef: Boolean(item.noSizeRef || (!L && !W && !H)),
      sizeTolerance: Number.isFinite(tol) && tol > 0 ? tol : base.sizeTolerance,
      ignoreYaw: typeof item.ignoreYaw === "boolean" ? item.ignoreYaw : base.ignoreYaw,
      axisMode: item.axisMode || base.axisMode
    };

    if (L && W && H) {
      sizeRows.push({
        subtype: String(item.subtype || `${clsName} (${classSpecs[clsName].group})`),
        className: clsName,
        L,
        W,
        H
      });
    }
  }

  if (Array.isArray(parsed.sizeReference)) {
    sizeRows.length = 0;
    for (const r of parsed.sizeReference) {
      const cName = String(r.className || r.class || "").trim();
      const L = parseRangePair(r.L);
      const W = parseRangePair(r.W);
      const H = parseRangePair(r.H);
      if (cName && L && W && H) {
        sizeRows.push({ subtype: String(r.subtype || cName), className: cName, L, W, H });
      }
    }
  }

  const hasBarrier = validClasses.some((c) => /barrier|barricade|guardrail|rao_chan/i.test(c));
  const mapping = autoMapLabelsToJob(validClasses);
  if (parsed.labelMapping && typeof parsed.labelMapping === "object") {
    Object.assign(mapping, parsed.labelMapping);
  }

  const rules = {
    ...DEFAULT_RULE_TOGGLES,
    R2_DISTANCE: validClasses.length > 0 || Boolean(parsed.maxDistanceByGroup),
    R3_SIZE_REF: sizeRows.length > 0,
    R4_BARRIER_AXIS: hasBarrier,
    ...(parsed.rules || {})
  };

  const id = `guideline_${Date.now()}`;
  const profile = normalizeProfile(
    {
      id,
      name: String(parsed.name || parsed.title || fallbackName).trim(),
      builtin: false,
      taxonomyMode: validClasses.length > 0 ? "strict_profile" : "job_labels",
      validClasses,
      labelMapping: mapping,
      barrierLabelName: hasBarrier ? mapping.Barrier || "Barrier" : "",
      rules,
      maxDistanceVehicle: Number(distGroup.Vehicle) || 50,
      maxDistanceVru: Number(distGroup.VRU) || 40,
      maxDistanceStatic: Number(distGroup.Static) || 30,
      defaultObjectType: parsed.defaultObjectType === "shape" ? "shape" : "track",
      classSpecs,
      sizeReferenceRows: sizeRows,
      checklistItems: Array.isArray(parsed.checklist) ? parsed.checklist : GENERIC_CHECKLIST_ITEMS,
      issueTypes: Array.isArray(parsed.issueTypes) ? parsed.issueTypes : MENTOR_ISSUE_TYPES,
      sensorConfig: parsed.sensor || parsed.sensorConfig || {},
      qaThresholds: parsed.qaThresholds || parsed.thresholds || {}
    },
    id
  );

  return {
    profile,
    summary: `Đã nạp Guideline JSON "${profile.name}": ${validClasses.length} nhãn, ${sizeRows.length} dòng kích thước.`
  };
}

/**
 * Phân tích bảng CSV / TSV thành Profile
 */
function parseGuidelineCsv(text, fallbackName = "CSV Guideline") {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("#"));
  if (lines.length < 2) {
    throw new Error("File CSV cần có ít nhất 1 dòng tiêu đề và 1 dòng dữ liệu.");
  }

  const sep = lines[0].includes("\t") ? "\t" : lines[0].includes(";") ? ";" : ",";
  const splitRow = (row) => row.split(sep).map((c) => c.trim().replace(/^["']|["']$/g, ""));
  const headers = splitRow(lines[0]).map((h) => h.toLowerCase().replace(/[\s\-]+/g, "_"));

  const idx = (names) => headers.findIndex((h) => names.includes(h));
  const iClass = idx(["class", "classname", "class_name", "label", "nhan"]);
  const iSub = idx(["subtype", "sub_type", "type", "loai", "description"]);
  const iGroup = idx(["group", "nhom", "category"]);
  const iDist = idx(["max_distance", "maxdistance", "distance", "range", "khoang_cach"]);
  const iLMin = idx(["l_min", "lmin", "length_min"]);
  const iLMax = idx(["l_max", "lmax", "length_max"]);
  const iL = idx(["l", "length", "dai"]);
  const iWMin = idx(["w_min", "wmin", "width_min"]);
  const iWMax = idx(["w_max", "wmax", "width_max"]);
  const iW = idx(["w", "width", "rong"]);
  const iHMin = idx(["h_min", "hmin", "height_min"]);
  const iHMax = idx(["h_max", "hmax", "height_max"]);
  const iH = idx(["h", "height", "cao"]);
  const iTol = idx(["tolerance_pct", "tol_pct", "tolerance", "sai_so"]);
  const iIgnoreYaw = idx(["ignore_yaw", "ignoreyaw"]);

  if (iClass < 0) {
    throw new Error("Không tìm thấy cột 'class' (hoặc 'label') trong dòng tiêu đề CSV.");
  }

  const classesMap = new Map();
  const sizeReference = [];

  for (let r = 1; r < lines.length; r++) {
    const cols = splitRow(lines[r]);
    const clsName = cols[iClass];
    if (!clsName) continue;

    const L =
      iLMin >= 0 && iLMax >= 0
        ? parseRangePair([cols[iLMin], cols[iLMax]])
        : iL >= 0
          ? parseRangePair(cols[iL])
          : null;
    const W =
      iWMin >= 0 && iWMax >= 0
        ? parseRangePair([cols[iWMin], cols[iWMax]])
        : iW >= 0
          ? parseRangePair(cols[iW])
          : null;
    const H =
      iHMin >= 0 && iHMax >= 0
        ? parseRangePair([cols[iHMin], cols[iHMax]])
        : iH >= 0
          ? parseRangePair(cols[iH])
          : null;

    const sub = (iSub >= 0 && cols[iSub]) || clsName;
    if (L && W && H) {
      sizeReference.push({ subtype: sub, className: clsName, L, W, H });
    }

    const prev = classesMap.get(clsName) || { name: clsName, L: null, W: null, H: null };
    if (iGroup >= 0 && cols[iGroup]) prev.group = cols[iGroup];
    if (iDist >= 0 && Number(cols[iDist]) > 0) prev.maxDistance = Number(cols[iDist]);
    if (iTol >= 0 && Number(cols[iTol]) > 0) prev.tolerancePct = Number(cols[iTol]);
    if (iIgnoreYaw >= 0 && cols[iIgnoreYaw]) {
      prev.ignoreYaw = /^(1|true|yes|y)$/i.test(cols[iIgnoreYaw]);
    }
    const mergePair = (oldP, newP) =>
      !newP ? oldP : !oldP ? [...newP] : [Math.min(oldP[0], newP[0]), Math.max(oldP[1], newP[1])];
    prev.L = mergePair(prev.L, L);
    prev.W = mergePair(prev.W, W);
    prev.H = mergePair(prev.H, H);
    classesMap.set(clsName, prev);
  }

  return parseGuidelineJson(
    {
      name: fallbackName,
      classes: Array.from(classesMap.values()),
      sizeReference
    },
    fallbackName
  );
}

/**
 * Phân tích văn bản tự do / Markdown copy từ Guideline PDF / Notion / Word
 */
function parseGuidelineTextOrMarkdown(text, fallbackName = "Guideline Text", jobLabels = []) {
  const lines = text.split(/\r?\n/);
  const sizeReference = [];
  const classEntryMap = new Map();
  const checklist = [];

  const registerClassRange = (clsName, subtype, L, W, H, maxDist = null) => {
    const clean = clsName.trim().replace(/^[*`\-•]+|[*`]+$/g, "");
    if (!clean || clean.length > 38) return;
    if (/^(class|loại|type|nhãn|label|stt)$/i.test(clean)) return;

    const prev = classEntryMap.get(clean) || { name: clean, L: null, W: null, H: null };
    const mergePair = (a, b) => (!b ? a : !a ? [...b] : [Math.min(a[0], b[0]), Math.max(a[1], b[1])]);
    prev.L = mergePair(prev.L, L);
    prev.W = mergePair(prev.W, W);
    prev.H = mergePair(prev.H, H);
    if (maxDist && Number.isFinite(maxDist)) prev.maxDistance = maxDist;
    classEntryMap.set(clean, prev);

    if (L && W && H) {
      sizeReference.push({ subtype: subtype || clean, className: clean, L, W, H });
    }
  };

  // 1. Bắt các bảng Markdown hoặc dòng có >= 3 khoảng số L, W, H (vd: "| Car | Sedan | 3.8-5.0 | 1.7-1.9 | 1.4-1.6 |")
  const rangeRegex = /(\d+(?:[.,]\d+)?\s*(?:[-–—~]|to|đến)\s*\d+(?:[.,]\d+)?)/gi;
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;

    // Bắt dòng checklist nếu có dạng "- [ ] ..."
    const chkMatch = line.match(/^[-*]\s*\[[ xX]?\]\s+(.{8,})/);
    if (chkMatch) {
      checklist.push(chkMatch[1].trim());
    }

    if (line.includes("|") && !/^\|?[\s\-:|]+\|?$/.test(line)) {
      const cells = line
        .split("|")
        .map((c) => c.trim())
        .filter(Boolean);
      if (cells.length >= 4) {
        const pairCells = cells.map((c) => ({ raw: c, pair: parseRangePair(c) }));
        const validPairs = pairCells.filter((x) => x.pair !== null);
        if (validPairs.length >= 3) {
          const textCells = pairCells.filter((x) => x.pair === null).map((x) => x.raw);
          if (textCells.length >= 1) {
            // Tìm tên class (ưu tiên ô khớp known token hoặc ô đầu/thứ 2)
            const c1 = textCells[0];
            const c2 = textCells[1] || textCells[0];
            const isC2ClassLike = /^[A-Za-z][A-Za-z0-9_]+$/.test(c2) && /\s/.test(c1);
            const clsName = isC2ClassLike ? c2 : c1;
            const subtype = textCells.length >= 2 ? (isC2ClassLike ? c1 : c2) : clsName;
            const distCell = cells.find((c) => /^\d+\s*m$/i.test(c));
            const maxD = distCell ? parseFloat(distCell) : null;
            registerClassRange(
              clsName,
              subtype,
              validPairs[0].pair,
              validPairs[1].pair,
              validPairs[2].pair,
              maxD
            );
            continue;
          }
        }
      }
    }

    // Dòng text thường chứa tên class + 3 khoảng số L, W, H
    const matches = [...line.matchAll(rangeRegex)];
    if (matches.length >= 3) {
      const prefix = line.slice(0, matches[0].index).replace(/[:|,\-–—]+$/, "").trim();
      const tokenMatch = prefix.match(/([A-Za-z][A-Za-z0-9_]{1,30})\s*$/);
      if (tokenMatch) {
        const L = parseRangePair(matches[0][1]);
        const W = parseRangePair(matches[1][1]);
        const H = parseRangePair(matches[2][1]);
        if (L && W && H) {
          registerClassRange(tokenMatch[1], prefix, L, W, H);
        }
      }
    }
  }

  // 2. Bắt danh sách tên class được liệt kê (hoặc kết hợp với jobLabels nếu có)
  const knownCandidates = [
    "Car", "Truck", "Trailer", "Bus", "Construction_vehicle",
    "Pedestrian", "Two_wheels", "Motorcycle", "Bicycle", "Cyclist",
    "Barrier", "Traffic_cone", "Van", "Pickup", "Person", "Motorbike"
  ];
  for (const cand of [...jobLabels, ...knownCandidates]) {
    if (!cand) continue;
    const escaped = cand.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const rx = new RegExp(`\\b${escaped}\\b`, "i");
    if (rx.test(text)) {
      const existingKey = Array.from(classEntryMap.keys()).find(
        (k) => k.toLowerCase() === cand.toLowerCase()
      );
      if (!existingKey) {
        classEntryMap.set(cand, { name: cand });
      }
    }
  }

  // 3. Bắt khoảng cách theo nhóm (Vehicle / VRU / Static)
  const fullLower = text.toLowerCase();
  const findGroupDist = (pattern, fallback) => {
    const m = text.match(pattern);
    return m ? parseFloat(m[1]) : fallback;
  };
  const distVehicle = findGroupDist(/(?:vehicle|xe\s+cơ\s+giới|ô\s*tô)[^\n\d]{0,30}(\d{2,3})\s*m/i, 50);
  const distVru = findGroupDist(/(?:vru|pedestrian|người\s+đi\s+bộ)[^\n\d]{0,30}(\d{2,3})\s*m/i, 40);
  const distStatic = findGroupDist(/(?:static|barrier|vật\s+tĩnh)[^\n\d]{0,30}(\d{2,3})\s*m/i, 30);

  // 4. Suy luận công tắc Luật & Thông số Sensor từ nội dung văn bản
  const hasBarrier = Array.from(classEntryMap.keys()).some((c) =>
    /barrier|barricade|guardrail|rao_chan/i.test(c)
  );
  const mentionsShapeOnly = /single[\s-]frame|chỉ\s+vẽ\s+shape|không\s+dùng\s+track|annotation\s+type\s*:\s*shape/i.test(fullLower);
  const mentionsAllowPitchRoll = /9[\s-]?dof|cho\s+phép\s+pitch|allow\s+pitch/i.test(fullLower);
  const mentionsSeparateRider = /tách\s+riêng\s+người\s+lái|rider\s+separate|không\s+gộp\s+người/i.test(fullLower);

  let frameIntervalSec = 0.5;
  const hzMatch = text.match(/(\d+(?:\.\d+)?)\s*hz\b/i);
  if (hzMatch && parseFloat(hzMatch[1]) > 0) {
    frameIntervalSec = Number((1 / parseFloat(hzMatch[1])).toFixed(3));
  }

  const occMatch = text.match(/occluded[^\n\d]{0,25}(>=|>)\s*(\d{1,2})/i);
  const qaThresholds = {};
  if (occMatch) {
    qaThresholds.occludedOperator = occMatch[1];
    qaThresholds.occludedThreshold = parseInt(occMatch[2], 10);
  }

  const classesArr = Array.from(classEntryMap.values());
  if (classesArr.length === 0 && sizeReference.length === 0) {
    throw new Error(
      "Không tìm thấy thông tin nhãn hoặc bảng kích thước trong văn bản. Hãy dùng JSON/CSV mẫu hoặc dán phần Quy định Class & Kích thước của Guideline."
    );
  }

  return parseGuidelineJson(
    {
      name: fallbackName,
      classes: classesArr,
      sizeReference: sizeReference.length > 0 ? sizeReference : undefined,
      maxDistanceByGroup: {
        Vehicle: distVehicle,
        VRU: distVru,
        Static: distStatic
      },
      defaultObjectType: mentionsShapeOnly ? "shape" : "track",
      rules: {
        R2_DISTANCE: /\d+\s*m\b/i.test(text),
        R3_SIZE_REF: sizeReference.length > 0,
        R4_BARRIER_AXIS: hasBarrier,
        R5_PITCH_ROLL: !mentionsAllowPitchRoll,
        R9_WARN_SHAPE: !mentionsShapeOnly
      },
      sensor: {
        frameIntervalSec,
        allowPitchRoll: mentionsAllowPitchRoll,
        mergeRiderWithTwoWheels: !mentionsSeparateRider
      },
      qaThresholds,
      checklist: checklist.length >= 2 ? checklist : undefined
    },
    fallbackName
  );
}

/**
 * Hàm tổng hợp: nhận nội dung file hoặc chuỗi text bất kỳ (.json, .csv, .tsv, .md, .txt) và trả về Profile hoàn chỉnh
 */
export function importGuidelineContent(rawContent, fileName = "Custom_Guideline", jobLabels = []) {
  const text = String(rawContent || "").trim();
  if (!text) {
    throw new Error("Nội dung Guideline trống.");
  }

  const cleanName =
    String(fileName || "Custom Guideline")
      .replace(/\.[^.]+$/, "")
      .replace(/[_-]+/g, " ")
      .trim() || "Custom Guideline";

  const ext = (String(fileName || "").split(".").pop() || "").toLowerCase();

  if (ext === "json" || text.startsWith("{")) {
    const parsed = JSON.parse(text);
    return parseGuidelineJson(parsed, cleanName);
  }

  if (ext === "csv" || ext === "tsv") {
    return parseGuidelineCsv(text, cleanName);
  }

  // Tự động nhận diện CSV nếu dòng đầu có dấu phẩy và cột class/label
  const firstLine = text.split(/\r?\n/)[0] || "";
  if (/^(class|classname|label)[,;\t]/i.test(firstLine.trim())) {
    return parseGuidelineCsv(text, cleanName);
  }

  return parseGuidelineTextOrMarkdown(text, cleanName, jobLabels);
}

/**
 * Tạo file JSON mẫu (Simplified Guideline Template) dựa trên nhãn của Job đang mở để người dùng tải về điền dễ dàng
 */
export function buildGuidelineTemplateJson(jobLabels = [], profileName = "Task Guideline Mới") {
  const seedLabels =
    Array.isArray(jobLabels) && jobLabels.length > 0
      ? jobLabels
      : ["Car", "Truck", "Bus", "Pedestrian", "Motorcycle", "Bicycle"];

  const classes = seedLabels.map((lbl) => {
    const spec = inferClassSpecFromLabelName(lbl);
    return {
      name: lbl,
      group: spec.group,
      maxDistance: spec.maxDistance,
      L: spec.L || [1.0, 5.0],
      W: spec.W || [0.5, 2.0],
      H: spec.H || [1.0, 2.0],
      tolerancePct: Math.round((spec.sizeTolerance || 0.1) * 100),
      ignoreYaw: Boolean(spec.ignoreYaw)
    };
  });

  const hasBarrier = seedLabels.some((l) => /barrier|barricade/i.test(l));

  return {
    name: profileName,
    defaultObjectType: "track",
    maxDistanceByGroup: {
      Vehicle: 50,
      VRU: 40,
      Static: 30
    },
    sensor: {
      frameIntervalSec: 0.5,
      allowPitchRoll: false,
      mergeRiderWithTwoWheels: true,
      occludedMinPoints: 5
    },
    rules: {
      ...DEFAULT_RULE_TOGGLES,
      R4_BARRIER_AXIS: hasBarrier
    },
    classes,
    checklist: GENERIC_CHECKLIST_ITEMS.map((c) => c.text)
  };
}
