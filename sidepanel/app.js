/**
 * Side Panel Controller — CVAT 3D Fusion Suite (Multi-Guideline / Dynamic Profile)
 * Tích hợp:
 *  - Tab 1: Point Cloud Boost, Realtime Camera 2D Overlay & Fusion Pre-label đa model (127.0.0.1:8766)
 *  - Tab 2: QA Audit động theo các luật đang bật của từng Profile + V2 Controlled Assist (ObjectState.save · Ctrl+Z)
 *  - Tab 3: Bảng tham chiếu kích thước & Checklist động theo Profile/Job (ẩn sạch Barrier khi task không có Barrier)
 *  - Tab 4: Tạo mẫu Báo cáo / Mentor Issue & tùy chọn gửi trực tiếp CVAT
 *  - Tab 5: Quản lý Profile, Upload/Dán Guideline (.json/.csv/.md/.txt/.docx/.pdf), Chỉnh sửa Class Specs & 15 công tắc Luật QA
 */

import {
  VALID_CLASSES,
  QA_RULE_CATALOG,
  MENTOR_ISSUE_TYPES,
  GENERIC_CHECKLIST_ITEMS,
  DEFAULT_SETTINGS
} from "../src/qa/guideline_config.js";
import { runMetadataQA } from "../src/qa/rules_metadata.js";
import { runPcdFrameQA } from "../src/qa/rules_pcd.js";
import {
  buildLockSizePlan,
  dispatchSafeEditToTab,
  saveBackupSnapshot as saveV2BackupSnapshot,
  loadBackupSnapshot
} from "../src/fix/safe_actions.js";
import {
  formatMentorIssueDraft,
  prefillIssueFromFinding
} from "../src/issue/issue_formatter.js";
import {
  CANONICAL_AI_CLASSES,
  BUILTIN_PROFILES,
  cloneProfile,
  normalizeProfile,
  inferClassSpecFromLabelName,
  autoMapLabelsToJob,
  createProfileFromJobLabels,
  isBarrierRelevantForContext,
  getActiveSizeReferenceRows,
  buildEffectiveQaSettings,
  buildAgentGuidelinePayload,
  applyLabelMappingToTracks
} from "../src/qa/profile_manager.js";
import {
  importGuidelineContent,
  buildGuidelineTemplateJson
} from "../src/qa/guideline_importer.js";

const SETTINGS_KEY_V2 = "fusion3d_settings_v2";
const SETTINGS_KEY_V1 = "fusion3d_settings_v1";
const PROFILES_KEY = "fusion3d_profiles_v1";
const JOB_PROFILES_KEY = "fusion3d_job_profiles_v1";
const LOCAL_CALIBS_KEY = "fusion3d_local_calibs_v1";
const VIEW_BOOST_KEY = "fusion3d_view_boost_v1";
const CHECKLIST_PREFIX = "fusion3d_checklist_job_";

const DEFAULT_FUSION_SETTINGS = {
  ...DEFAULT_SETTINGS,
  useSecondDet: true,
  useCamera: true,
  useOdom: true,
  calibTemplate: "/cuboid-calib/{path_noext}.json"
};

const $ = (id) => document.getElementById(id);

const state = {
  tabId: null,
  originPattern: "",
  pendingOriginPattern: "",
  ctx: null,
  meta: null,
  rawAnnotations: null,
  labelIdToName: {},
  settings: { ...DEFAULT_FUSION_SETTINGS },
  profiles: {
    guideline_v1_ai20k: cloneProfile(BUILTIN_PROFILES.guideline_v1_ai20k),
    generic_auto: cloneProfile(BUILTIN_PROFILES.generic_auto)
  },
  activeProfileId: "generic_auto",
  jobProfileMap: {},

  // Pre-label state
  tracks: [],
  selected: new Set(),
  stopRequested: false,

  // Camera 2D Overlay & View Boost state
  localCalibs: {},
  overlayEnabled: false,
  viewBoost: {
    pointMul: 1.0,
    brightness: 100,
    brightenCam2d: false,
    camArrowShortcutEnabled: true,
    colorPointsByCuboid: false,
    dimOutsidePoints: false,
    colorSelectedOnly: false,
    showEgoDistance: false
  },
  overlayShowLabels: true,
  overlaySelectedOnly: false,
  showAllSizeRef: false,

  // QA & V2 state
  findingsV1: [],
  findingsPcd: [],
  activeSevFilter: "ALL",
  activeClassFilter: "ALL",
  activeRuleFilter: "ALL",
  activeFrameScope: "ALL", // "ALL" | "CURRENT"
  activeSortMode: "frame_asc", // "frame_asc" | "frame_desc" | "sev_frame" | "track_asc"
  pendingLockPlan: null,
  lastDay0ReportText: "",
  pendingGuidelineFileName: ""
};

let pcdWorker = null;
let pcdReqId = 0;

// --- Common Helpers ---
function showToast(message, type = "info", durationMs = 3500) {
  const el = $("toastBanner");
  el.textContent = message;
  el.className = `toast ${type}`;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.add("hidden"), durationMs);
}

function log(msg) {
  const el = $("runLog");
  el.textContent += msg + "\n";
  el.scrollTop = el.scrollHeight;
}

function wrapPi(a) {
  let r = a;
  while (r > Math.PI) r -= 2 * Math.PI;
  while (r < -Math.PI) r += 2 * Math.PI;
  return r;
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const res = String(reader.result || "");
      const idx = res.indexOf(",");
      resolve(idx >= 0 ? res.slice(idx + 1) : res);
    };
    reader.onerror = () => reject(reader.error || new Error("Không đọc được file"));
    reader.readAsDataURL(file);
  });
}

function switchTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-pane").forEach((p) => {
    p.classList.toggle("active", p.id === `tab-${tabName}`);
  });
}

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab || null;
}

async function content(action, payload = {}, allowAutoInject = true) {
  const tab = await activeTab();
  if (!tab?.id) throw new Error("Không tìm thấy tab đang mở.");
  state.tabId = tab.id;
  try {
    const r = await chrome.tabs.sendMessage(tab.id, { target: "fusion_content", action, payload });
    if (!r?.ok) throw new Error(r?.error || `Lỗi khi gọi ${action}`);
    return r.result;
  } catch (err) {
    if (allowAutoInject && /Receiving end does not exist|Could not establish connection/i.test(err.message)) {
      const u = new URL(tab.url);
      await chrome.runtime.sendMessage({
        type: "FUSION_INJECT_TAB",
        tabId: tab.id,
        originPattern: `${u.origin}/*`
      });
      await new Promise((res) => setTimeout(res, 260));
      return content(action, payload, false);
    }
    throw err;
  }
}

async function agent(endpoint, body) {
  const r = await chrome.runtime.sendMessage({
    type: "FUSION_AGENT",
    endpoint,
    method: body !== undefined ? "POST" : "GET",
    body
  });
  if (!r?.ok) {
    const detail = typeof r?.data === "object" ? r.data?.detail || JSON.stringify(r.data) : r?.data;
    throw new Error(r?.error || detail || `HTTP ${r?.status}`);
  }
  return r.data;
}

function parsePcdInWorker(base64) {
  if (!pcdWorker) {
    pcdWorker = new Worker(chrome.runtime.getURL("src/qa/pcd_worker.js"), { type: "module" });
  }
  const binaryStr = atob(base64);
  const bytes = new Uint8Array(binaryStr.length);
  for (let i = 0; i < binaryStr.length; i++) {
    bytes[i] = binaryStr.charCodeAt(i);
  }
  return new Promise((resolve, reject) => {
    const id = ++pcdReqId;
    const onMsg = (e) => {
      if (e.data?.id !== id) return;
      pcdWorker.removeEventListener("message", onMsg);
      if (e.data.ok) resolve(e.data.result);
      else reject(new Error(e.data.error || "PCD Worker parse error"));
    };
    pcdWorker.addEventListener("message", onMsg);
    pcdWorker.postMessage({ id, buffer: bytes.buffer }, [bytes.buffer]);
  });
}

// --- Profiles & Settings Load / Save ---
function getActiveProfile() {
  return (
    state.profiles[state.activeProfileId] ||
    state.profiles.generic_auto ||
    cloneProfile(BUILTIN_PROFILES.generic_auto)
  );
}

function getEffectiveQaSettings() {
  return buildEffectiveQaSettings(state.settings, getActiveProfile(), state.ctx?.labels || []);
}

async function loadProfiles() {
  const stored = await chrome.storage.local.get([PROFILES_KEY, JOB_PROFILES_KEY]);
  const saved = stored[PROFILES_KEY];
  state.jobProfileMap = stored[JOB_PROFILES_KEY] && typeof stored[JOB_PROFILES_KEY] === "object"
    ? stored[JOB_PROFILES_KEY]
    : {};

  const map = {
    guideline_v1_ai20k: cloneProfile(BUILTIN_PROFILES.guideline_v1_ai20k),
    generic_auto: cloneProfile(BUILTIN_PROFILES.generic_auto)
  };
  if (saved?.profiles && typeof saved.profiles === "object") {
    for (const [id, raw] of Object.entries(saved.profiles)) {
      map[id] = normalizeProfile(raw, id);
      if (id === "guideline_v1_ai20k" || id === "generic_auto") {
        map[id].builtin = true;
      }
    }
  }
  state.profiles = map;
  state.activeProfileId = saved?.activeProfileId && map[saved.activeProfileId]
    ? saved.activeProfileId
    : "generic_auto";
}

async function saveProfiles() {
  await chrome.storage.local.set({
    [PROFILES_KEY]: {
      activeProfileId: state.activeProfileId,
      profiles: state.profiles
    },
    [JOB_PROFILES_KEY]: state.jobProfileMap
  });
}

async function bindProfileToCurrentJob(profileId) {
  const jobId = state.ctx?.jobId;
  if (jobId != null && state.profiles[profileId]) {
    state.jobProfileMap[String(jobId)] = profileId;
    await chrome.storage.local.set({ [JOB_PROFILES_KEY]: state.jobProfileMap });
  }
}

/**
 * Tự động đồng bộ Profile phù hợp với Job đang mở:
 * - Nếu Job đã từng được gán Profile riêng trong jobProfileMap -> khôi phục đúng Profile đó.
 * - Nếu chưa từng gán và đang ở Guideline V1 mặc định nhưng nhãn Job không khớp V1 (vd thiếu Barrier ở Lab Day 13)
 *   -> tự chuyển sang Generic CVAT 3D (Tự động theo Job) & tự map nhãn để không báo sai luật V1 / Barrier!
 */
async function reconcileProfileForConnectedJob() {
  const jobId = state.ctx?.jobId;
  const jobLabels = state.ctx?.labels || [];
  if (!jobId || jobLabels.length === 0) return;

  const rememberedId = state.jobProfileMap[String(jobId)];
  if (rememberedId && state.profiles[rememberedId]) {
    if (state.activeProfileId !== rememberedId) {
      state.activeProfileId = rememberedId;
      syncSettingsToUI();
      renderSizeTable($("sizeSearchInput")?.value || "");
      initIssueForm();
    }
    return;
  }

  const active = getActiveProfile();
  if (active.id === "guideline_v1_ai20k") {
    const jobLower = new Set(jobLabels.map((l) => l.toLowerCase()));
    const v1MatchCount = VALID_CLASSES.filter((c) => jobLower.has(c.toLowerCase())).length;
    const isExactV1 = v1MatchCount === VALID_CLASSES.length && jobLabels.length === VALID_CLASSES.length;
    if (!isExactV1) {
      state.activeProfileId = "generic_auto";
      const gen = getActiveProfile();
      gen.labelMapping = autoMapLabelsToJob(jobLabels);
      await saveProfiles();
      syncSettingsToUI();
      renderSizeTable($("sizeSearchInput")?.value || "");
      showToast(
        `Đã tự chuyển sang "Generic CVAT 3D" theo bộ nhãn của Job #${jobId} (ẩn kiểm tra Barrier thừa). Bạn có thể Upload Guideline riêng nếu cần.`,
        "info",
        4500
      );
      return;
    }
  } else if (active.id === "generic_auto") {
    active.labelMapping = autoMapLabelsToJob(jobLabels);
    await saveProfiles();
    syncSettingsToUI();
  }
}

function renderProfileSelectors() {
  const list = Object.values(state.profiles);
  const optsHtml = list
    .map((p) => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.name)}</option>`)
    .join("");
  $("quickProfileSelect").innerHTML = optsHtml;
  $("cfgProfileSelect").innerHTML = optsHtml;
  $("quickProfileSelect").value = state.activeProfileId;
  $("cfgProfileSelect").value = state.activeProfileId;

  const active = getActiveProfile();
  $("metaProfileBadge").textContent = `Profile: ${active.name.slice(0, 22)}`;
  $("btnDeleteProfile").disabled = Boolean(active.builtin);
}

function renderLabelMappingUI() {
  const active = getActiveProfile();
  const container = $("labelMappingList");
  container.innerHTML = "";
  for (const item of CANONICAL_AI_CLASSES) {
    const row = document.createElement("div");
    row.className = "mapping-row";
    const val = active.labelMapping?.[item.key] ?? "";
    row.innerHTML = `
      <span class="ai-cls" title="${escapeHtml(item.label)}">${escapeHtml(item.key)}</span>
      <input type="text" class="input-sm map-input" data-ai="${escapeHtml(item.key)}"
        list="jobLabelsDatalist" value="${escapeHtml(val)}" placeholder="Để trống = Bỏ qua" />
    `;
    container.appendChild(row);
  }
}

function renderRuleTogglesUI() {
  const active = getActiveProfile();
  const rules = active.rules || {};
  const v1Box = $("ruleTogglesV1");
  const pcdBox = $("ruleTogglesPcd");
  v1Box.innerHTML = "";
  pcdBox.innerHTML = "";

  for (const item of QA_RULE_CATALOG) {
    const row = document.createElement("label");
    row.className = "checkbox-row small";
    const checked = rules[item.key] !== false;
    row.innerHTML = `
      <input type="checkbox" class="rule-toggle-cb" data-rule="${escapeHtml(item.key)}" ${checked ? "checked" : ""} />
      <span>${escapeHtml(item.label)}</span>
    `;
    const cb = row.querySelector("input");
    cb.addEventListener("change", () => {
      if (item.key === "R4_BARRIER_AXIS") {
        $("cfgBarrierAxisGroup").classList.toggle("hidden", !cb.checked);
      }
      updateQaScanButtonLabelsFromDom();
    });
    (item.stage === "PCD" ? pcdBox : v1Box).appendChild(row);
  }
}

function updateQaScanButtonLabelsFromDom() {
  const cbs = Array.from(document.querySelectorAll(".rule-toggle-cb"));
  if (!cbs.length) return;
  let v1Cnt = 0;
  let pcdCnt = 0;
  for (const cb of cbs) {
    if (!cb.checked) continue;
    if (cb.dataset.rule.startsWith("PCD")) pcdCnt++;
    else if (cb.dataset.rule !== "R9_WARN_SHAPE") v1Cnt++;
  }
  $("btnRunV1").textContent = `Quét QA (${v1Cnt}/9 luật)`;
  $("btnRunPcd").textContent = `Quét PCD (${pcdCnt}/5 luật)`;
}

function createClassSpecRowElement(spec) {
  const card = document.createElement("div");
  card.className = "spec-card";
  const fmtPair = (p) => (Array.isArray(p) && p.length === 2 ? `${p[0]}-${p[1]}` : "");
  const tolPct = Math.round(((spec.sizeTolerance ?? 0.1) || 0.1) * 100);

  card.innerHTML = `
    <div class="spec-row-top">
      <input type="text" class="spec-name" value="${escapeHtml(spec.className || "")}" placeholder="Tên Class" title="Tên nhãn trong CVAT" />
      <select class="spec-group" title="Nhóm đối tượng">
        <option value="Vehicle" ${spec.group === "Vehicle" ? "selected" : ""}>Vehicle</option>
        <option value="VRU" ${spec.group === "VRU" ? "selected" : ""}>VRU</option>
        <option value="Static" ${spec.group === "Static" ? "selected" : ""}>Static</option>
      </select>
      <input type="number" class="spec-dist" min="5" max="300" value="${escapeHtml(spec.maxDistance ?? 50)}" title="Khoảng cách tối đa (m)" placeholder="Dist(m)" />
      <button type="button" class="btn btn-xs btn-ghost spec-del" title="Xóa class này">✕</button>
    </div>
    <div class="spec-row-dims">
      <input type="text" class="spec-L" value="${escapeHtml(fmtPair(spec.L))}" placeholder="L: 3.8-5.2" title="Chiều dài L min-max (m)" />
      <input type="text" class="spec-W" value="${escapeHtml(fmtPair(spec.W))}" placeholder="W: 1.7-2.1" title="Chiều rộng W min-max (m)" />
      <input type="text" class="spec-H" value="${escapeHtml(fmtPair(spec.H))}" placeholder="H: 1.4-2.0" title="Chiều cao H min-max (m)" />
      <input type="number" class="spec-tol" min="1" max="50" value="${escapeHtml(tolPct)}" title="Dung sai kích thước (±%)" placeholder="±%" />
    </div>
    <label class="checkbox-row small mt-6">
      <input type="checkbox" class="spec-ignore-yaw" ${spec.ignoreYaw ? "checked" : ""} />
      <span>Bỏ qua cảnh báo nhảy góc Yaw cho class này</span>
    </label>
  `;
  card.querySelector(".spec-del").onclick = () => card.remove();
  return card;
}

function renderClassSpecsEditorUI() {
  const active = getActiveProfile();
  const container = $("classSpecsContainer");
  container.innerHTML = "";
  const specs = Object.values(active.classSpecs || {});
  for (const s of specs) {
    container.appendChild(createClassSpecRowElement(s));
  }
}

function readClassSpecsFromEditorUI() {
  const parsePair = (s) => {
    const m = String(s || "").trim().replace(/,/g, ".").match(/(\d+(?:\.\d+)?)\s*[-–~]\s*(\d+(?:\.\d+)?)/);
    if (!m) return null;
    const a = parseFloat(m[1]);
    const b = parseFloat(m[2]);
    return Number.isFinite(a) && Number.isFinite(b) && a > 0 && b >= a ? [a, b] : null;
  };
  const out = {};
  const sizeRows = [];
  document.querySelectorAll("#classSpecsContainer .spec-card").forEach((card) => {
    const className = card.querySelector(".spec-name").value.trim();
    if (!className) return;
    const group = card.querySelector(".spec-group").value;
    const maxDistance = parseFloat(card.querySelector(".spec-dist").value) || 50;
    const L = parsePair(card.querySelector(".spec-L").value);
    const W = parsePair(card.querySelector(".spec-W").value);
    const H = parsePair(card.querySelector(".spec-H").value);
    const tolPct = parseFloat(card.querySelector(".spec-tol").value) || 10;
    const ignoreYaw = card.querySelector(".spec-ignore-yaw").checked;
    const isBar = /barrier|barricade|guardrail|rao_chan/i.test(className);

    out[className] = {
      className,
      group,
      maxDistance,
      L,
      W,
      H,
      noSizeRef: !L || !W || !H,
      sizeTolerance: tolPct / 100,
      ignoreYaw,
      axisMode: isBar ? "barrier_sync" : "length_is_x"
    };
    if (L && W && H) {
      sizeRows.push({
        subtype: `${className} (${group})`,
        className,
        L,
        W,
        H
      });
    }
  });
  return { classSpecs: out, sizeRows };
}

function updateJobLabelsDatalist() {
  const labels = state.ctx?.labels || [];
  $("jobLabelsDatalist").innerHTML = labels
    .map((l) => `<option value="${escapeHtml(l)}"></option>`)
    .join("");
}

function updateBarrierVisibilityUI() {
  const active = getActiveProfile();
  const barrierRelevant = isBarrierRelevantForContext(active, state.ctx?.labels || []);
  $("barrierPrelabelGroup").classList.toggle("hidden", !barrierRelevant);
  $("btnV2RotateBarrier").classList.toggle("hidden", !barrierRelevant);
  $("barrierAxisBadge").classList.toggle("hidden", !barrierRelevant);
  $("cfgBarrierAxisGroup").classList.toggle("hidden", !Boolean(active.rules?.R4_BARRIER_AXIS));
}

function checkProfileJobMismatch() {
  const banner = $("profileMismatchBanner");
  const jobLabels = state.ctx?.labels || [];
  updateBarrierVisibilityUI();

  if (!state.ctx?.connected || jobLabels.length === 0) {
    banner.classList.add("hidden");
    return;
  }
  const active = getActiveProfile();
  const jobLower = new Set(jobLabels.map((l) => l.toLowerCase()));
  const mappedTargets = Object.values(active.labelMapping || {}).filter(Boolean);
  const matchedCount = mappedTargets.filter((t) => jobLower.has(t.toLowerCase())).length;

  if (matchedCount === 0 || (active.taxonomyMode === "strict_profile" && active.validClasses.some((c) => !jobLower.has(c.toLowerCase())))) {
    banner.classList.remove("hidden");
    banner.textContent =
      `⚠ Nhãn của Job hiện tại (${jobLabels.slice(0, 6).join(", ")}${jobLabels.length > 6 ? "..." : ""}) khác với Profile "${active.name}". ` +
      `Bấm "Tự khớp nhãn" hoặc "Upload Guideline" để đồng bộ!`;
  } else {
    banner.classList.add("hidden");
  }
}

async function selectProfile(profileId) {
  if (!state.profiles[profileId]) return;
  state.activeProfileId = profileId;
  await bindProfileToCurrentJob(profileId);
  const active = getActiveProfile();
  state.settings = {
    ...state.settings,
    ...(active.qaThresholds || {}),
    ...(active.sensorConfig || {})
  };
  await saveProfiles();
  await persistSettings(false);
  initIssueForm();
  if (state.ctx?.jobId) await loadChecklistForJob(state.ctx.jobId);
  else await loadChecklistForJob("default");
  checkProfileJobMismatch();
  showToast(`Đã chuyển sang Profile: ${active.name}`, "info", 2500);
}

async function runAutoMapForActiveJob() {
  if (!state.ctx?.connected || !(state.ctx.labels || []).length) {
    await connect();
  }
  const jobLabels = state.ctx?.labels || [];
  if (!jobLabels.length) {
    showToast("Chưa đọc được danh sách nhãn của Job CVAT hiện tại.", "error");
    return;
  }

  let active = getActiveProfile();
  if (active.id === "guideline_v1_ai20k") {
    const v1Set = new Set(VALID_CLASSES.map((c) => c.toLowerCase()));
    const overlap = jobLabels.filter((l) => v1Set.has(l.toLowerCase())).length;
    if (overlap < VALID_CLASSES.length || jobLabels.length !== VALID_CLASSES.length) {
      state.activeProfileId = "generic_auto";
      active = getActiveProfile();
    }
  }

  active.labelMapping = autoMapLabelsToJob(jobLabels);
  const hasBarrier = jobLabels.some((l) => /barrier|barricade|guardrail|rao_chan/i.test(l));
  if (!hasBarrier && active.rules) {
    active.rules.R4_BARRIER_AXIS = false;
    active.enableBarrierAxisRule = false;
    active.barrierLabelName = "";
  }
  if (active.id !== "guideline_v1_ai20k") {
    active.taxonomyMode = "job_labels";
  }

  await bindProfileToCurrentJob(active.id);
  await saveProfiles();
  syncSettingsToUI();
  renderSizeTable($("sizeSearchInput")?.value || "");
  checkProfileJobMismatch();

  if (state.tracks.length > 0) {
    state.tracks = applyLabelMappingToTracks(state.tracks, active.labelMapping);
    renderTracks();
  }

  const mappedCnt = CANONICAL_AI_CLASSES.filter((c) => Boolean(active.labelMapping?.[c.key])).length;
  showToast(
    `Đã tự khớp ${mappedCnt}/${CANONICAL_AI_CLASSES.length} class AI sang nhãn Job (${active.name}).`,
    "success",
    3800
  );
}

async function loadSettings() {
  await loadProfiles();
  const stored = await chrome.storage.local.get([SETTINGS_KEY_V2, SETTINGS_KEY_V1]);
  const v1 = stored[SETTINGS_KEY_V1] || {};
  const v2 = stored[SETTINGS_KEY_V2] || {};
  const active = getActiveProfile();
  const merged = {
    ...DEFAULT_FUSION_SETTINGS,
    ...v1,
    ...v2,
    ...(active.qaThresholds || {}),
    ...(active.sensorConfig || {})
  };
  if (v1.barrierMode && !v2.barrierAxisMode) {
    merged.barrierAxisMode = v1.barrierMode;
  }
  state.settings = merged;
  syncSettingsToUI();
}

function syncSettingsToUI() {
  const s = state.settings;
  const active = getActiveProfile();
  renderProfileSelectors();
  renderLabelMappingUI();
  renderRuleTogglesUI();
  renderClassSpecsEditorUI();

  $("cfgTaxonomyMode").value = active.taxonomyMode || "job_labels";
  $("cfgDistVehicle").value = String(active.maxDistanceVehicle ?? 50);
  $("cfgDistVru").value = String(active.maxDistanceVru ?? 40);
  $("cfgDistStatic").value = String(active.maxDistanceStatic ?? 30);
  $("cfgChecklistEditor").value = (active.checklistItems || GENERIC_CHECKLIST_ITEMS)
    .map((c) => c.text)
    .join("\n");

  // Pre-label controls
  $("optSecondDet").checked = Boolean(s.useSecondDet);
  $("optCamera").checked = Boolean(s.useCamera);
  $("optOdom").checked = Boolean(s.useOdom);
  $("optBarrier").value = s.barrierAxisMode;
  $("optObjectType").value = active.defaultObjectType === "shape" ? "shape" : "track";
  $("optCalib").value = s.calibTemplate || "/cuboid-calib/{path_noext}.json";

  // Thresholds & Sensor controls
  $("cfgBarrierAxis").value = s.barrierAxisMode;
  $("cfgOccOp").value = s.occludedOperator;
  $("cfgOccThr").value = String(s.occludedThreshold);
  $("cfgOccMode").value = s.occludedCountMode;
  $("cfgCoordFrame").value = s.coordinateFrame;
  $("cfgFrameDt").value = String(active.sensorConfig?.frameIntervalSec ?? s.frameIntervalSec ?? 0.5);
  $("cfgOccMinPts").value = String(active.sensorConfig?.occludedMinPoints ?? s.occludedMinPoints ?? 5);
  $("cfgPitchRoll").value = String(s.maxPitchRollDeg);
  $("cfgYawJump").value = String(s.maxYawJumpVehicleDeg);
  $("cfgAccel").value = String(s.maxSecondOrderAccelM);
  $("cfgSizeTol").value = String(s.trackSizeTolerancePct);
  $("cfgIgnorePedYaw").checked = Boolean(s.ignorePedestrianYaw);
  $("cfgMergeRider").checked = active.sensorConfig?.mergeRiderWithTwoWheels !== false;
  $("cfgAllowPitchRoll").checked = Boolean(active.sensorConfig?.allowPitchRoll);
  $("cfgEnableV2").checked = Boolean(s.enableV2Assist);
  $("cfgEnableDirectIssue").checked = Boolean(s.enableDirectCvatIssuePost);

  $("v2QuickBar").classList.toggle("hidden", !s.enableV2Assist);
  $("btnPostCvatIssue").classList.toggle("hidden", !s.enableDirectCvatIssuePost);

  const modeBadge = $("metaModeBadge");
  if (s.enableV2Assist) {
    modeBadge.textContent = "V2 Assist (ObjectState)";
    modeBadge.classList.add("v2-active");
  } else {
    modeBadge.textContent = "Read-only (V1)";
    modeBadge.classList.remove("v2-active");
  }

  $("barrierAxisBadge").textContent =
    s.barrierAxisMode === "width_is_x" ? "Barrier: dx=W, dy=L" : "Barrier: dx=L, dy=W";

  updateQaScanButtonLabelsFromDom();
  updateBarrierVisibilityUI();
}

async function persistSettings(showMsg = false) {
  await chrome.storage.local.set({ [SETTINGS_KEY_V2]: state.settings });
  syncSettingsToUI();
  renderSizeTable($("sizeSearchInput")?.value || "");
  renderFindings();
  if (showMsg) showToast("Đã lưu cấu hình & Profile thành công.", "success");
}

async function savePrelabelSettings() {
  const active = getActiveProfile();
  active.defaultObjectType = $("optObjectType").value === "shape" ? "shape" : "track";
  state.settings = {
    ...state.settings,
    useSecondDet: $("optSecondDet").checked,
    useCamera: $("optCamera").checked,
    useOdom: $("optOdom").checked,
    barrierAxisMode: $("optBarrier").value,
    calibTemplate: $("optCalib").value.trim() || "/cuboid-calib/{path_noext}.json"
  };
  await saveProfiles();
  await persistSettings(false);
}

async function saveAllSettingsFromTab() {
  const active = getActiveProfile();
  const mapping = {};
  document.querySelectorAll("#labelMappingList .map-input").forEach((inp) => {
    mapping[inp.dataset.ai] = inp.value.trim();
  });
  if (mapping.Motorcycle && mapping.Motorcycle === mapping.Bicycle) {
    mapping.Two_wheels = mapping.Motorcycle;
  } else {
    mapping.Two_wheels = mapping.Motorcycle || mapping.Bicycle || "";
  }

  const rules = {};
  document.querySelectorAll(".rule-toggle-cb").forEach((cb) => {
    rules[cb.dataset.rule] = cb.checked;
  });

  const { classSpecs, sizeRows } = readClassSpecsFromEditorUI();
  const specNames = Object.keys(classSpecs);

  active.taxonomyMode = $("cfgTaxonomyMode").value;
  active.labelMapping = mapping;
  active.rules = rules;
  active.enableDistanceRule = Boolean(rules.R2_DISTANCE);
  active.maxDistanceVehicle = Math.max(5, parseFloat($("cfgDistVehicle").value) || 50);
  active.maxDistanceVru = Math.max(5, parseFloat($("cfgDistVru").value) || 40);
  active.maxDistanceStatic = Math.max(5, parseFloat($("cfgDistStatic").value) || 30);
  active.maxDistanceDefault = active.maxDistanceVehicle;
  active.enableSizeRefRule = Boolean(rules.R3_SIZE_REF);
  active.enableBarrierAxisRule = Boolean(rules.R4_BARRIER_AXIS);
  active.warnShapeInsteadOfTrack = Boolean(rules.R9_WARN_SHAPE);
  active.defaultObjectType = $("optObjectType").value === "shape" ? "shape" : "track";
  active.barrierLabelName = mapping.Barrier || (rules.R4_BARRIER_AXIS ? "Barrier" : "");

  if (specNames.length > 0) {
    active.classSpecs = classSpecs;
    if (active.id !== "guideline_v1_ai20k") {
      active.sizeReferenceRows = sizeRows;
      if (active.taxonomyMode === "strict_profile") {
        active.validClasses = specNames;
      }
    }
  }

  const chkLines = $("cfgChecklistEditor")
    .value.split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  if (chkLines.length > 0) {
    active.checklistItems = chkLines.map((text, idx) => ({
      id: `chk_${active.id}_${idx + 1}`,
      text
    }));
  }

  const frameIntervalSec = Math.max(0.02, parseFloat($("cfgFrameDt").value) || 0.5);
  const occludedMinPoints = Math.max(0, parseInt($("cfgOccMinPts").value, 10) || 0);
  const allowPitchRoll = $("cfgAllowPitchRoll").checked;
  const mergeRiderWithTwoWheels = $("cfgMergeRider").checked;

  active.sensorConfig = {
    ...(active.sensorConfig || {}),
    frameIntervalSec,
    occludedMinPoints,
    allowPitchRoll,
    mergeRiderWithTwoWheels
  };

  state.settings = {
    ...state.settings,
    useSecondDet: $("optSecondDet").checked,
    useCamera: $("optCamera").checked,
    useOdom: $("optOdom").checked,
    calibTemplate: $("optCalib").value.trim() || "/cuboid-calib/{path_noext}.json",
    barrierAxisMode: $("cfgBarrierAxis").value,
    occludedOperator: $("cfgOccOp").value,
    occludedThreshold: Math.max(1, parseInt($("cfgOccThr").value, 10) || 6),
    occludedCountMode: $("cfgOccMode").value,
    coordinateFrame: $("cfgCoordFrame").value,
    frameIntervalSec,
    occludedMinPoints,
    allowPitchRoll,
    mergeRiderWithTwoWheels,
    maxPitchRollDeg: parseFloat($("cfgPitchRoll").value) || 5.0,
    maxYawJumpVehicleDeg: parseFloat($("cfgYawJump").value) || 45,
    maxSecondOrderAccelM: parseFloat($("cfgAccel").value) || 3.5,
    trackSizeTolerancePct: parseFloat($("cfgSizeTol").value) || 5.0,
    ignorePedestrianYaw: $("cfgIgnorePedYaw").checked,
    enableV2Assist: $("cfgEnableV2").checked,
    enableDirectCvatIssuePost: $("cfgEnableDirectIssue").checked
  };

  active.qaThresholds = {
    barrierAxisMode: state.settings.barrierAxisMode,
    occludedOperator: state.settings.occludedOperator,
    occludedThreshold: state.settings.occludedThreshold,
    occludedCountMode: state.settings.occludedCountMode,
    coordinateFrame: state.settings.coordinateFrame,
    maxPitchRollDeg: state.settings.maxPitchRollDeg,
    maxYawJumpVehicleDeg: state.settings.maxYawJumpVehicleDeg,
    maxSecondOrderAccelM: state.settings.maxSecondOrderAccelM,
    trackSizeTolerancePct: state.settings.trackSizeTolerancePct,
    ignorePedestrianYaw: state.settings.ignorePedestrianYaw
  };

  await bindProfileToCurrentJob(active.id);
  await saveProfiles();
  await persistSettings(true);
  await loadChecklistForJob(state.ctx?.jobId || "default");
  checkProfileJobMismatch();
}

// --- Guideline Upload / Paste Modal ---
function openGuidelineUploadModal() {
  const jobId = state.ctx?.jobId;
  state.pendingGuidelineFileName = "";
  $("guidelineApplyMode").value = getActiveProfile().builtin ? "new" : "overwrite";
  $("guidelineProfileName").value = jobId ? `Job #${jobId} Guideline` : "Custom Task Guideline";
  $("guidelineFileStatus").textContent = "Chưa chọn file (có thể chọn file hoặc dán trực tiếp nội dung bên dưới).";
  $("guidelineModal").classList.remove("hidden");
}

function closeGuidelineUploadModal() {
  $("guidelineModal").classList.add("hidden");
}

function downloadGuidelineTemplate() {
  const jobLabels = state.ctx?.labels || [];
  const jobId = state.ctx?.jobId;
  const tpl = buildGuidelineTemplateJson(
    jobLabels,
    jobId ? `Guideline Job #${jobId}` : "Task Guideline Mẫu"
  );
  const blob = new Blob([JSON.stringify(tpl, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = jobId ? `guideline_template_job_${jobId}.json` : "guideline_template.json";
  a.click();
  URL.revokeObjectURL(url);
  showToast("Đã tải file JSON mẫu! Bạn có thể sửa số liệu rồi Upload lại bất kỳ lúc nào.", "success", 4000);
}

async function handleGuidelineFileSelected(file) {
  if (!file) return;
  state.pendingGuidelineFileName = file.name;
  const cleanTitle = file.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim();
  if (cleanTitle) $("guidelineProfileName").value = cleanTitle;

  const ext = (file.name.split(".").pop() || "").toLowerCase();
  if (ext === "pdf" || ext === "docx") {
    $("guidelineFileStatus").textContent = `Đang trích xuất văn bản từ ${file.name} qua Agent...`;
    try {
      const b64 = await fileToBase64(file);
      const res = await agent("/extract_guideline_text", { filename: file.name, file_base64: b64 });
      $("guidelineRawText").value = res.text || "";
      $("guidelineFileStatus").textContent = `Đã trích xuất văn bản từ "${file.name}". Kiểm tra nội dung rồi bấm Phân tích & Áp dụng.`;
    } catch (err) {
      $("guidelineFileStatus").textContent = `Không trích xuất tự động được từ ${file.name}: ${err.message}. Bạn hãy copy-paste nội dung từ file vào ô bên dưới.`;
      showToast(`Hãy dán văn bản từ ${file.name} hoặc bật Fusion Agent để đọc PDF/DOCX.`, "error", 4500);
    }
    return;
  }

  const text = await file.text();
  $("guidelineRawText").value = text;
  $("guidelineFileStatus").textContent = `Đã nạp file "${file.name}" (${text.length.toLocaleString()} ký tự). Bấm Phân tích & Áp dụng.`;
}

async function applyGuidelineFromModal() {
  const rawText = $("guidelineRawText").value.trim();
  if (!rawText) {
    showToast("Hãy chọn file hoặc dán nội dung Guideline vào ô văn bản.", "error");
    return;
  }
  const desiredName = $("guidelineProfileName").value.trim() || "Uploaded Guideline";
  const mode = $("guidelineApplyMode").value;
  const fileName = state.pendingGuidelineFileName || `${desiredName}.txt`;

  try {
    const { profile, summary } = importGuidelineContent(rawText, fileName, state.ctx?.labels || []);
    profile.name = desiredName;

    let targetId = profile.id;
    const cur = getActiveProfile();
    if (mode === "overwrite" && !cur.builtin) {
      targetId = cur.id;
      profile.id = targetId;
    }

    // Nếu đang mở Job CVAT, tự khớp labelMapping sang nhãn thực tế của Job
    if ((state.ctx?.labels || []).length > 0) {
      const jobMap = autoMapLabelsToJob(state.ctx.labels);
      profile.labelMapping = { ...profile.labelMapping, ...jobMap };
    }

    state.profiles[targetId] = normalizeProfile(profile, targetId);
    state.activeProfileId = targetId;
    await bindProfileToCurrentJob(targetId);
    await saveProfiles();
    syncSettingsToUI();
    initIssueForm();
    renderSizeTable($("sizeSearchInput")?.value || "");
    await loadChecklistForJob(state.ctx?.jobId || "default");
    checkProfileJobMismatch();
    closeGuidelineUploadModal();
    showToast(summary, "success", 4500);
  } catch (err) {
    showToast(`Lỗi phân tích Guideline: ${err.message}`, "error", 5000);
  }
}

// --- Snapshot Helpers ---
async function refreshSnapshotBadge(jobId) {
  if (!jobId) return;
  const snap = await loadBackupSnapshot(jobId);
  if (snap) {
    $("snapInfo").textContent = `Snapshot #${jobId}: ${snap.trackCount} tracks (${snap.savedAt.slice(11, 19)})`;
    $("btnDownloadSnap").disabled = false;
  } else {
    $("snapInfo").textContent = `Snapshot #${jobId}: chưa có`;
    $("btnDownloadSnap").disabled = true;
  }
}

async function downloadSnapshotForCurrentJob() {
  const jobId = state.ctx?.jobId;
  if (!jobId) {
    showToast("Chưa xác định Job ID hiện tại.", "error");
    return;
  }
  const snap = await loadBackupSnapshot(jobId);
  if (!snap) {
    showToast(`Chưa có snapshot JSON nào cho Job #${jobId}.`, "info");
    return;
  }
  const blob = new Blob([JSON.stringify(snap, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `fusion3d_job_${jobId}_backup.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// --- Connection & Agent Health ---
function frameName(f) {
  const frames = state.meta?.frames || [];
  if (!frames.length || !state.ctx) return null;
  const start = state.meta?.start_frame ?? state.ctx.startFrame;
  const jobLen = state.ctx.stopFrame - state.ctx.startFrame + 1;
  const idx = frames.length > jobLen ? f : f - start;
  return frames[idx]?.name || frames[f - start]?.name || null;
}

function seqKey(name) {
  if (!name) return "seq0";
  const noext = name.replace(/\.[^./]+$/, "");
  return noext.replace(/[-_]?\d+$/, "");
}

function calibPath(name) {
  if (!name) return null;
  const noext = name.replace(/\.[^./]+$/, "");
  const base = noext.split("/").pop();
  return $("optCalib")
    .value.trim()
    .replace(/\{path_noext\}/g, noext)
    .replace(/\{name_noext\}/g, base);
}

async function connect() {
  try {
    const tab = await activeTab();
    if (!tab?.url || !/^https?:\/\//.test(tab.url)) {
      $("statusDot").className = "brand-dot";
      $("connectionSubtitle").textContent = "Hãy mở tab CVAT 3D (http/https)";
      $("jobInfo").textContent = "Hãy mở tab job CVAT 3D trên trình duyệt.";
      $("btnDomain").classList.add("hidden");
      return;
    }

    state.tabId = tab.id;
    const u = new URL(tab.url);
    const originPattern = `${u.origin}/*`;
    state.originPattern = originPattern;
    const isBuiltIn =
      originPattern === "http://localhost:8080/*" || originPattern === "https://cvat.edufun.ai/*";
    const hasPerm = isBuiltIn || (await chrome.permissions.contains({ origins: [originPattern] }));

    if (!hasPerm) {
      state.pendingOriginPattern = originPattern;
      $("statusDot").className = "brand-dot warn";
      $("connectionSubtitle").textContent = `Chưa cấp quyền cho ${u.host}`;
      $("btnDomain").classList.remove("hidden");
      $("jobInfo").textContent = `Chưa cấp quyền truy cập domain ${u.host}. Bấm "Cấp quyền".`;
      return;
    }
    $("btnDomain").classList.add("hidden");

    const ctx = await content("GET_CONTEXT");
    state.ctx = ctx;

    $("statusDot").className = ctx.connected ? "brand-dot online" : "brand-dot warn";
    $("connectionSubtitle").textContent = ctx.connected
      ? `Đã kết nối ${u.host} (${ctx.dimension})`
      : `${u.host} — Hãy mở 1 job CVAT 3D`;

    $("metaJobId").textContent = ctx.jobId ? `Job #${ctx.jobId}` : "Job: —";
    $("metaFrameRange").textContent =
      ctx.jobId && ctx.stopFrame >= ctx.startFrame
        ? `Frames: ${ctx.startFrame}–${ctx.stopFrame} (#${ctx.currentFrame})`
        : "Frames: —";

    if (Array.isArray(ctx.labelObjects)) {
      for (const l of ctx.labelObjects) {
        if (l.id != null) state.labelIdToName[l.id] = l.name;
      }
    }

    if (!ctx.connected || !ctx.jobId) {
      $("jobInfo").textContent = `Đã nối ${u.host} — Hãy mở 1 Job CVAT 3D cụ thể.`;
      return;
    }

    state.meta = await content("FETCH_META", { jobId: ctx.jobId, org: ctx.org });
    if (!$("optFrom").value || Number($("optFrom").value) < ctx.startFrame || Number($("optFrom").value) > ctx.stopFrame) {
      $("optFrom").value = ctx.startFrame;
    }
    if (!$("optTo").value || Number($("optTo").value) < ctx.startFrame || Number($("optTo").value) > ctx.stopFrame) {
      $("optTo").value = ctx.stopFrame;
    }

    const seqSet = new Set();
    for (let f = ctx.startFrame; f <= ctx.stopFrame; f++) {
      seqSet.add(seqKey(frameName(f)));
    }
    const n0 = frameName(ctx.startFrame);
    $("jobInfo").textContent =
      `Job #${ctx.jobId} (org: ${ctx.org || "none"}) · ${seqSet.size} chuỗi · ${(ctx.labels || []).length} nhãn (${(ctx.labels || []).join(", ")}) · Frame đầu: ${n0 || "?"}`;

    updateJobLabelsDatalist();
    await reconcileProfileForConnectedJob();
    renderSizeTable($("sizeSearchInput")?.value || "");
    checkProfileJobMismatch();

    await refreshSnapshotBadge(ctx.jobId);
    await loadChecklistForJob(ctx.jobId);
    await pushViewBoostState();
    if (state.overlayEnabled && Object.keys(state.localCalibs).length > 0) {
      await pushCamOverlayState(false);
    }
  } catch (e) {
    $("statusDot").className = "brand-dot warn";
    $("connectionSubtitle").textContent = `Không đọc được trang: ${e.message.slice(0, 40)}`;
    $("jobInfo").textContent = "Lỗi kết nối: " + e.message;
  }
}

async function requestDomainPermission() {
  if (!state.pendingOriginPattern) return;
  const tab = await activeTab();
  const ok = await chrome.permissions.request({ origins: [state.pendingOriginPattern] });
  if (!ok) {
    showToast("Bạn đã từ chối cấp quyền cho domain này.", "error");
    return;
  }
  await chrome.runtime.sendMessage({
    type: "FUSION_INJECT_TAB",
    tabId: tab?.id,
    originPattern: state.pendingOriginPattern
  });
  showToast(`Đã cấp quyền an toàn cho ${state.pendingOriginPattern}`, "success");
  await connect();
}

async function checkAgent() {
  try {
    const h = await agent("/health");
    const cpOk = h.centerpoint?.loaded ? "CP" : "noCP";
    const ppOk = h.pointpillars?.loaded ? "+PP" : "";
    const dev = h.centerpoint?.device || "cpu";
    const vram = h.vram_allocated_mb != null ? ` · ${h.vram_allocated_mb}MB` : "";
    $("btnAgent").textContent = `${cpOk}${ppOk} (${dev}${vram})`;
    $("btnAgent").className = "pill ok";
  } catch (e) {
    $("btnAgent").textContent = "agent OFF";
    $("btnAgent").className = "pill bad";
    log(e.message);
  }
}

// ============================================================================
// POINT CLOUD SIZE BOOST & 3D VIEW BRIGHTNESS
// ============================================================================
function updateViewBoostUi() {
  const pm = Number(state.viewBoost.pointMul) || 1.0;
  const br = Number(state.viewBoost.brightness) || 100;
  const cam2d = Boolean(state.viewBoost.brightenCam2d);
  const arrowCam = state.viewBoost.camArrowShortcutEnabled !== false;
  const colPts = Boolean(state.viewBoost.colorPointsByCuboid);
  const dimOut = Boolean(state.viewBoost.dimOutsidePoints);
  const selOnly = Boolean(state.viewBoost.colorSelectedOnly);
  const showDist = Boolean(state.viewBoost.showEgoDistance);
  $("optPointSize").value = String(pm);
  $("valPointSize").textContent = `${pm.toFixed(1)}x`;
  $("optViewBrightness").value = String(br);
  $("valViewBrightness").textContent = `${Math.round(br)}%`;
  $("optBrightenCam2d").checked = cam2d;
  $("optCamArrowShortcut").checked = arrowCam;
  $("optColorPointsByCuboid").checked = colPts;
  $("optDimOutsidePoints").checked = dimOut;
  $("optColorSelectedOnly").checked = selOnly;
  $("optShowEgoDistance").checked = showDist;
  $("pointColorSubOptions").classList.toggle("hidden", !colPts);
  if (!colPts) $("pointColorStatus").classList.add("hidden");
  $("viewBoostBadge").textContent = `${pm.toFixed(1)}x · ${Math.round(br)}%`;
}

async function loadViewBoostFromStorage() {
  try {
    const stored = await chrome.storage.local.get(VIEW_BOOST_KEY);
    const saved = stored[VIEW_BOOST_KEY];
    if (saved && typeof saved === "object") {
      const pm = Number(saved.pointMul);
      const br = Number(saved.brightness);
      if (Number.isFinite(pm) && pm >= 0.5 && pm <= 10) state.viewBoost.pointMul = pm;
      if (Number.isFinite(br) && br >= 30 && br <= 300) state.viewBoost.brightness = br;
      state.viewBoost.brightenCam2d = Boolean(saved.brightenCam2d);
      if (typeof saved.camArrowShortcutEnabled === "boolean") {
        state.viewBoost.camArrowShortcutEnabled = saved.camArrowShortcutEnabled;
      }
      state.viewBoost.colorPointsByCuboid = Boolean(saved.colorPointsByCuboid);
      state.viewBoost.dimOutsidePoints = Boolean(saved.dimOutsidePoints);
      state.viewBoost.colorSelectedOnly = Boolean(saved.colorSelectedOnly);
      state.viewBoost.showEgoDistance = Boolean(saved.showEgoDistance);
    }
  } catch {}
  updateViewBoostUi();
}

async function saveViewBoostToStorage() {
  try {
    await chrome.storage.local.set({ [VIEW_BOOST_KEY]: state.viewBoost });
  } catch {}
}

async function pushViewBoostState() {
  if (!state.tabId) return;
  try {
    const res = await content("SET_VIEW_BOOST", state.viewBoost);
    const statusEl = $("pointColorStatus");
    if (state.viewBoost.colorPointsByCuboid && res) {
      statusEl.classList.remove("hidden");
      if (res.pointColorShaderPatched === false) {
        statusEl.textContent = "⚠ Hãy F5 (tải lại) tab CVAT 1 lần để kích hoạt shader tô màu điểm GPU.";
      } else {
        statusEl.textContent = `Đang tô màu điểm theo ${res.coloredBoxCount ?? 0} cuboid trên frame hiện tại.`;
      }
    } else {
      statusEl.classList.add("hidden");
    }
  } catch {}
}

// ============================================================================
// REALTIME CAMERA 2D PROJECTION OVERLAY
// ============================================================================
async function loadLocalCalibsFromStorage() {
  const stored = await chrome.storage.local.get(LOCAL_CALIBS_KEY);
  const saved = stored[LOCAL_CALIBS_KEY];
  if (saved && typeof saved === "object") {
    state.localCalibs = saved.calibs || {};
    state.overlayEnabled = Boolean(saved.enabled);
    state.overlayShowLabels = saved.showLabels !== false;
    state.overlaySelectedOnly = Boolean(saved.selectedOnly);
    if (saved.lastPath) $("inputLocalCalibPath").value = saved.lastPath;
  }
  $("optCamOverlay").checked = state.overlayEnabled;
  $("optOverlayLabels").checked = state.overlayShowLabels;
  $("optOverlaySelectedOnly").checked = state.overlaySelectedOnly;
  updateOverlayStatusText();
}

async function saveLocalCalibsToStorage() {
  await chrome.storage.local.set({
    [LOCAL_CALIBS_KEY]: {
      calibs: state.localCalibs,
      enabled: state.overlayEnabled,
      showLabels: state.overlayShowLabels,
      selectedOnly: state.overlaySelectedOnly,
      lastPath: $("inputLocalCalibPath").value.trim()
    }
  });
}

function buildFrameNameMap() {
  const map = {};
  if (!state.ctx) return map;
  for (let f = state.ctx.startFrame; f <= state.ctx.stopFrame; f++) {
    const n = frameName(f);
    if (n) map[f] = n;
  }
  return map;
}

function updateOverlayStatusText(extra = "") {
  const keys = Object.keys(state.localCalibs || {}).filter((k) => k !== "default");
  const hasAny = Object.keys(state.localCalibs || {}).length > 0;
  $("overlayBadge").textContent = state.overlayEnabled ? "Đang bật" : "Tắt";
  if (!hasAny) {
    $("overlayStatus").textContent = "Chưa nạp calib.json (chọn thư mục Calib hoặc file calib.json 1 lần).";
  } else {
    const listStr = keys.length ? keys.sort().join(", ") : "default";
    $("overlayStatus").textContent = `Đã nạp calib (${listStr})${extra ? ` · ${extra}` : ""}`;
  }
}

async function pushCamOverlayState(showFeedback = true) {
  state.overlayEnabled = $("optCamOverlay").checked;
  state.overlayShowLabels = $("optOverlayLabels").checked;
  state.overlaySelectedOnly = $("optOverlaySelectedOnly").checked;
  await saveLocalCalibsToStorage();

  if (state.overlayEnabled && Object.keys(state.localCalibs).length === 0) {
    state.overlayEnabled = false;
    $("optCamOverlay").checked = false;
    updateOverlayStatusText();
    if (showFeedback) {
      showToast("Hãy nạp thư mục chứa calib.json trước khi bật chiếu Camera 2D.", "error");
    }
    return;
  }

  try {
    const res = await content("SET_CAM_OVERLAY", {
      enabled: state.overlayEnabled,
      calibs: state.localCalibs,
      frameNames: buildFrameNameMap(),
      showLabels: state.overlayShowLabels,
      selectedOnly: state.overlaySelectedOnly
    });
    if (state.overlayEnabled) {
      const info = `${res.drawnPanels || 0} cam (${(res.matchedCams || []).join(", ")}) · ${res.boxesDrawn || 0} khung`;
      updateOverlayStatusText(info);
      if (showFeedback) showToast(`Đã bật chiếu Cuboid 3D lên ${res.drawnPanels || 0} camera 2D!`, "success");
    } else {
      updateOverlayStatusText("Đã tắt");
    }
  } catch (err) {
    updateOverlayStatusText(`Lỗi: ${err.message}`);
    if (showFeedback) showToast(`Lỗi Camera 2D Overlay: ${err.message}`, "error");
  }
}

async function importCalibFilesFromBrowser(fileList) {
  const files = Array.from(fileList || []).filter((f) => f.name.toLowerCase().endsWith(".json"));
  if (!files.length) {
    showToast("Không tìm thấy file .json nào trong thư mục đã chọn.", "error");
    return;
  }
  const calibs = {};
  for (const f of files) {
    if (files.length > 1 && !f.name.toLowerCase().includes("calib")) continue;
    try {
      const parsed = JSON.parse(await f.text());
      if (!parsed || (!parsed.calib && !parsed.cameras && !parsed.cam01 && !parsed.CAM_FRONT)) continue;
      const rel = (f.webkitRelativePath || f.name).toLowerCase();
      const m = rel.match(/sequence[_-]?(\d+)/i) || rel.match(/s(\d{2})f\d+/i);
      const key = m ? `s${String(Number(m[1])).padStart(2, "0")}` : "default";
      calibs[key] = parsed;
      if (!calibs.default) calibs.default = parsed;
    } catch {}
  }
  const cnt = Object.keys(calibs).length;
  if (!cnt) {
    showToast("Không đọc được file calib.json hợp lệ nào.", "error");
    return;
  }
  state.localCalibs = calibs;
  state.overlayEnabled = true;
  $("optCamOverlay").checked = true;
  await saveLocalCalibsToStorage();
  await pushCamOverlayState(true);
}

async function loadCalibViaAgent() {
  const p = $("inputLocalCalibPath").value.trim();
  if (!p) {
    showToast("Hãy dán đường dẫn thư mục chứa calib.json.", "error");
    return;
  }
  try {
    const r = await agent("/load_local_calib", { path: p });
    state.localCalibs = r.calibs || {};
    state.overlayEnabled = true;
    $("optCamOverlay").checked = true;
    await saveLocalCalibsToStorage();
    await pushCamOverlayState(true);
  } catch (err) {
    showToast(`Không nạp được calib từ Agent: ${err.message}`, "error", 4500);
  }
}

// ============================================================================
// TAB 1: FUSION PRE-LABEL
// ============================================================================
function summarizeGuidelineSafety(trkList) {
  let occOutConflict = 0;
  let highCnt = 0;
  let uncertainCnt = 0;
  let missedCnt = 0;
  for (const t of trkList) {
    if (t.level === "HIGH") highCnt++;
    if (t.tags.includes("UNCERTAIN_CLASS")) uncertainCnt++;
    if (t.tags.includes("MISSED?")) missedCnt++;
    const outSet = new Set(t.outside_frames || []);
    for (const kf of t.keyframes || []) {
      if (kf.occluded && outSet.has(kf.frame)) occOutConflict++;
    }
  }
  const active = getActiveProfile();
  const el = $("qaSummary");
  el.classList.remove("hidden");
  el.textContent =
    `Tổng ${trkList.length} tracks (${active.name}): ${highCnt} HIGH · ${uncertainCnt} cần xem class · ${missedCnt} gợi ý bỏ sót · ` +
    `${occOutConflict} xung đột occluded+outside.`;
}

async function runRange(fromFrame, toFrame) {
  await savePrelabelSettings();
  if (!state.ctx?.connected) await connect();
  if (!state.ctx?.connected) return;

  const useSecond = $("optSecondDet").checked;
  const useCam = $("optCamera").checked;
  const useOdom = $("optOdom").checked;
  const barrierMode = $("optBarrier").value;
  const activeProfile = getActiveProfile();
  const agentCfg = buildAgentGuidelinePayload(activeProfile, state.ctx?.labels || []);

  state.stopRequested = false;
  $("btnRun").disabled = true;
  $("btnRunCur").disabled = true;
  $("btnStop").disabled = false;
  $("runLog").textContent = "";

  try {
    await agent("/reset", { job_id: state.ctx.jobId, barrier_axis_mode: barrierMode });
    const total = toFrame - fromFrame + 1;
    $("prog").max = total + 1;
    $("prog").value = 0;

    for (let f = fromFrame; f <= toFrame; f++) {
      if (state.stopRequested) {
        log(`⏹ Đã dừng tại frame ${f} theo yêu cầu.`);
        break;
      }
      const name = frameName(f);
      const sq = seqKey(name);
      const bundle = await content("FETCH_FRAME", {
        jobId: state.ctx.jobId,
        frame: f,
        org: state.ctx.org,
        calibPath: useCam ? calibPath(name) : null,
        withImages: useCam
      });
      if (bundle.calibError) log(`  ⚠ f${f}: calib ${bundle.calibError}`);
      if (bundle.imagesError) log(`  ⚠ f${f}: ảnh ${bundle.imagesError}`);

      const r = await agent("/frame", {
        job_id: state.ctx.jobId,
        frame: f,
        seq: sq,
        pcd_base64: bundle.pcd_base64,
        calib: bundle.calib || null,
        images_zip_base64: bundle.images_zip_base64 || null,
        use_camera: useCam,
        use_second_detector: useSecond,
        use_odometry: useOdom,
        include_traffic_cone: Boolean(activeProfile.labelMapping?.Traffic_cone),
        enabled_classes: agentCfg.enabled_classes,
        split_two_wheels: agentCfg.split_two_wheels,
        merge_rider: agentCfg.merge_rider,
        enable_size_clamp: agentCfg.enable_size_clamp,
        class_size_ranges: agentCfg.class_size_ranges,
        max_distance_by_group: agentCfg.max_distance_by_group,
        max_distance_by_class: agentCfg.max_distance_by_class
      });

      const odomStr = r.odometry?.valid ? ` · ICP Δyaw=${r.odometry.dyaw_deg}°` : "";
      const camDenom = r.total_cams || r.images_found?.length || 0;
      const camStr = useCam
        ? ` · cam ${r.images_found.length}${camDenom ? "/" + camDenom : ""}${r.camera && !r.camera.ok ? " (" + r.camera.error + ")" : ""}`
        : "";
      log(`f${f} [${sq.split("/").pop()}]: ${r.count} box (CP:${r.raw_cp} PP:${r.raw_pp}${odomStr}${camStr}) ${r.elapsed_ms}ms`);
      $("prog").value = f - fromFrame + 1;
    }

    const t = await agent("/tracks", {
      job_id: state.ctx.jobId,
      barrier_axis_mode: barrierMode,
      label_mapping: activeProfile.labelMapping || null,
      frame_dt: agentCfg.frame_dt,
      max_frame_gap: agentCfg.max_frame_gap,
      occluded_min_points: agentCfg.occluded_min_points,
      lock_track_size: agentCfg.lock_track_size
    });
    state.tracks = applyLabelMappingToTracks(t.tracks || [], activeProfile.labelMapping);
    state.selected.clear();
    for (const trk of state.tracks) {
      if (!trk.tags.includes("MISSED?")) state.selected.add(trk.track_id);
    }
    $("prog").value = $("prog").max;
    log(`✅ Hoàn tất ghép track: ${state.tracks.length} tracks.`);
    summarizeGuidelineSafety(state.tracks);
    renderTracks();
  } catch (e) {
    log("❌ Lỗi: " + e.message);
    showToast(`Lỗi Pre-label: ${e.message}`, "error");
  } finally {
    $("btnRun").disabled = false;
    $("btnRunCur").disabled = false;
    $("btnStop").disabled = true;
  }
}

function trackVisible(t) {
  const f = new Set([...document.querySelectorAll(".flt:checked")].map((x) => x.value));
  if (t.tags.includes("MISSED?")) return f.has("MISSED");
  if (t.tags.includes("UNCERTAIN_CLASS") || t.tags.includes("DETECTOR_DISAGREE")) {
    return f.has("UNCERTAIN") || f.has(t.level);
  }
  return f.has(t.level);
}

function flipTrackYaw180(t) {
  for (const kf of t.keyframes) {
    kf.points[5] = Number(wrapPi(kf.points[5] + Math.PI).toFixed(4));
  }
  t._flipped = !t._flipped;
}

function renderTracks() {
  const box = $("tracks");
  box.innerHTML = "";
  const list = state.tracks.filter(trackVisible);
  $("trackCount").textContent = `${state.selected.size}/${state.tracks.length} được chọn`;

  for (const t of list) {
    const row = document.createElement("div");
    row.className = "trk";

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = state.selected.has(t.track_id);
    cb.onclick = (e) => {
      e.stopPropagation();
      cb.checked ? state.selected.add(t.track_id) : state.selected.delete(t.track_id);
      $("trackCount").textContent = `${state.selected.size}/${state.tracks.length} được chọn`;
    };

    const info = document.createElement("div");
    const kf = t.keyframes.find((k) => k.frame === t.refFrame) || t.keyframes[0];
    const p = kf.points;

    const head = document.createElement("div");
    head.className = "trk-head";
    const title = document.createElement("b");
    title.textContent = `#${t.track_id} ${t.className}`;
    head.appendChild(title);

    const lvl = document.createElement("span");
    lvl.className = `tag ${t.level}`;
    lvl.textContent = t.level;
    head.appendChild(lvl);

    for (const g of t.tags) {
      const tg = document.createElement("span");
      tg.className = `tag ${g === "MISSED?" ? "MISSED" : g}`;
      tg.textContent = g;
      head.appendChild(tg);
    }

    const metaLine = document.createElement("div");
    metaLine.className = "muted small";
    const voteStr = Object.keys(t.votes || {}).length > 1 ? ` · vote ${JSON.stringify(t.votes)}` : "";
    const outStr = t.outside_frames?.length ? ` · end@${t.outside_frames.join(",")}` : "";
    metaLine.textContent =
      `f${t.first_frame}–${t.last_frame} (${t.length} kf${outStr}) · nguồn: ${t.sources.join("+")} · sc ${t.score}` +
      ` · ${p[6].toFixed(2)}×${p[7].toFixed(2)}×${p[8].toFixed(2)}m${voteStr}`;

    const ctrl = document.createElement("div");
    ctrl.className = "trk-ctrl";
    ctrl.onclick = (e) => e.stopPropagation();

    const selCls = document.createElement("select");
    const active = getActiveProfile();
    const jobClassChoices =
      state.ctx?.labels && state.ctx.labels.length > 0
        ? state.ctx.labels
        : active.validClasses?.length > 0
          ? active.validClasses
          : VALID_CLASSES;
    const classChoices = jobClassChoices.includes(t.className)
      ? jobClassChoices
      : [t.className, ...jobClassChoices];
    for (const c of classChoices) {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = c;
      if (c === t.className) opt.selected = true;
      selCls.appendChild(opt);
    }
    selCls.onchange = () => {
      t.className = selCls.value;
      t.tags = t.tags.filter((x) => x !== "UNCERTAIN_CLASS" && x !== "DETECTOR_DISAGREE");
      renderTracks();
    };

    const btnFlip = document.createElement("button");
    btnFlip.textContent = t._flipped ? "↺ Đã lật 180°" : "↻ Lật mũi tên 180°";
    btnFlip.onclick = () => {
      flipTrackYaw180(t);
      renderTracks();
    };

    ctrl.append(selCls, btnFlip);
    info.append(head, metaLine, ctrl);

    row.append(cb, info);
    row.onclick = () => content("JUMP_FRAME", { frame: t.refFrame }).catch((e) => log(e.message));
    box.appendChild(row);
  }
}

async function injectSelectedTracks() {
  const chosen = state.tracks.filter((t) => state.selected.has(t.track_id));
  if (!chosen.length) {
    showToast("Chưa chọn track nào để chèn.", "error");
    return;
  }
  $("btnInject").disabled = true;
  try {
    if (state.ctx?.jobId) {
      const raw = await content("FETCH_RAW_ANNOTATIONS", { jobId: state.ctx.jobId, org: state.ctx.org });
      state.rawAnnotations = raw;
      const snap = await saveV2BackupSnapshot(state.ctx.jobId, raw, "before_prelabel_inject");
      await refreshSnapshotBadge(state.ctx.jobId);
      log(`💾 Đã lưu backup snapshot trước khi ghi (${snap.trackCount} tracks hiện có).`);
    }
    const active = getActiveProfile();
    const objectType = $("optObjectType").value === "shape" ? "shape" : "track";
    const allowPitchRoll = Boolean(active.sensorConfig?.allowPitchRoll);
    const r = await content("INJECT_TRACKS", { tracks: chosen, objectType, allowPitchRoll });
    if (r.objectType === "shape") {
      log(
        `✅ Đã chèn ${r.created} Cuboid Shape độc lập vào bộ nhớ CVAT.\n` +
        `   → Hỗ trợ Ctrl+Z hoàn tác. Kiểm tra trên Top/Side/Front/Camera rồi bấm Save!`
      );
      showToast(`Đã chèn ${r.created} Cuboid Shape vào CVAT (hỗ trợ Ctrl+Z).`, "success", 4500);
    } else {
      log(
        `✅ Đã chèn ${r.created} tracks (${r.keyframes} keyframes, ${r.outsideCount} mốc outside) vào bộ nhớ CVAT.\n` +
        `   → Hỗ trợ Ctrl+Z hoàn tác. Kiểm tra trên Top/Side/Front/Camera rồi bấm Save!`
      );
      showToast(`Đã chèn ${r.created} tracks vào CVAT (hỗ trợ Ctrl+Z).`, "success", 4500);
    }
    for (const s of r.skipped || []) log("  ⚠ bỏ qua: " + s);
  } catch (e) {
    log("❌ Lỗi chèn CVAT: " + e.message);
    showToast(`Lỗi chèn CVAT: ${e.message}`, "error");
  } finally {
    $("btnInject").disabled = false;
  }
}

// ============================================================================
// TAB 2: QA AUDIT (METADATA + PCD + V2 SAFE ASSIST)
// ============================================================================
async function runV1Scan() {
  await connect();
  if (!state.ctx?.jobId) {
    showToast("Chưa phát hiện Job CVAT 3D. Hãy mở 1 job rồi thử lại.", "error");
    return;
  }

  const jobId = state.ctx.jobId;
  $("btnRunV1").disabled = true;
  $("btnRunV1").textContent = "Đang quét...";

  try {
    const rawAnn = await content("FETCH_RAW_ANNOTATIONS", { jobId, org: state.ctx.org });
    state.rawAnnotations = rawAnn;

    if (Object.keys(state.labelIdToName).length === 0) {
      try {
        const meta = await content("FETCH_JOB_META", { jobId, org: state.ctx.org });
        for (const l of meta.labels || []) {
          state.labelIdToName[l.id] = l.name;
        }
      } catch {}
    }

    const eff = getEffectiveQaSettings();
    const { findings, summary } = runMetadataQA(rawAnn, state.labelIdToName, eff, {
      stopFrame: state.ctx.stopFrame
    });

    state.findingsV1 = findings;
    $("scanTiming").textContent = `${summary.totalTracks} tracks · ${summary.elapsedMs} ms`;
    populateFilterDropdowns();
    renderFindings();

    showToast(
      `Quét xong (${eff.profileName}): ${summary.errors} Error, ${summary.warnings} Warning, ${summary.infos} Info.`,
      summary.errors > 0 ? "error" : "success"
    );
  } catch (err) {
    showToast(`Lỗi quét QA: ${err.message}`, "error");
  } finally {
    $("btnRunV1").disabled = false;
    updateQaScanButtonLabelsFromDom();
  }
}

async function runPcdScanCurrentFrame() {
  await connect();
  if (!state.ctx?.jobId) {
    showToast("Chưa phát hiện Job CVAT 3D.", "error");
    return;
  }

  const jobId = state.ctx.jobId;
  const frame = state.ctx.currentFrame ?? 0;

  $("btnRunPcd").disabled = true;
  $("btnRunPcd").textContent = "Đang parse PCD...";

  try {
    if (!state.rawAnnotations) {
      state.rawAnnotations = await content("FETCH_RAW_ANNOTATIONS", { jobId, org: state.ctx.org });
    }

    const pcdResp = await content("FETCH_FRAME_PCD", {
      jobId,
      frame,
      org: state.ctx.org,
      urlTemplate: state.settings.pcdUrlTemplate
    });

    const parsed = await parsePcdInWorker(pcdResp.base64);

    const frameBoxes = [];
    for (const tr of state.rawAnnotations.tracks || []) {
      const shapes = [...(tr.shapes || [])].sort((a, b) => a.frame - b.frame);
      const exact = shapes.find((s) => s.frame === frame);
      const prev = [...shapes].reverse().find((s) => s.frame <= frame);
      const active = exact || (prev && !prev.outside ? prev : null);
      if (active && !active.outside) {
        frameBoxes.push({
          trackId: tr.id,
          className: state.labelIdToName[tr.label_id] || `Label(${tr.label_id})`,
          points: active.points,
          occluded: Boolean(active.occluded),
          outside: Boolean(active.outside),
          frame
        });
      }
    }
    for (const sh of state.rawAnnotations.shapes || []) {
      if (sh.frame === frame && !sh.outside && Array.isArray(sh.points) && sh.points.length >= 9) {
        frameBoxes.push({
          trackId: sh.id ?? null,
          className: state.labelIdToName[sh.label_id] || `Label(${sh.label_id})`,
          points: sh.points,
          occluded: Boolean(sh.occluded),
          outside: false,
          frame
        });
      }
    }

    const res = runPcdFrameQA(parsed.xyz, frameBoxes, getEffectiveQaSettings());
    state.findingsPcd = res.findings;
    populateFilterDropdowns();
    renderFindings();

    showToast(
      `PCD Frame #${frame} (${parsed.format}, ${parsed.validPoints.toLocaleString()} điểm): ${res.findings.length} cảnh báo/gợi ý.`,
      "success"
    );
  } catch (err) {
    showToast(`Lỗi quét PCD: ${err.message}`, "error");
  } finally {
    $("btnRunPcd").disabled = false;
    updateQaScanButtonLabelsFromDom();
  }
}

function getAllFindings() {
  return [...state.findingsV1, ...state.findingsPcd];
}

function updateCurrentFrameFilterLabel() {
  const cur = state.ctx?.currentFrame;
  const opt = $("optFilterCurFrame");
  if (opt) {
    opt.textContent = typeof cur === "number" ? `Frame hiện tại (#${cur})` : "Frame hiện tại (#—)";
  }
}

function sortFindingsList(items) {
  const sevRank = { Error: 0, Warning: 1, Info: 2 };
  const mode = state.activeSortMode || "frame_asc";
  return [...items].sort((a, b) => {
    const fa = Number.isFinite(Number(a.frame)) ? Number(a.frame) : 1e9;
    const fb = Number.isFinite(Number(b.frame)) ? Number(b.frame) : 1e9;
    const sa = sevRank[a.severity] ?? 9;
    const sb = sevRank[b.severity] ?? 9;
    const ta = a.trackId != null ? Number(a.trackId) : 1e9;
    const tb = b.trackId != null ? Number(b.trackId) : 1e9;

    if (mode === "frame_desc") {
      if (fa !== fb) return fb - fa;
      if (sa !== sb) return sa - sb;
      return ta - tb;
    }
    if (mode === "sev_frame") {
      if (sa !== sb) return sa - sb;
      if (fa !== fb) return fa - fb;
      return ta - tb;
    }
    if (mode === "track_asc") {
      if (ta !== tb) return ta - tb;
      if (fa !== fb) return fa - fb;
      return sa - sb;
    }
    // Mặc định frame_asc: Frame tăng dần (0 -> N), cùng frame thì Error -> Warning -> Info -> Track ID
    if (fa !== fb) return fa - fb;
    if (sa !== sb) return sa - sb;
    return ta - tb;
  });
}

function populateFilterDropdowns() {
  const all = getAllFindings();
  const classes = [...new Set(all.map((f) => f.className).filter(Boolean))].sort();
  const rules = [...new Set(all.map((f) => f.ruleId).filter(Boolean))].sort();

  const selClass = $("filterClass");
  selClass.innerHTML =
    `<option value="ALL">Mọi Class (${classes.length})</option>` +
    classes.map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");
  selClass.value = classes.includes(state.activeClassFilter) ? state.activeClassFilter : "ALL";

  const selRule = $("filterRule");
  selRule.innerHTML =
    `<option value="ALL">Mọi Luật (${rules.length})</option>` +
    rules.map((r) => `<option value="${escapeHtml(r)}">${escapeHtml(r)}</option>`).join("");
  selRule.value = rules.includes(state.activeRuleFilter) ? state.activeRuleFilter : "ALL";
  updateCurrentFrameFilterLabel();
}

function renderFindings() {
  updateCurrentFrameFilterLabel();
  const allRaw = getAllFindings();
  const curFrame = state.ctx?.currentFrame;
  const scoped =
    state.activeFrameScope === "CURRENT" && typeof curFrame === "number"
      ? allRaw.filter((f) => Number(f.frame) === curFrame)
      : allRaw;

  $("cntTotal").textContent = String(scoped.length);
  $("cntError").textContent = String(scoped.filter((f) => f.severity === "Error").length);
  $("cntWarning").textContent = String(scoped.filter((f) => f.severity === "Warning").length);
  $("cntInfo").textContent = String(scoped.filter((f) => f.severity === "Info").length);

  const filtered = sortFindingsList(
    scoped.filter((f) => {
      if (state.activeSevFilter !== "ALL" && f.severity !== state.activeSevFilter) return false;
      if (state.activeClassFilter !== "ALL" && f.className !== state.activeClassFilter) return false;
      if (state.activeRuleFilter !== "ALL" && f.ruleId !== state.activeRuleFilter) return false;
      return true;
    })
  );

  const list = $("findingsList");
  if (filtered.length === 0) {
    list.innerHTML = `<div class="empty-state">${
      allRaw.length === 0
        ? "Chưa có kết quả hoặc job sạch 0 lỗi theo Profile hiện tại. Bấm <b>Quét QA</b> để kiểm tra."
        : state.activeFrameScope === "CURRENT" && typeof curFrame === "number"
          ? `Không có cảnh báo nào tại Frame #${curFrame} (chuyển sang "Tất cả Job" để xem các frame khác).`
          : "Không có mục nào khớp bộ lọc hiện tại."
    }</div>`;
    return;
  }

  list.innerHTML = "";
  for (const f of filtered) {
    const card = document.createElement("div");
    card.className = `finding-card sev-${f.severity}`;

    card.innerHTML = `
      <div class="finding-head">
        <div class="finding-title">${escapeHtml(f.title)}</div>
        <span class="tag sev-${f.severity}">${f.severity}</span>
      </div>
      <div class="finding-tags">
        <span class="tag">Frame #${f.frame ?? "?"}</span>
        ${f.trackId != null ? `<span class="tag">Track #${f.trackId}</span>` : ""}
        <span class="tag">${escapeHtml(f.className || "Unknown")}</span>
        <span class="tag">${escapeHtml(f.ruleId)}</span>
      </div>
      <div class="finding-msg">${escapeHtml(f.message)}</div>
      ${f.suggestion ? `<div class="finding-sug">💡 ${escapeHtml(f.suggestion)}</div>` : ""}
      <div class="finding-actions"></div>
    `;

    const actionsEl = card.querySelector(".finding-actions");

    const btnJump = document.createElement("button");
    btnJump.className = "btn btn-xs btn-secondary";
    btnJump.textContent = `Nhảy tới Frame #${f.frame ?? 0}`;
    btnJump.addEventListener("click", async () => {
      try {
        if (f.trackId != null) {
          await content("SELECT_OBJECT", { serverId: f.trackId, frame: f.frame });
        } else {
          await content("JUMP_FRAME", { frame: f.frame });
        }
      } catch (err) {
        showToast(`Không nhảy được frame: ${err.message}`, "error");
      }
    });
    actionsEl.appendChild(btnJump);

    const btnIssue = document.createElement("button");
    btnIssue.className = "btn btn-xs btn-ghost";
    btnIssue.textContent = "Tạo Issue Draft";
    btnIssue.addEventListener("click", () => {
      const pre = prefillIssueFromFinding(f, state.ctx?.jobId);
      if ([...$("issueTypeSelect").options].some((o) => o.value === pre.issueType)) {
        $("issueTypeSelect").value = pre.issueType;
      }
      $("issueFrame").value = String(pre.frame);
      $("issueTrackId").value = String(pre.trackId);
      $("issueClass").value = pre.className;
      $("issueDesc").value = pre.description;
      updateIssuePreview();
      switchTab("issue");
    });
    actionsEl.appendChild(btnIssue);

    if (state.settings.enableV2Assist) {
      if (f.ruleId === "R7_OCCLUDED_AND_OUTSIDE") {
        const btnOffOcc = document.createElement("button");
        btnOffOcc.className = "btn btn-xs btn-warn";
        btnOffOcc.textContent = "V2: Tắt occluded";
        btnOffOcc.addEventListener("click", () =>
          runV2SafeAction({
            kind: "RESOLVE_OCCLUDED_OUTSIDE",
            serverId: f.trackId,
            frame: f.frame,
            turnOff: "occluded"
          })
        );
        const btnOffOut = document.createElement("button");
        btnOffOut.className = "btn btn-xs btn-warn";
        btnOffOut.textContent = "V2: Tắt outside";
        btnOffOut.addEventListener("click", () =>
          runV2SafeAction({
            kind: "RESOLVE_OCCLUDED_OUTSIDE",
            serverId: f.trackId,
            frame: f.frame,
            turnOff: "outside"
          })
        );
        actionsEl.append(btnOffOcc, btnOffOut);
      }

      if (f.ruleId === "R6_SIZE_CONSISTENCY" && f.meta?.deviatedKeyframes?.length) {
        const btnLock = document.createElement("button");
        btnLock.className = "btn btn-xs btn-warn";
        btnLock.textContent = "V2: Khóa Size Keyframes...";
        btnLock.addEventListener("click", () => openLockSizeModal(f));
        actionsEl.appendChild(btnLock);
      }

      if (f.ruleId === "R8_YAW_JUMP" && f.meta?.near180) {
        const btnFlip = document.createElement("button");
        btnFlip.className = "btn btn-xs btn-warn";
        btnFlip.textContent = "V2: Flip Yaw 180°";
        btnFlip.addEventListener("click", () =>
          runV2SafeAction({
            kind: "FLIP_YAW_180",
            serverId: f.trackId,
            frame: f.frame
          })
        );
        actionsEl.appendChild(btnFlip);
      }

      if (f.ruleId === "R4_BARRIER_AXIS") {
        const btnRot = document.createElement("button");
        btnRot.className = "btn btn-xs btn-warn";
        btnRot.textContent = "V2: Xoay 90° + Hoán L/W";
        btnRot.addEventListener("click", () =>
          runV2SafeAction({
            kind: "ROTATE_BARRIER_90",
            serverId: f.trackId,
            frame: f.frame
          })
        );
        actionsEl.appendChild(btnRot);
      }

      if ((f.ruleId === "PCD1_FLOATING" || f.ruleId === "PCD1_SINKING") && f.meta?.canSnapGround) {
        const btnSnap = document.createElement("button");
        btnSnap.className = "btn btn-xs btn-warn";
        btnSnap.textContent = `V2: Snap Ground (z=${f.meta.zGround.toFixed(2)})`;
        btnSnap.addEventListener("click", () =>
          runV2SafeAction({
            kind: "SNAP_TO_GROUND",
            serverId: f.trackId,
            frame: f.frame,
            zGround: f.meta.zGround
          })
        );
        actionsEl.appendChild(btnSnap);
      }
    }

    list.appendChild(card);
  }
}

async function runV2SafeAction(actionPayload) {
  if (!state.settings.enableV2Assist) {
    showToast("V2 Controlled Assist đang tắt. Hãy bật trong tab Settings.", "error");
    return;
  }
  try {
    const tab = await activeTab();
    if (!tab?.id) throw new Error("Không tìm thấy tab đang mở.");
    const res = await dispatchSafeEditToTab(tab.id, actionPayload, {
      jobId: state.ctx?.jobId,
      rawAnnotations: state.rawAnnotations
    });
    if (state.ctx?.jobId) await refreshSnapshotBadge(state.ctx.jobId);
    const detail =
      res?.oldYawDeg != null
        ? ` (#${res.clientID}: yaw ${res.oldYawDeg}° → ${res.newYawDeg}°, L×W ${res.oldDims} → ${res.newDims})`
        : "";
    showToast(
      `Đã áp dụng ${actionPayload.kind}${detail}. Bấm Ctrl+Z nếu cần hoàn tác.`,
      "success",
      4500
    );
    return res;
  } catch (err) {
    showToast(`Lỗi V2 Safe Action: ${err.message}`, "error", 4500);
  }
}

function openLockSizeModal(finding) {
  const plan = buildLockSizePlan(finding);
  if (!plan) return;
  state.pendingLockPlan = plan;

  $("lockSizeTitle").textContent = `Khóa Size Track #${plan.trackId} (${plan.className})`;
  $("lockSizeSubtitle").textContent =
    `Kích thước chuẩn từ keyframe #${plan.refFrame}: ${plan.targetDims.dx.toFixed(2)} × ${plan.targetDims.dy.toFixed(2)} × ${plan.targetDims.dz.toFixed(2)} m. Chỉ ghi lên keyframe thật và dịch tâm z thêm ΔH/2 để đáy box giữ nguyên.`;

  const list = $("lockSizeList");
  list.innerHTML = "";
  plan.items.forEach((item, idx) => {
    const row = document.createElement("label");
    row.className = "check-item";
    row.innerHTML = `
      <input type="checkbox" data-idx="${idx}" ${item.selected ? "checked" : ""} />
      <span>
        <b>Frame #${item.frame}</b> — Lệch ${item.maxDevPct}% · Dịch tâm Δz = ${item.deltaZShift >= 0 ? "+" : ""}${item.deltaZShift} m
        ${item.suspectedException ? '<span class="text-amber"> (Nghi ngoại lệ >25%, mặc định bỏ chọn)</span>' : ""}
      </span>
    `;
    list.appendChild(row);
  });

  $("lockSizeModal").classList.remove("hidden");
}

// ============================================================================
// TAB 3: DYNAMIC SIZE TABLE & PROFILE CHECKLIST
// ============================================================================
function renderSizeTable(query = "") {
  const q = query.trim().toLowerCase();
  const active = getActiveProfile();
  const swappedBarrier = state.settings.barrierAxisMode === "width_is_x";
  const barrierName = String(active.barrierLabelName || "Barrier").toLowerCase();
  const baseRows = getActiveSizeReferenceRows(
    active,
    state.ctx?.labels || [],
    Boolean(state.showAllSizeRef)
  );
  const rows = baseRows.filter(
    (r) =>
      !q ||
      String(r.className || "").toLowerCase().includes(q) ||
      String(r.subtype || "").toLowerCase().includes(q)
  );

  $("sizeCardTitle").textContent = `Bảng Kích Thước (${active.name.slice(0, 24)})`;
  const tbody = $("sizeTableBody");
  if (rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4" class="muted">Không có dòng kích thước nào khớp (bật "Hiện mọi mẫu" hoặc thêm trong Settings).</td></tr>`;
    return;
  }

  tbody.innerHTML = rows
    .map((r) => {
      const isSwapped = String(r.className || "").toLowerCase() === barrierName && swappedBarrier;
      const colL = isSwapped ? `${r.W[0]}–${r.W[1]} (dx)` : `${r.L[0]}–${r.L[1]}`;
      const colW = isSwapped ? `${r.L[0]}–${r.L[1]} (dy)` : `${r.W[0]}–${r.W[1]}`;
      return `
        <tr>
          <td>
            <div>${escapeHtml(r.subtype)}</div>
            <div class="ref-class">${escapeHtml(r.className)}</div>
          </td>
          <td>${colL}</td>
          <td>${colW}</td>
          <td>${r.H[0]}–${r.H[1]}</td>
        </tr>
      `;
    })
    .join("");
}

async function loadChecklistForJob(jobId) {
  const active = getActiveProfile();
  const key = `${CHECKLIST_PREFIX}${active.id}_${jobId || "default"}`;
  const stored = await chrome.storage.local.get(key);
  const checkedMap = stored[key] || {};
  renderChecklist(checkedMap, key);
}

function renderChecklist(checkedMap, storageKey) {
  const active = getActiveProfile();
  const items =
    Array.isArray(active.checklistItems) && active.checklistItems.length > 0
      ? active.checklistItems
      : GENERIC_CHECKLIST_ITEMS;

  $("checklistCardTitle").textContent = `Checklist ${items.length} Bước Trước Khi Submit`;
  const container = $("checklistContainer");
  container.innerHTML = "";
  let doneCount = 0;

  for (const item of items) {
    const isChecked = Boolean(checkedMap[item.id]);
    if (isChecked) doneCount++;

    const label = document.createElement("label");
    label.className = `check-item ${isChecked ? "checked" : ""}`;
    label.innerHTML = `
      <input type="checkbox" ${isChecked ? "checked" : ""} />
      <span>${escapeHtml(item.text)}</span>
    `;
    const cb = label.querySelector("input");
    cb.addEventListener("change", async () => {
      checkedMap[item.id] = cb.checked;
      await chrome.storage.local.set({ [storageKey]: checkedMap });
      renderChecklist(checkedMap, storageKey);
    });
    container.appendChild(label);
  }

  const pct = items.length ? Math.round((doneCount / items.length) * 100) : 0;
  $("checklistProgressFill").style.width = `${pct}%`;
  $("checklistProgressText").textContent = `${doneCount} / ${items.length}`;
}

// ============================================================================
// TAB 4: MENTOR ISSUE DRAFT
// ============================================================================
function getActiveIssueCatalog() {
  const active = getActiveProfile();
  return Array.isArray(active.issueTypes) && active.issueTypes.length > 0
    ? active.issueTypes
    : MENTOR_ISSUE_TYPES;
}

function initIssueForm() {
  const catalog = getActiveIssueCatalog();
  const sel = $("issueTypeSelect");
  const prevVal = sel.value;
  sel.innerHTML = catalog
    .map((t) => `<option value="${escapeHtml(t.code)}">${escapeHtml(t.label)}</option>`)
    .join("");
  if (catalog.some((t) => t.code === prevVal)) {
    sel.value = prevVal;
  }
  updateIssuePreview();
}

function updateIssuePreview() {
  const catalog = getActiveIssueCatalog();
  const code = $("issueTypeSelect").value;
  const meta = catalog.find((t) => t.code === code);
  $("issueTypeExample").textContent = meta?.example ? `Ví dụ: ${meta.example}` : "";

  const xyzRaw = $("issueXyz")
    .value.split(",")
    .map((v) => parseFloat(v.trim()))
    .filter(Number.isFinite);

  const draft = formatMentorIssueDraft({
    issueType: code,
    issueTypes: catalog,
    jobId: state.ctx?.jobId ?? "?",
    frame: parseInt($("issueFrame").value, 10) || 0,
    trackId: $("issueTrackId").value || "?",
    className: $("issueClass").value || "?",
    xyz: xyzRaw.length >= 3 ? xyzRaw : null,
    description: $("issueDesc").value,
    teamVote: $("issueTeamVote").value
  });

  $("issueDraftPreview").textContent = draft;
}

// ============================================================================
// TAB 5: DAY 0 PROBE
// ============================================================================
async function runDay0Diagnostic() {
  await connect();
  if (!state.ctx?.connected) {
    showToast("Hãy mở 1 tab job CVAT 3D trước khi chạy Probe.", "error");
    return;
  }

  $("btnRunDay0").disabled = true;
  $("btnRunDay0").textContent = "Đang kiểm chứng...";

  try {
    const probe = await content("RUN_DAY0_PROBE");
    const jobId = probe.jobInfo?.id || state.ctx?.jobId;

    let apiSummary = null;
    let noiseStats = null;
    if (jobId) {
      try {
        const rawAnn = await content("FETCH_RAW_ANNOTATIONS", { jobId, org: state.ctx.org });
        const qa = runMetadataQA(rawAnn, state.labelIdToName, getEffectiveQaSettings(), {
          stopFrame: probe.jobInfo?.stopFrame
        });
        apiSummary = {
          tracks: rawAnn.tracks?.length || 0,
          shapes: rawAnn.shapes?.length || 0,
          outsideKeyframesInApi: (rawAnn.tracks || []).reduce(
            (acc, t) => acc + (t.shapes || []).filter((s) => s.outside).length,
            0
          )
        };
        noiseStats = qa.summary.noiseStats;
      } catch (e) {
        apiSummary = { error: e.message };
      }
    }

    let pcdCheck = null;
    if (jobId) {
      try {
        const frame = probe.jobInfo?.currentFrame ?? 0;
        const pcdResp = await content("FETCH_FRAME_PCD", {
          jobId,
          frame,
          org: state.ctx.org,
          urlTemplate: state.settings.pcdUrlTemplate
        });
        const parsed = await parsePcdInWorker(pcdResp.base64);
        const zs = [];
        for (let i = 0; i < parsed.validPoints; i += Math.max(1, Math.floor(parsed.validPoints / 4000))) {
          zs.push(parsed.xyz[i * 3 + 2]);
        }
        zs.sort((a, b) => a - b);
        pcdCheck = {
          endpoint: pcdResp.url,
          format: parsed.format,
          fields: parsed.fields,
          validPoints: parsed.validPoints,
          zP05: Number((zs[Math.floor(zs.length * 0.05)] ?? 0).toFixed(2)),
          zP15: Number((zs[Math.floor(zs.length * 0.15)] ?? 0).toFixed(2)),
          zMedian: Number((zs[Math.floor(zs.length * 0.5)] ?? 0).toFixed(2))
        };
      } catch (e) {
        pcdCheck = { error: e.message };
      }
    }

    const fullReport = {
      day0Version: "fusion-v3-multi-guideline",
      activeProfile: getActiveProfile().name,
      probe,
      apiAnnotationsCheck: apiSummary,
      backgroundNoiseStats: noiseStats,
      pcdCheck
    };

    const pretty = JSON.stringify(fullReport, null, 2);
    state.lastDay0ReportText = pretty;
    $("day0Output").textContent = pretty;
    $("day0Output").classList.remove("hidden");
    $("btnCopyDay0").disabled = false;
    showToast("Đã hoàn tất Probe kỹ thuật!", "success");
  } catch (err) {
    showToast(`Lỗi Probe: ${err.message}`, "error");
  } finally {
    $("btnRunDay0").disabled = false;
    $("btnRunDay0").textContent = "Chạy Probe Kỹ Thuật";
  }
}

// ============================================================================
// EVENTS & INIT
// ============================================================================
function bindEvents() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  // Header
  $("btnConnect").onclick = connect;
  $("btnDomain").onclick = requestDomainPermission;
  $("btnAgent").onclick = checkAgent;

  // Point Cloud Size & Brightness
  $("optPointSize").addEventListener("input", () => {
    state.viewBoost.pointMul = parseFloat($("optPointSize").value) || 1.0;
    updateViewBoostUi();
    saveViewBoostToStorage();
    pushViewBoostState();
  });
  $("optViewBrightness").addEventListener("input", () => {
    state.viewBoost.brightness = parseFloat($("optViewBrightness").value) || 100;
    updateViewBoostUi();
    saveViewBoostToStorage();
    pushViewBoostState();
  });
  $("optBrightenCam2d").addEventListener("change", () => {
    state.viewBoost.brightenCam2d = $("optBrightenCam2d").checked;
    saveViewBoostToStorage();
    pushViewBoostState();
  });
  $("optCamArrowShortcut").addEventListener("change", () => {
    state.viewBoost.camArrowShortcutEnabled = $("optCamArrowShortcut").checked;
    saveViewBoostToStorage();
    pushViewBoostState();
  });
  $("optColorPointsByCuboid").addEventListener("change", () => {
    state.viewBoost.colorPointsByCuboid = $("optColorPointsByCuboid").checked;
    updateViewBoostUi();
    saveViewBoostToStorage();
    pushViewBoostState();
  });
  $("optDimOutsidePoints").addEventListener("change", () => {
    state.viewBoost.dimOutsidePoints = $("optDimOutsidePoints").checked;
    saveViewBoostToStorage();
    pushViewBoostState();
  });
  $("optColorSelectedOnly").addEventListener("change", () => {
    state.viewBoost.colorSelectedOnly = $("optColorSelectedOnly").checked;
    saveViewBoostToStorage();
    pushViewBoostState();
  });
  $("optShowEgoDistance").addEventListener("change", () => {
    state.viewBoost.showEgoDistance = $("optShowEgoDistance").checked;
    saveViewBoostToStorage();
    pushViewBoostState();
  });
  $("btnResetViewBoost").onclick = () => {
    state.viewBoost = {
      pointMul: 1.0,
      brightness: 100,
      brightenCam2d: false,
      camArrowShortcutEnabled: $("optCamArrowShortcut").checked,
      colorPointsByCuboid: $("optColorPointsByCuboid").checked,
      dimOutsidePoints: $("optDimOutsidePoints").checked,
      colorSelectedOnly: $("optColorSelectedOnly").checked,
      showEgoDistance: $("optShowEgoDistance").checked
    };
    updateViewBoostUi();
    saveViewBoostToStorage();
    pushViewBoostState();
    showToast("Đã đặt lại Size 1.0x và Độ sáng 100%.", "success");
  };

  // Camera 2D Projection Overlay
  $("optCamOverlay").addEventListener("change", () => pushCamOverlayState(true));
  $("optOverlayLabels").addEventListener("change", () => pushCamOverlayState(false));
  $("optOverlaySelectedOnly").addEventListener("change", () => pushCamOverlayState(false));
  $("btnPickCalibDir").onclick = () => $("inputCalibDir").click();
  $("btnPickCalibFiles").onclick = () => $("inputCalibFiles").click();
  $("inputCalibDir").addEventListener("change", (e) => {
    importCalibFilesFromBrowser(e.target.files);
    e.target.value = "";
  });
  $("inputCalibFiles").addEventListener("change", (e) => {
    importCalibFilesFromBrowser(e.target.files);
    e.target.value = "";
  });
  $("btnLoadCalibAgent").onclick = loadCalibViaAgent;

  // Tab 1: Pre-label
  $("optBarrier").addEventListener("change", () => savePrelabelSettings());
  $("optObjectType").addEventListener("change", () => savePrelabelSettings());
  $("btnQuickUploadGuideline").onclick = openGuidelineUploadModal;
  $("btnRun").onclick = () => runRange(Number($("optFrom").value), Number($("optTo").value));
  $("btnRunCur").onclick = async () => {
    state.ctx = await content("GET_CONTEXT");
    const cur = state.ctx.currentFrame ?? Number($("optFrom").value);
    await runRange(cur, cur);
  };
  $("btnStop").onclick = () => {
    state.stopRequested = true;
  };
  $("btnInject").onclick = injectSelectedTracks;
  $("btnDownloadSnap").onclick = downloadSnapshotForCurrentJob;
  $("btnSelAll").onclick = () => {
    state.tracks.forEach((t) => state.selected.add(t.track_id));
    renderTracks();
  };
  $("btnSelHigh").onclick = () => {
    state.selected.clear();
    state.tracks.forEach((t) => t.level === "HIGH" && state.selected.add(t.track_id));
    renderTracks();
  };
  $("btnSelNone").onclick = () => {
    state.selected.clear();
    renderTracks();
  };
  document.querySelectorAll(".flt").forEach((x) => (x.onchange = renderTracks));

  // Tab 2: QA Audit & V2 Quick Actions
  $("btnRunV1").onclick = runV1Scan;
  $("btnRunPcd").onclick = runPcdScanCurrentFrame;

  document.querySelectorAll(".kpi-card").forEach((card) => {
    card.addEventListener("click", () => {
      document.querySelectorAll(".kpi-card").forEach((c) => c.classList.remove("active-filter"));
      card.classList.add("active-filter");
      state.activeSevFilter = card.dataset.sev;
      renderFindings();
    });
  });

  $("filterFrameScope").addEventListener("change", async (e) => {
    state.activeFrameScope = e.target.value;
    if (state.activeFrameScope === "CURRENT") {
      try {
        const latest = await content("GET_CONTEXT");
        if (latest?.connected) {
          state.ctx = latest;
          if (latest.jobId && latest.stopFrame >= latest.startFrame) {
            $("metaFrameRange").textContent = `Frames: ${latest.startFrame}–${latest.stopFrame} (#${latest.currentFrame})`;
          }
        }
      } catch {}
    }
    renderFindings();
  });
  $("sortFindings").addEventListener("change", (e) => {
    state.activeSortMode = e.target.value;
    renderFindings();
  });
  $("filterClass").addEventListener("change", (e) => {
    state.activeClassFilter = e.target.value;
    renderFindings();
  });
  $("filterRule").addEventListener("change", (e) => {
    state.activeRuleFilter = e.target.value;
    renderFindings();
  });

  // Tự động đồng bộ Frame hiện tại trên tab QA Audit khi đang lọc theo "Frame hiện tại"
  setInterval(async () => {
    if (state.activeFrameScope !== "CURRENT") return;
    const qaPane = $("tab-qa");
    if (!qaPane || !qaPane.classList.contains("active")) return;
    if (!state.ctx?.connected) return;
    try {
      const latest = await content("GET_CONTEXT");
      if (latest?.connected && latest.currentFrame !== state.ctx.currentFrame) {
        state.ctx.currentFrame = latest.currentFrame;
        if (latest.jobId && latest.stopFrame >= latest.startFrame) {
          $("metaFrameRange").textContent = `Frames: ${latest.startFrame}–${latest.stopFrame} (#${latest.currentFrame})`;
        }
        renderFindings();
      }
    } catch {}
  }, 700);

  $("btnV2FlipYaw").onclick = async () => {
    await connect();
    const obj = state.ctx?.activeObject;
    if (!obj) {
      showToast("Hãy chọn 1 cuboid trên CVAT 3D trước.", "error");
      return;
    }
    await runV2SafeAction({
      kind: "FLIP_YAW_180",
      clientId: obj.clientID,
      serverId: obj.serverID,
      frame: obj.frame
    });
  };

  $("btnV2RotateBarrier").onclick = async () => {
    await connect();
    const obj = state.ctx?.activeObject;
    if (!obj) {
      showToast("Hãy chọn 1 cuboid trên CVAT 3D trước.", "error");
      return;
    }
    await runV2SafeAction({
      kind: "ROTATE_BARRIER_90",
      clientId: obj.clientID,
      serverId: obj.serverID,
      frame: obj.frame
    });
  };

  // Lock Size Modal
  const closeLockModal = () => $("lockSizeModal").classList.add("hidden");
  $("btnCloseLockModal").onclick = closeLockModal;
  $("btnCancelLockSize").onclick = closeLockModal;
  $("lockSizeBackdrop").onclick = closeLockModal;
  $("btnConfirmLockSize").onclick = async () => {
    const plan = state.pendingLockPlan;
    if (!plan) return;
    const checks = Array.from($("lockSizeList").querySelectorAll("input[type=checkbox]"));
    const selectedFrames = checks
      .filter((c) => c.checked)
      .map((c) => plan.items[parseInt(c.dataset.idx, 10)].frame);
    if (selectedFrames.length === 0) {
      showToast("Chưa chọn keyframe nào.", "error");
      return;
    }
    closeLockModal();
    await runV2SafeAction({
      kind: "LOCK_TRACK_SIZE_KEYFRAMES",
      serverId: plan.trackId,
      targetDims: plan.targetDims,
      keyframes: selectedFrames
    });
  };

  // Tab 3: Size & Checklist
  $("sizeSearchInput").addEventListener("input", (e) => renderSizeTable(e.target.value));
  $("optShowAllSizeRef").addEventListener("change", (e) => {
    state.showAllSizeRef = e.target.checked;
    renderSizeTable($("sizeSearchInput").value);
  });
  $("btnResetChecklist").onclick = async () => {
    const active = getActiveProfile();
    const key = `${CHECKLIST_PREFIX}${active.id}_${state.ctx?.jobId || "default"}`;
    await chrome.storage.local.remove(key);
    renderChecklist({}, key);
    showToast("Đã đặt lại Checklist.", "info");
  };

  // Tab 4: Mentor Issue
  ["issueTypeSelect", "issueFrame", "issueTrackId", "issueClass", "issueXyz", "issueDesc", "issueTeamVote"].forEach(
    (id) => {
      $(id).addEventListener("input", updateIssuePreview);
      $(id).addEventListener("change", updateIssuePreview);
    }
  );

  $("btnPrefillSelected").onclick = async () => {
    await connect();
    const obj = state.ctx?.activeObject;
    if (!obj) {
      showToast("Chưa có box nào đang được chọn trên CVAT 3D.", "error");
      return;
    }
    $("issueFrame").value = String(obj.frame ?? 0);
    $("issueTrackId").value = String(obj.serverID ?? obj.clientID ?? "");
    $("issueClass").value = obj.labelName || "";
    if (Array.isArray(obj.points) && obj.points.length >= 3) {
      $("issueXyz").value = `${obj.points[0].toFixed(2)}, ${obj.points[1].toFixed(2)}, ${obj.points[2].toFixed(2)}`;
    }
    updateIssuePreview();
    showToast("Đã lấy thông tin từ box đang chọn.", "success");
  };

  $("btnCopyIssueDraft").onclick = async () => {
    await navigator.clipboard.writeText($("issueDraftPreview").textContent);
    showToast("Đã copy mẫu Issue vào clipboard!", "success");
  };

  $("btnPostCvatIssue").onclick = async () => {
    if (!state.ctx?.jobId) {
      showToast("Chưa kết nối Job CVAT.", "error");
      return;
    }
    try {
      const message = $("issueDraftPreview").textContent;
      const frame = parseInt($("issueFrame").value, 10) || 0;
      await content("POST_CVAT_ISSUE", {
        jobId: state.ctx.jobId,
        frame,
        message,
        org: state.ctx.org
      });
      showToast("Đã gửi Issue lên CVAT thành công!", "success");
    } catch (err) {
      showToast(`Gửi POST /api/issues thất bại: ${err.message}`, "error");
    }
  };

  // Tab 5: Profiles, Rule Toggles, Class Specs & Guideline Upload
  $("quickProfileSelect").addEventListener("change", (e) => selectProfile(e.target.value));
  $("cfgProfileSelect").addEventListener("change", (e) => selectProfile(e.target.value));
  $("btnQuickAutoMap").onclick = runAutoMapForActiveJob;
  $("btnAutoMapLabels").onclick = runAutoMapForActiveJob;

  $("btnNewFromJob").onclick = async () => {
    if (!state.ctx?.connected || !(state.ctx.labels || []).length) {
      await connect();
    }
    const jobLabels = state.ctx?.labels || [];
    if (!jobLabels.length) {
      showToast("Hãy mở 1 Job CVAT 3D trước để tạo Profile tự động từ nhãn của Job.", "error");
      return;
    }
    const defaultName = state.ctx?.jobId ? `Task Job #${state.ctx.jobId}` : "Task Profile Mới";
    const name = window.prompt("Nhập tên cho Profile tạo từ Job hiện tại:", defaultName);
    if (!name || !name.trim()) return;

    const prof = createProfileFromJobLabels(name.trim(), jobLabels);
    state.profiles[prof.id] = prof;
    state.activeProfileId = prof.id;
    await bindProfileToCurrentJob(prof.id);
    await saveProfiles();
    syncSettingsToUI();
    renderSizeTable($("sizeSearchInput")?.value || "");
    await loadChecklistForJob(state.ctx?.jobId || "default");
    checkProfileJobMismatch();
    showToast(`Đã tạo Profile "${prof.name}" với ${jobLabels.length} nhãn từ Job!`, "success", 4000);
  };

  $("btnNewProfile").onclick = async () => {
    const name = window.prompt("Nhập tên Profile bản sao:", `${getActiveProfile().name} (Copy)`);
    if (!name || !name.trim()) return;
    const id = `custom_${Date.now()}`;
    const base = cloneProfile(getActiveProfile());
    base.id = id;
    base.name = name.trim();
    base.builtin = false;
    state.profiles[id] = base;
    state.activeProfileId = id;
    await bindProfileToCurrentJob(id);
    await saveProfiles();
    syncSettingsToUI();
    checkProfileJobMismatch();
    showToast(`Đã tạo Profile "${base.name}".`, "success");
  };

  $("btnDeleteProfile").onclick = async () => {
    const active = getActiveProfile();
    if (active.builtin) return;
    if (!window.confirm(`Xóa Profile "${active.name}"?`)) return;
    delete state.profiles[active.id];
    for (const [jId, pId] of Object.entries(state.jobProfileMap)) {
      if (pId === active.id) delete state.jobProfileMap[jId];
    }
    state.activeProfileId = "generic_auto";
    await saveProfiles();
    syncSettingsToUI();
    renderSizeTable($("sizeSearchInput")?.value || "");
    checkProfileJobMismatch();
    showToast(`Đã xóa Profile "${active.name}".`, "info");
  };

  $("btnEnableAllRules").onclick = () => {
    document.querySelectorAll(".rule-toggle-cb").forEach((cb) => (cb.checked = true));
    $("cfgBarrierAxisGroup").classList.remove("hidden");
    updateQaScanButtonLabelsFromDom();
  };
  $("btnDisableAllRules").onclick = () => {
    document.querySelectorAll(".rule-toggle-cb").forEach((cb) => (cb.checked = false));
    $("cfgBarrierAxisGroup").classList.add("hidden");
    updateQaScanButtonLabelsFromDom();
  };

  $("btnAddClassSpec").onclick = () => {
    $("classSpecsContainer").appendChild(
      createClassSpecRowElement(inferClassSpecFromLabelName("New_Class"))
    );
  };

  $("btnSyncSpecsFromJob").onclick = async () => {
    if (!state.ctx?.connected) await connect();
    const jobLabels = state.ctx?.labels || [];
    if (!jobLabels.length) {
      showToast("Chưa đọc được nhãn từ Job CVAT hiện tại.", "error");
      return;
    }
    const { classSpecs } = readClassSpecsFromEditorUI();
    const container = $("classSpecsContainer");
    for (const lbl of jobLabels) {
      const exists = Object.keys(classSpecs).some((k) => k.toLowerCase() === lbl.toLowerCase());
      if (!exists) {
        container.appendChild(createClassSpecRowElement(inferClassSpecFromLabelName(lbl)));
      }
    }
    showToast(`Đã đồng bộ ${jobLabels.length} nhãn từ Job vào bảng chỉnh sửa. Bấm Lưu để áp dụng!`, "info");
  };

  // Guideline Upload Modal events
  $("btnOpenUploadGuideline").onclick = openGuidelineUploadModal;
  $("btnCloseGuidelineModal").onclick = closeGuidelineUploadModal;
  $("btnCancelGuidelineModal").onclick = closeGuidelineUploadModal;
  $("guidelineModalBackdrop").onclick = closeGuidelineUploadModal;
  $("btnDownloadGuidelineTpl").onclick = downloadGuidelineTemplate;
  $("btnModalDownloadTpl").onclick = downloadGuidelineTemplate;
  $("btnPickGuidelineFile").onclick = () => $("fileGuidelineUpload").click();
  $("fileGuidelineUpload").addEventListener("change", (e) => {
    const f = e.target.files?.[0];
    if (f) handleGuidelineFileSelected(f);
    e.target.value = "";
  });
  $("btnApplyGuidelineModal").onclick = applyGuidelineFromModal;

  $("btnExportProfile").onclick = () => {
    const active = getActiveProfile();
    const blob = new Blob([JSON.stringify(active, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `fusion3d_profile_${active.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  $("btnImportProfile").onclick = () => $("fileImportProfile").click();
  $("fileImportProfile").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const { profile, summary } = importGuidelineContent(await file.text(), file.name, state.ctx?.labels || []);
      state.profiles[profile.id] = profile;
      state.activeProfileId = profile.id;
      await bindProfileToCurrentJob(profile.id);
      await saveProfiles();
      syncSettingsToUI();
      initIssueForm();
      renderSizeTable($("sizeSearchInput")?.value || "");
      await loadChecklistForJob(state.ctx?.jobId || "default");
      checkProfileJobMismatch();
      showToast(summary, "success");
    } catch (err) {
      showToast(`File JSON không hợp lệ: ${err.message}`, "error");
    } finally {
      e.target.value = "";
    }
  });

  $("btnRunDay0").onclick = runDay0Diagnostic;
  $("btnCopyDay0").onclick = async () => {
    if (!state.lastDay0ReportText) return;
    await navigator.clipboard.writeText(state.lastDay0ReportText);
    showToast("Đã copy Báo cáo kỹ thuật!", "success");
  };
  $("btnSaveSettings").onclick = saveAllSettingsFromTab;
  $("btnDownloadSnapshotSettings").onclick = downloadSnapshotForCurrentJob;
}

(async function init() {
  await loadSettings();
  await loadViewBoostFromStorage();
  await loadLocalCalibsFromStorage();
  renderSizeTable();
  await loadChecklistForJob("default");
  initIssueForm();
  bindEvents();
  await checkAgent();
  await connect();

  if (chrome.tabs?.onActivated) {
    chrome.tabs.onActivated.addListener(() => connect());
  }
  if (chrome.tabs?.onUpdated) {
    chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
      if (tabId === state.tabId && changeInfo.status === "complete") {
        connect();
      }
    });
  }
})();
