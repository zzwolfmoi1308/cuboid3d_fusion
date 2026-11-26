/**
 * Automated Test Suite — CVAT 3D Cuboid QA & Assist (v4)
 * Chạy bằng: node tests/run_qa_tests.mjs
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { runMetadataQA } from "../src/qa/rules_metadata.js";
import { parsePcdBuffer } from "../src/qa/pcd_worker.js";
import { runPcdFrameQA } from "../src/qa/rules_pcd.js";
import { buildLockSizePlan } from "../src/fix/safe_actions.js";
import { formatMentorIssueDraft } from "../src/issue/issue_formatter.js";
import {
  BUILTIN_PROFILES,
  autoMapLabelsToJob,
  shouldSplitTwoWheels,
  getEnabledCanonicalClasses,
  isBarrierRelevantForContext,
  getActiveSizeReferenceRows,
  buildEffectiveQaSettings,
  createProfileFromJobLabels
} from "../src/qa/profile_manager.js";
import {
  importGuidelineContent,
  buildGuidelineTemplateJson
} from "../src/qa/guideline_importer.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(
  fs.readFileSync(path.join(__dirname, "qa_fixture_cases.json"), "utf8")
);

let passed = 0;
function test(name, fn) {
  try {
    fn();
    console.log(`  ✓ ${name}`);
    passed++;
  } catch (err) {
    console.error(`  ✗ ${name}\n    ${err.message}`);
    process.exitCode = 1;
  }
}

console.log("\n=== 1. V1 Metadata QA (9 Rules Recall Tests) ===");
const { findings, summary } = runMetadataQA(
  fixture.rawAnnotations,
  fixture.labelIdToName,
  {},
  { stopFrame: 10 }
);

test("R1: Bắt sai Taxonomy (Error)", () => {
  const f = findings.find((x) => x.ruleId === "R1_TAXONOMY" && x.trackId === 101);
  assert.ok(f, "Phải bắt được R1_TAXONOMY cho track 101");
  assert.equal(f.severity, "Error");
});

test("R2: Bắt vượt phạm vi 50m + gợi ý UNCERTAIN_SCOPE khi sát biên ±2m", () => {
  const f = findings.find((x) => x.ruleId === "R2_DISTANCE_RANGE" && x.trackId === 102);
  assert.ok(f, "Phải bắt được R2_DISTANCE_RANGE cho track 102");
  assert.equal(f.meta.nearBoundary, true);
  assert.equal(f.meta.suggestedIssueType, "UNCERTAIN_SCOPE");
});

test("R3 & R5: Bắt kích thước ngoài bảng trang 7 và pitch/roll khác 0", () => {
  const fSize = findings.find((x) => x.ruleId === "R3_SIZE_REFERENCE" && x.trackId === 103);
  const fPR = findings.find((x) => x.ruleId === "R5_PITCH_ROLL_NONZERO" && x.trackId === 103);
  assert.ok(fSize, "Phải bắt được R3_SIZE_REFERENCE cho track 103");
  assert.ok(fPR, "Phải bắt được R5_PITCH_ROLL_NONZERO cho track 103");
});

test("R4 + Barrier Axis Sync: Bắt xoay nhầm 90° ở mode length_is_x và sạch ở mode width_is_x", () => {
  const fBar = findings.find((x) => x.ruleId === "R4_BARRIER_AXIS" && x.trackId === 104);
  assert.ok(fBar, "Phải bắt được R4_BARRIER_AXIS khi mode=length_is_x");

  const swappedRes = runMetadataQA(
    fixture.rawAnnotations,
    fixture.labelIdToName,
    { barrierAxisMode: "width_is_x" },
    { stopFrame: 10 }
  );
  const fBarSwapped = swappedRes.findings.find(
    (x) => (x.ruleId === "R4_BARRIER_AXIS" || x.ruleId === "R3_SIZE_REFERENCE") && x.trackId === 104
  );
  assert.equal(fBarSwapped, undefined, "Khi mode=width_is_x thì dx=0.55, dy=2.2 phải hợp lệ cả Luật 3 & 4");
});

test("R6: Bắt lệch kích thước trong Track > 5% trên keyframe + V2 LockSizePlan dịch ΔH/2", () => {
  const f = findings.find((x) => x.ruleId === "R6_SIZE_CONSISTENCY" && x.trackId === 105);
  assert.ok(f, "Phải bắt được R6_SIZE_CONSISTENCY cho track 105");
  const plan = buildLockSizePlan(f);
  assert.equal(plan.refFrame, 0);
  assert.equal(plan.items.length, 1);
  assert.equal(plan.items[0].frame, 4);
  assert.equal(plan.items[0].selected, true);
});

test("R7a & R7b: Bắt xung đột occluded+outside (Error) và chuỗi occluded > 6 frames", () => {
  const fConflict = findings.find((x) => x.ruleId === "R7_OCCLUDED_AND_OUTSIDE" && x.trackId === 106);
  const fStreak = findings.find((x) => x.ruleId === "R7_OCCLUDED_STREAK" && x.trackId === 106);
  assert.ok(fConflict, "Phải bắt được R7_OCCLUDED_AND_OUTSIDE");
  assert.equal(fConflict.severity, "Error");
  assert.ok(fStreak, "Phải bắt được R7_OCCLUDED_STREAK (7 frames > 6)");
});

test("R8: Bắt nhảy yaw 180° + đột biến Δ²p và bỏ qua yaw cho Pedestrian", () => {
  const fYaw = findings.find((x) => x.ruleId === "R8_YAW_JUMP" && x.trackId === 107);
  const fAcc = findings.find((x) => x.ruleId === "R8_ACCEL_SPIKE" && x.trackId === 107);
  const fPedYaw = findings.find((x) => x.ruleId === "R8_YAW_JUMP" && x.trackId === 108);
  assert.ok(fYaw && fYaw.meta.near180, "Phải bắt được nhảy yaw ~180° cho track 107");
  assert.ok(fAcc, "Phải bắt được đột biến gia tốc bậc 2 cho track 107");
  assert.equal(fPedYaw, undefined, "Pedestrian quay đầu không bị báo sai khi ignorePedestrianYaw=true");
});

test("R9: Bắt tái sử dụng Track ID sau outside, single-frame track và shape lẻ", () => {
  assert.ok(findings.find((x) => x.ruleId === "R9_TRACK_REAPPEAR_AFTER_OUTSIDE" && x.trackId === 108));
  assert.ok(findings.find((x) => x.ruleId === "R9_SINGLE_FRAME_TRACK" && x.trackId === 109));
  assert.ok(findings.find((x) => x.ruleId === "R9_SHAPE_INSTEAD_OF_TRACK" && x.trackId === 901));
});

test("R8 Ego-Rotation Compensation: 4 xe cùng quay +55° do ego rẽ → 0 báo sai nhờ trừ median(Δyaw)", () => {
  const deg55 = (55 * Math.PI) / 180;
  const egoTurnRaw = {
    shapes: [],
    tracks: [1, 2, 3, 4].map((id) => ({
      id,
      label_id: 1,
      shapes: [
        { frame: 0, occluded: false, outside: false, points: [10 + id * 3, 2, -1, 0, 0, 0.0, 4.5, 1.85, 1.55, 0, 0, 0, 0, 0, 0, 0] },
        { frame: 1, occluded: false, outside: false, points: [10 + id * 3, 2, -1, 0, 0, deg55, 4.5, 1.85, 1.55, 0, 0, 0, 0, 0, 0, 0] }
      ]
    }))
  };
  const res = runMetadataQA(egoTurnRaw, fixture.labelIdToName, { maxYawJumpVehicleDeg: 45 });
  const yawErrors = res.findings.filter((x) => x.ruleId === "R8_YAW_JUMP");
  assert.equal(yawErrors.length, 0, "median(Δyaw) phải khử hoàn toàn góc quay ego 55°");
});

console.log("\n=== 2. V1.5 PCD Parser & 5 PCD Checks ===");
test("PCD Parser (ASCII) + Floating Box + Zero-Point Unoccluded + Missed Object Hint", () => {
  const pts = [];
  // Tạo mặt đất phẳng z = -1.8m quanh (10, 0) và quanh (20, 5)
  for (let x = 6; x <= 14; x += 0.5) {
    for (let y = -4; y <= 4; y += 0.5) {
      pts.push(`${x.toFixed(2)} ${y.toFixed(2)} -1.80`);
    }
  }
  // Điểm thân xe trong box 1 tại (10, 0), nhưng box đặt lơ lửng đáy ở z = -1.30m (cao hơn đất +50cm)
  for (let i = 0; i < 20; i++) {
    pts.push(`${(9.0 + (i % 5) * 0.4).toFixed(2)} ${(-0.6 + Math.floor(i / 5) * 0.3).toFixed(2)} -0.70`);
  }
  // Cụm điểm chưa có box tại (12, 3) cao 1.5m so với đất (-1.8 -> -0.3)
  for (let i = 0; i < 24; i++) {
    pts.push(`${(12.0 + (i % 4) * 0.2).toFixed(2)} ${(3.0 + Math.floor(i / 4) * 0.15).toFixed(2)} ${(-1.2 + (i % 3) * 0.4).toFixed(2)}`);
  }

  const pcdText = [
    "# .PCD v0.7",
    "VERSION 0.7",
    "FIELDS x y z",
    "SIZE 4 4 4",
    "TYPE F F F",
    "COUNT 1 1 1",
    `WIDTH ${pts.length}`,
    "HEIGHT 1",
    `POINTS ${pts.length}`,
    "DATA ascii",
    ...pts
  ].join("\n");

  const buf = new TextEncoder().encode(pcdText).buffer;
  const parsed = parsePcdBuffer(buf);
  assert.equal(parsed.format, "ascii");
  assert.equal(parsed.validPoints, pts.length);

  const pcdRes = runPcdFrameQA(parsed.xyz, [
    {
      trackId: 201,
      className: "Car",
      frame: 0,
      occluded: false,
      outside: false,
      // Box tâm z = -0.50, H = 1.60 => đáy = -1.30 (lơ lửng +0.50m so với đất -1.80)
      points: [10, 0, -0.5, 0, 0, 0, 4.5, 1.85, 1.6, 0, 0, 0, 0, 0, 0, 0]
    },
    {
      trackId: 202,
      className: "Car",
      frame: 0,
      occluded: false,
      outside: false,
      // Box ở chỗ không có điểm nào & occluded=false
      points: [30, 15, -1.0, 0, 0, 0, 4.5, 1.85, 1.55, 0, 0, 0, 0, 0, 0, 0]
    }
  ]);

  assert.ok(pcdRes.findings.find((f) => f.ruleId === "PCD1_FLOATING" && f.trackId === 201), "Phải bắt được box lơ lửng");
  assert.ok(pcdRes.findings.find((f) => f.ruleId === "PCD2_ZERO_POINTS_UNOCCLUDED" && f.trackId === 202), "Phải bắt được box 0 điểm không occluded");
  assert.ok(pcdRes.findings.find((f) => f.ruleId === "PCD5_MISSED_OBJECT_HINT"), "Phải gợi ý được cụm điểm bỏ sót");
});

console.log("\n=== 3. V2.5 Mentor Issue Draft Formatter ===");
test("Định dạng chuẩn Mentor Issue Draft", () => {
  const txt = formatMentorIssueDraft({
    issueType: "UNCERTAIN_CLASS",
    jobId: 42,
    frame: 12,
    trackId: 7,
    className: "Car",
    xyz: [14.5, -2.1, -0.9],
    description: "Xe bán tải hay SUV không rõ thùng sau"
  });
  assert.ok(txt.includes("[UNCERTAIN_CLASS] Job #42 | Frame 12 | Track #7 (Car)"));
  assert.ok(txt.includes("(14.50, -2.10, -0.90)"));
});

console.log("\n=== 4. Multi-Guideline Profile, Barrier Auto-Hide & Guideline Upload ===");
test("Lab Day 13 (không có nhãn Barrier, tách riêng Motorcycle & Bicycle): tự tắt Barrier & tách 2 bánh", () => {
  const day13Labels = ["Car", "Truck", "Pedestrian", "Motorcycle", "Bicycle"];
  assert.equal(
    isBarrierRelevantForContext(BUILTIN_PROFILES.guideline_v1_ai20k, day13Labels),
    false,
    "Job không có nhãn Barrier thì isBarrierRelevantForContext phải trả về false"
  );

  const sizeRows = getActiveSizeReferenceRows(BUILTIN_PROFILES.guideline_v1_ai20k, day13Labels, false);
  assert.equal(
    sizeRows.some((r) => r.className === "Barrier"),
    false,
    "Bảng kích thước phải tự ẩn Barrier khi Job không có Barrier"
  );

  const mapping = autoMapLabelsToJob(day13Labels);
  assert.equal(mapping.Motorcycle, "Motorcycle");
  assert.equal(mapping.Bicycle, "Bicycle");
  assert.equal(mapping.Barrier, "");
  assert.equal(shouldSplitTwoWheels(mapping), true, "Phải bật chế độ tách riêng Bicycle và Motorcycle");

  const enabled = getEnabledCanonicalClasses(mapping);
  assert.ok(enabled.includes("Motorcycle") && enabled.includes("Bicycle"));
  assert.ok(!enabled.includes("Barrier") && !enabled.includes("Two_wheels"));

  const day13Prof = createProfileFromJobLabels("Lab Day 13", day13Labels);
  assert.equal(day13Prof.rules.R4_BARRIER_AXIS, false);
  const eff = buildEffectiveQaSettings({}, day13Prof, day13Labels);
  const res = runMetadataQA(fixture.rawAnnotations, fixture.labelIdToName, eff, { stopFrame: 10 });
  assert.equal(
    res.findings.some((f) => f.ruleId === "R4_BARRIER_AXIS"),
    false,
    "Không được phát sinh cảnh báo R4_BARRIER_AXIS trên bài Lab không có Barrier"
  );
});

test("Upload Guideline dạng Markdown / CSV / Template JSON + Công tắc tắt riêng từng luật (R5, R6, Person Yaw)", () => {
  const mdGuideline = `
# Guideline Highway 10Hz
Sensor rate: 10Hz
Vehicle max range: 70m
| Class | Subtype | L | W | H |
| --- | --- | --- | --- | --- |
| Van | Cargo Van | 4.8 - 6.5 | 1.9 - 2.3 | 1.9 - 2.6 |
| Person | Walker | 0.4 - 0.9 | 0.4 - 0.9 | 1.4 - 1.95 |
- [ ] Kiểm tra đủ nhãn Van và Person
- [ ] Soát kỹ biên 70m trước khi nộp
  `;
  const { profile } = importGuidelineContent(mdGuideline, "highway_10hz.md", ["Van", "Person"]);
  assert.equal(profile.sensorConfig.frameIntervalSec, 0.1, "10Hz phải tự quy đổi sang 0.1s/frame");
  assert.equal(profile.rules.R4_BARRIER_AXIS, false, "Không có Barrier trong guideline thì R4 phải tắt");
  assert.equal(profile.checklistItems.length, 2);
  assert.ok(profile.classSpecs.Van && profile.classSpecs.Person);

  // Tắt riêng R5 (cho phép đường dốc) và R6, kiểm tra Person quay đầu 170° không báo lỗi Yaw
  profile.rules.R5_PITCH_ROLL = false;
  profile.rules.R6_SIZE_CONSISTENCY = false;
  const eff = buildEffectiveQaSettings({}, profile, ["Van", "Person"]);
  const customRaw = {
    shapes: [],
    tracks: [
      {
        id: 501,
        label_id: 10,
        shapes: [
          { frame: 0, occluded: false, outside: false, points: [62, 0, -1, 0.25, 0, 0, 5.5, 2.0, 2.2, 0, 0, 0, 0, 0, 0, 0] },
          { frame: 1, occluded: false, outside: false, points: [63, 0, -1, 0.25, 0, 0, 4.9, 2.0, 2.2, 0, 0, 0, 0, 0, 0, 0] }
        ]
      },
      {
        id: 502,
        label_id: 11,
        shapes: [
          { frame: 0, occluded: false, outside: false, points: [10, 2, -1, 0, 0, 0.0, 0.6, 0.6, 1.7, 0, 0, 0, 0, 0, 0, 0] },
          { frame: 1, occluded: false, outside: false, points: [10.2, 2, -1, 0, 0, 2.95, 0.6, 0.6, 1.7, 0, 0, 0, 0, 0, 0, 0] }
        ]
      }
    ]
  };
  const customLabels = { 10: "Van", 11: "Person" };
  const qaRes = runMetadataQA(customRaw, customLabels, eff, { stopFrame: 2 });
  assert.equal(qaRes.findings.some((f) => f.ruleId === "R2_DISTANCE_RANGE"), false, "63m <= 70m hợp lệ");
  assert.equal(qaRes.findings.some((f) => f.ruleId === "R3_SIZE_REFERENCE"), false, "Van 5.5x2.0x2.2 nằm trong bảng mới");
  assert.equal(qaRes.findings.some((f) => f.ruleId === "R5_PITCH_ROLL_NONZERO"), false, "R5 đã tắt không được báo");
  assert.equal(qaRes.findings.some((f) => f.ruleId === "R6_SIZE_CONSISTENCY"), false, "R6 đã tắt không được báo");
  assert.equal(qaRes.findings.some((f) => f.ruleId === "R8_YAW_JUMP" && f.trackId === 502), false, "Person tự động được nhận diện bỏ qua nhảy yaw");

  // Kiểm tra CSV & JSON Template round-trip
  const csvText = "class,group,max_distance,L,W,H\nForklift,Vehicle,35,2.2-3.8,1.0-1.6,1.8-2.6";
  const csvImported = importGuidelineContent(csvText, "warehouse.csv");
  assert.equal(csvImported.profile.classSpecs.Forklift.maxDistance, 35);

  const tpl = buildGuidelineTemplateJson(["Car", "Cyclist"], "Tpl Test");
  const tplImported = importGuidelineContent(JSON.stringify(tpl), "tpl.json");
  assert.equal(tplImported.profile.validClasses.length, 2);
});

console.log(`\nHoàn tất: ${passed} test passed (${summary.elapsedMs} ms cho V1 scan).\n`);
