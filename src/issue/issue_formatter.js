/**
 * V2.5 Mentor Issue Draft Formatter (Đa Guideline / Đa Profile)
 * Hỗ trợ danh sách loại Issue tùy chỉnh từ từng Profile, fallback về 5 loại chuẩn mặc định.
 */

import { MENTOR_ISSUE_TYPES } from "../qa/guideline_config.js";

export function formatMentorIssueDraft({
  issueType = "UNCERTAIN_CLASS",
  issueTypes = MENTOR_ISSUE_TYPES,
  jobId = "?",
  frame = 0,
  trackId = "?",
  className = "?",
  xyz = null,
  description = "",
  teamVote = ""
}) {
  const catalog = Array.isArray(issueTypes) && issueTypes.length > 0 ? issueTypes : MENTOR_ISSUE_TYPES;
  const validType = catalog.find((t) => t.code === issueType)?.code || issueType;
  const posStr =
    Array.isArray(xyz) && xyz.length >= 3
      ? `(${Number(xyz[0]).toFixed(2)}, ${Number(xyz[1]).toFixed(2)}, ${Number(xyz[2]).toFixed(2)})`
      : "(chưa chọn box)";

  const lines = [
    `[${validType}] Job #${jobId} | Frame ${frame} | Track #${trackId} (${className})`,
    `- Vị trí tâm (x, y, z): ${posStr}`,
    `- Mô tả ngắn: ${description.trim() || "(nhập mô tả tình huống chưa rõ)"}`
  ];

  if (teamVote && teamVote.trim()) {
    lines.push(`- Kết quả Team Majority Voting: ${teamVote.trim()}`);
  } else {
    lines.push(`- Đề xuất / Team Vote: (đang chờ thảo luận nhóm hoặc xác nhận từ mentor)`);
  }

  return lines.join("\n");
}

export function prefillIssueFromFinding(finding, jobId) {
  let issueType = finding?.meta?.suggestedIssueType || "UNCERTAIN_CLASS";
  if (finding?.ruleId === "R2_DISTANCE_RANGE") issueType = "UNCERTAIN_SCOPE";
  else if (finding?.ruleId?.startsWith("R7_")) issueType = "ATTRIBUTE_CHECK";
  else if (finding?.ruleId === "R9_TRACK_REAPPEAR_AFTER_OUTSIDE") issueType = "UNCERTAIN_TRACK";
  else if (finding?.ruleId === "PCD4_SPARSE_POINTS") issueType = "LOW_VISIBILITY";

  return {
    issueType,
    jobId: jobId ?? "?",
    frame: finding?.frame ?? 0,
    trackId: finding?.trackId ?? "?",
    className: finding?.className || "Unknown",
    description: finding?.message || ""
  };
}
