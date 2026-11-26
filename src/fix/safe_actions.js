/**
 * V2 Controlled Assist — Safe Actions (An Toàn Tuyệt Đối) cho CVAT 3D Fusion
 *
 * - Dùng chung kho snapshot `fusion3d_snapshot_job_<jobId>` với Pre-label.
 * - Mọi thao tác ghi đều đi qua PageBridge → ObjectState.save() (giữ Undo/Redo Ctrl+Z & Canvas 3D sync).
 * - Khóa Size chỉ ghi lên keyframe thật và dịch tâm z thêm ΔH/2 để đáy box giữ nguyên trên mặt đất.
 * - Xung đột occluded + outside bắt buộc người dùng tự chọn tắt occluded hay tắt outside.
 */

export const SNAPSHOT_KEY_PREFIX = "fusion3d_snapshot_job_";

export async function saveBackupSnapshot(jobId, rawAnnotations, reason = "before_v2_edit") {
  if (!jobId || !rawAnnotations) return null;
  const key = `${SNAPSHOT_KEY_PREFIX}${jobId}`;
  const entry = {
    jobId,
    reason,
    savedAt: new Date().toISOString(),
    trackCount: rawAnnotations.tracks?.length || 0,
    shapeCount: rawAnnotations.shapes?.length || 0,
    rawAnnotations
  };

  if (typeof chrome !== "undefined" && chrome.storage?.local) {
    await chrome.storage.local.set({ [key]: entry });
  }
  return entry;
}

export async function loadBackupSnapshot(jobId) {
  if (!jobId || typeof chrome === "undefined" || !chrome.storage?.local) return null;
  const key = `${SNAPSHOT_KEY_PREFIX}${jobId}`;
  const data = await chrome.storage.local.get(key);
  return data[key] || null;
}

export function buildLockSizePlan(finding) {
  const meta = finding?.meta;
  if (!meta || !meta.refDims || !Array.isArray(meta.deviatedKeyframes)) {
    return null;
  }
  return {
    trackId: finding.trackId,
    className: finding.className,
    refFrame: meta.refFrame,
    targetDims: { ...meta.refDims },
    items: meta.deviatedKeyframes.map((kf) => ({
      frame: kf.frame,
      oldDims: { dx: kf.dx, dy: kf.dy, dz: kf.dz },
      deltaZShift: Number(((meta.refDims.dz - kf.dz) / 2).toFixed(3)),
      maxDevPct: kf.maxDevPct,
      suspectedException: Boolean(kf.suspectedException),
      selected: !kf.suspectedException
    }))
  };
}

export async function dispatchSafeEditToTab(tabId, actionPayload, { jobId, rawAnnotations } = {}) {
  if (jobId && rawAnnotations) {
    await saveBackupSnapshot(jobId, rawAnnotations, actionPayload.kind);
  }

  return new Promise((resolve, reject) => {
    chrome.tabs.sendMessage(
      tabId,
      {
        target: "fusion_content",
        action: "APPLY_V2_SAFE_EDIT",
        payload: actionPayload
      },
      (resp) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
          return;
        }
        if (!resp || !resp.ok) {
          reject(new Error(resp?.error || "Thao tác V2 thất bại."));
          return;
        }
        resolve(resp.result);
      }
    );
  });
}
