/**
 * Background Service Worker (MV3) — CVAT 3D Fusion Pre-label
 * - Tách biệt 100% với cuboid3d_extension cũ (dùng key fusion3d_* và cổng 127.0.0.1:8766)
 * - Hỗ trợ tự động inject script vào tab đang mở mà không bắt buộc F5
 * - Hỗ trợ đăng ký động cho domain CVAT khác ngoài cvat.edufun.ai & localhost:8080
 */

const AGENT_BASE = "http://127.0.0.1:8766";
const ORIGINS_KEY = "fusion3d_registered_origins";

chrome.sidePanel?.setPanelBehavior?.({ openPanelOnActionClick: true }).catch(() => {});

function scriptIds(originPattern) {
  const safe = originPattern.replace(/[^a-zA-Z0-9]/g, "_").slice(0, 56);
  return { mainId: `fusion3d_main_${safe}`, isoId: `fusion3d_iso_${safe}` };
}

async function registerOriginScripts(originPattern) {
  if (
    !originPattern ||
    originPattern === "http://localhost:8080/*" ||
    originPattern === "https://cvat.edufun.ai/*"
  ) {
    return;
  }
  const { mainId, isoId } = scriptIds(originPattern);
  const existing = await chrome.scripting.getRegisteredContentScripts({ ids: [mainId, isoId] });
  const ids = new Set(existing.map((s) => s.id));
  const list = [];
  if (!ids.has(mainId)) {
    list.push({
      id: mainId,
      matches: [originPattern],
      js: ["src/page_bridge.js"],
      world: "MAIN",
      runAt: "document_start",
      persistAcrossSessions: true
    });
  }
  if (!ids.has(isoId)) {
    list.push({
      id: isoId,
      matches: [originPattern],
      js: ["src/content_bridge.js"],
      world: "ISOLATED",
      runAt: "document_idle",
      persistAcrossSessions: true
    });
  }
  if (list.length) await chrome.scripting.registerContentScripts(list);

  const stored = await chrome.storage.local.get(ORIGINS_KEY);
  const arr = Array.isArray(stored[ORIGINS_KEY]) ? stored[ORIGINS_KEY] : [];
  if (!arr.includes(originPattern)) {
    arr.push(originPattern);
    await chrome.storage.local.set({ [ORIGINS_KEY]: arr });
  }
}

async function injectIntoTab(tabId) {
  await chrome.scripting.executeScript({
    target: { tabId },
    files: ["src/page_bridge.js"],
    world: "MAIN"
  });
  await chrome.scripting.executeScript({
    target: { tabId },
    files: ["src/content_bridge.js"],
    world: "ISOLATED"
  });
}

chrome.runtime.onStartup.addListener(async () => {
  try {
    const stored = await chrome.storage.local.get(ORIGINS_KEY);
    for (const originPattern of stored[ORIGINS_KEY] || []) {
      if (await chrome.permissions.contains({ origins: [originPattern] })) {
        await registerOriginScripts(originPattern);
      }
    }
  } catch {}
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg || typeof msg !== "object") return false;

  if (msg.type === "FUSION_INJECT_TAB") {
    (async () => {
      try {
        if (msg.originPattern) await registerOriginScripts(msg.originPattern);
        if (typeof msg.tabId === "number") await injectIntoTab(msg.tabId);
        sendResponse({ ok: true });
      } catch (err) {
        sendResponse({ ok: false, error: err.message });
      }
    })();
    return true;
  }

  if (msg.type === "FUSION_AGENT") {
    (async () => {
      try {
        const opts = {
          method: msg.method || "GET",
          headers: { "Content-Type": "application/json" }
        };
        if (msg.body !== undefined) opts.body = JSON.stringify(msg.body);
        const resp = await fetch(`${AGENT_BASE}${msg.endpoint}`, opts);
        const text = await resp.text();
        let data;
        try {
          data = JSON.parse(text);
        } catch {
          data = text;
        }
        sendResponse({ ok: resp.ok, status: resp.status, data });
      } catch (err) {
        sendResponse({
          ok: false,
          error: `Không kết nối được Fusion Agent (${AGENT_BASE}): ${err.message}`
        });
      }
    })();
    return true;
  }

  return false;
});
