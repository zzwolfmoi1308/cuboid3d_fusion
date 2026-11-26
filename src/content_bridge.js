/**
 * Content Bridge (ISOLATED WORLD) — CVAT 3D Fusion (Pre-label + QA + V2 Assist + Issue)
 * - Xác thực kênh postMessage với page_bridge.js bằng sessionNonce ngẫu nhiên
 * - Tự động nhận diện đầy đủ CSRFToken + Auth Token + Organization Slug (kể cả cohort-5a trên cvat.edufun.ai)
 * - Tải meta, job/labels, raw annotations, PCD, calib JSON, context_image ZIP và hỗ trợ POST /api/issues
 */
(function () {
  if (window.__FUSION3D_CONTENT__) return;
  window.__FUSION3D_CONTENT__ = true;

  const nonce =
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `${Date.now()}_${Math.random().toString(36).slice(2)}`;

  let reqCounter = 0;
  let cachedReduxOrg = "";
  const pending = new Map();

  function handshake() {
    try {
      document.documentElement.setAttribute("data-fusion3d-init", nonce);
      document.dispatchEvent(new Event("fusion3d:init"));
    } finally {
      document.documentElement.removeAttribute("data-fusion3d-init");
    }
  }
  document.addEventListener("fusion3d:ready", handshake);
  handshake();

  window.addEventListener("message", (ev) => {
    const m = ev.data;
    if (ev.source !== window || !m || !m.__fusion3d || m.nonce !== nonce || m.source !== "page") {
      return;
    }
    const p = pending.get(m.reqId);
    if (!p) return;
    pending.delete(m.reqId);
    clearTimeout(p.timer);
    m.ok ? p.resolve(m.result) : p.reject(new Error(m.error || "PageBridge error"));
  });

  function callPage(action, payload = {}, timeoutMs = 300000) {
    handshake();
    return new Promise((resolve, reject) => {
      const reqId = `r${++reqCounter}_${Date.now()}`;
      const timer = setTimeout(() => {
        pending.delete(reqId);
        reject(new Error(`Timeout khi gọi PageBridge (${action})`));
      }, timeoutMs);
      pending.set(reqId, { resolve, reject, timer });
      window.postMessage(
        { __fusion3d: true, nonce, source: "content", reqId, action, payload },
        location.origin
      );
    });
  }

  function getCookie(name) {
    const m = document.cookie.match(new RegExp(`(?:^|; )${name}=([^;]*)`));
    return m ? decodeURIComponent(m[1]) : null;
  }

  function detectOrgSlug(overrideOrg = "") {
    if (overrideOrg) return String(overrideOrg).trim();
    if (cachedReduxOrg) return cachedReduxOrg;
    if (location.hostname === "localhost" || location.hostname === "127.0.0.1") return "";

    try {
      const q = new URLSearchParams(location.search).get("org");
      if (q) return q.trim();

      const pMatch = location.pathname.match(/\/(?:organizations|orgs)\/([^/]+)/);
      if (pMatch) return decodeURIComponent(pMatch[1]).trim();

      const directKeys = ["currentOrganization", "organization", "org", "current_org", "cvat_selected_org"];
      for (const k of directKeys) {
        const val = localStorage.getItem(k) || sessionStorage.getItem(k);
        if (!val) continue;
        try {
          const obj = JSON.parse(val);
          if (obj?.slug) return String(obj.slug).trim();
          if (obj?.name) return String(obj.name).trim();
        } catch {
          const clean = val.replace(/^["']|["']$/g, "").trim();
          if (clean && clean !== "null" && clean !== "undefined" && clean.length < 50) return clean;
        }
      }

      const headerEls = document.querySelectorAll(
        ".cvat-right-header, .cvat-header-right, header, nav, .ant-layout-header, .ant-dropdown-trigger"
      );
      for (const el of headerEls) {
        const m = (el.innerText || el.textContent || "").match(/cohort[-\w]+/i);
        if (m) return m[0].trim();
      }
    } catch {}

    if (location.hostname.includes("edufun.ai")) return "cohort-5a";
    return "";
  }

  function getAuthToken() {
    try {
      for (const k of ["token", "access_token", "authToken", "cvat_token", "key", "client/token"]) {
        const val = localStorage.getItem(k) || sessionStorage.getItem(k);
        if (val) {
          const clean = val.replace(/^["']|["']$/g, "").trim();
          if (clean && clean !== "null" && clean !== "undefined") return clean;
        }
      }
    } catch {}
    return null;
  }

  function buildHeaders(overrideOrg = "", extra = {}) {
    const h = { Accept: "application/vnd.cvat+json, application/json", ...extra };
    const csrf = getCookie("csrftoken");
    if (csrf) h["X-CSRFToken"] = csrf;
    const org = detectOrgSlug(overrideOrg);
    if (org) h["X-Organization"] = org;
    const token = getAuthToken();
    if (token) {
      h["Authorization"] =
        token.startsWith("Token ") || token.startsWith("Bearer ") ? token : `Token ${token}`;
    }
    return { headers: h, org };
  }

  function withOrg(path, org) {
    if (!org || /[?&]org=/.test(path)) return path;
    return `${path}${path.includes("?") ? "&" : "?"}org=${encodeURIComponent(org)}`;
  }

  async function cvatFetch(path, { overrideOrg = "", kind = "json", method = "GET", headers: extraHeaders = {}, body } = {}) {
    const { headers, org } = buildHeaders(overrideOrg, extraHeaders);
    const finalPath = withOrg(path, org);
    const url = finalPath.startsWith("http") ? finalPath : `${location.origin}${finalPath}`;
    const opts = { credentials: "include", method, headers };
    if (body !== undefined) opts.body = body;
    const resp = await fetch(url, opts);
    if (!resp.ok) {
      const txt = await resp.text().catch(() => "");
      throw new Error(`HTTP ${resp.status} tại ${finalPath}: ${txt.slice(0, 180)}`);
    }
    return kind === "json" ? resp.json() : resp.arrayBuffer();
  }

  function cvatGet(path, overrideOrg = "", kind = "json") {
    return cvatFetch(path, { overrideOrg, kind });
  }

  function toBase64(buf) {
    const bytes = new Uint8Array(buf);
    let s = "";
    for (let i = 0; i < bytes.length; i += 0x8000) {
      s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    }
    return btoa(s);
  }

  async function fetchPcdWithFallback(jobId, frame, org, customUrlTemplate = "") {
    const paths = [];
    if (customUrlTemplate) {
      paths.push(
        customUrlTemplate
          .replace(/\{jobId\}/g, String(jobId))
          .replace(/\{frame\}/g, String(frame))
      );
    }
    paths.push(
      `/api/jobs/${jobId}/data?type=frame&number=${frame}&quality=original`,
      `/api/jobs/${jobId}/data?type=frame&number=${frame}&quality=compressed`
    );
    let lastErr = null;
    for (const p of paths) {
      try {
        const buf = await cvatGet(p, org, "buf");
        if (buf.byteLength >= 32) return { buf, url: p };
      } catch (e) {
        lastErr = e;
      }
    }
    throw lastErr || new Error(`Không tải được PCD frame ${frame}`);
  }

  async function fetchFrameBundle({ jobId, frame, org, calibPath, withImages }) {
    const out = { frame };
    const { buf: pcdBuf } = await fetchPcdWithFallback(jobId, frame, org);
    out.pcd_base64 = toBase64(pcdBuf);

    if (calibPath) {
      try {
        const url = calibPath.startsWith("http") ? calibPath : `${location.origin}${calibPath}`;
        const r = await fetch(url, { credentials: "include" });
        if (r.ok) {
          out.calib = await r.json();
        } else {
          out.calibError = `HTTP ${r.status} (${calibPath})`;
        }
      } catch (e) {
        out.calibError = e.message;
      }
    }

    if (withImages) {
      try {
        const zipBuf = await cvatGet(
          `/api/jobs/${jobId}/data?type=context_image&number=${frame}&quality=original`,
          org,
          "buf"
        );
        if (zipBuf.byteLength > 32) {
          out.images_zip_base64 = toBase64(zipBuf);
        }
      } catch (e) {
        out.imagesError = e.message;
      }
    }
    return out;
  }

  async function fetchJobMeta(jobId, org) {
    if (!jobId) throw new Error("Thiếu jobId.");
    const job = await cvatGet(`/api/jobs/${jobId}`, org);
    let labels = [];
    try {
      const lblResp = await cvatGet(`/api/labels?job_id=${jobId}&page_size=200`, org);
      labels = Array.isArray(lblResp.results) ? lblResp.results : Array.isArray(lblResp) ? lblResp : [];
    } catch {}
    return { job, labels };
  }

  async function postCvatIssue({ jobId, frame, message, position = [0, 0], org }) {
    return cvatFetch("/api/issues", {
      overrideOrg: org,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job: Number(jobId),
        frame: Number(frame),
        position,
        message
      })
    });
  }

  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (!msg || msg.target !== "fusion_content") return false;
    (async () => {
      try {
        const { action, payload = {} } = msg;
        let result;
        if (action === "PING") {
          result = { ok: true, url: location.href };
        } else if (action === "GET_CONTEXT") {
          result = await callPage("GET_CONTEXT", payload);
          if (result?.org) cachedReduxOrg = String(result.org).trim();
          else result.org = detectOrgSlug();
        } else if (action === "FETCH_META") {
          result = await cvatGet(`/api/jobs/${payload.jobId}/data/meta`, payload.org);
        } else if (action === "FETCH_JOB_META") {
          result = await fetchJobMeta(payload.jobId, payload.org);
        } else if (action === "FETCH_RAW_ANNOTATIONS") {
          result = await cvatGet(`/api/jobs/${payload.jobId}/annotations`, payload.org);
        } else if (action === "FETCH_FRAME") {
          result = await fetchFrameBundle(payload);
        } else if (action === "FETCH_FRAME_PCD") {
          const { buf, url } = await fetchPcdWithFallback(
            payload.jobId,
            payload.frame,
            payload.org,
            payload.urlTemplate || ""
          );
          result = { ok: true, url, byteLength: buf.byteLength, base64: toBase64(buf) };
        } else if (action === "POST_CVAT_ISSUE") {
          result = await postCvatIssue(payload);
        } else {
          result = await callPage(action, payload);
        }
        sendResponse({ ok: true, result });
      } catch (err) {
        sendResponse({ ok: false, error: err?.message || String(err) });
      }
    })();
    return true;
  });
})();
