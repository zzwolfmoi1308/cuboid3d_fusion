/**
 * Page Bridge (MAIN WORLD) — CVAT 3D Fusion (Pre-label + QA + V2 Assist + Day 0)
 * - Kết nối Redux store & jobInstance qua React Fiber
 * - Nhảy frame & kích hoạt chọn track trên Canvas 3D
 * - Ghi Track 3D Cuboid đa keyframe qua cvat-core ObjectState API (giữ Undo/Redo Ctrl+Z, không tự Save)
 * - V2 Controlled Assist: sửa bán tự động an toàn qua ObjectState.save()
 * - Bộ kiểm chứng Day 0 (8 điểm kiểm tra thực tế trên job CVAT 3D)
 */
(function () {
  if (window.__FUSION3D_PAGE__) return;
  window.__FUSION3D_PAGE__ = true;

  // ============================================================================
  // POINT CLOUD SIZE BOOST, CUBOID-LABEL POINT COLORING & 3D VIEW BRIGHTNESS
  // ============================================================================
  const VIEW_BOOST_STORAGE_KEY = "fusion3d_view_boost_v1";
  const MAX_GPU_BOXES = 48;
  const viewBoostState = {
    pointMul: 1.0,
    brightness: 100,
    brightenCam2d: false,
    camArrowShortcutEnabled: true,
    colorPointsByCuboid: false,
    dimOutsidePoints: false,
    colorSelectedOnly: false,
    showEgoDistance: false,
    pointShaderPatched: false,
    pointColorShaderPatched: false
  };

  const egoDistState = {
    viewports: new Map(), // canvasEl -> { viewMat, projMat }
    boxes: [],            // cached frame boxes for 60fps overlay redraw during camera orbit
    activeId: null,
    rafId: null,
    unsubscribe: null,
    pollTimer: null
  };

  function mulMat4Vec4(m, x, y, z, w = 1) {
    return [
      m[0] * x + m[4] * y + m[8] * z + m[12] * w,
      m[1] * x + m[5] * y + m[9] * z + m[13] * w,
      m[2] * x + m[6] * y + m[10] * z + m[14] * w,
      m[3] * x + m[7] * y + m[11] * z + m[15] * w
    ];
  }

  const gpuBoxState = {
    count: 0,
    centerYaw: new Float32Array(MAX_GPU_BOXES * 4),
    halfSize: new Float32Array(MAX_GPU_BOXES * 4),
    color: new Float32Array(MAX_GPU_BOXES * 4),
    lastSig: "",
    unsubscribe: null,
    pollTimer: null,
    refreshPending: false
  };

  function parseColorToRgb01(colorStr) {
    const s = String(colorStr || "").trim();
    const hex6 = s.match(/^#([0-9a-f]{6})$/i);
    if (hex6) {
      const n = parseInt(hex6[1], 16);
      return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
    }
    const hex3 = s.match(/^#([0-9a-f]{3})$/i);
    if (hex3) {
      const h = hex3[1];
      return [
        parseInt(h[0] + h[0], 16) / 255,
        parseInt(h[1] + h[1], 16) / 255,
        parseInt(h[2] + h[2], 16) / 255
      ];
    }
    const rgb = s.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i);
    if (rgb) {
      return [
        Math.min(255, Number(rgb[1])) / 255,
        Math.min(255, Number(rgb[2])) / 255,
        Math.min(255, Number(rgb[3])) / 255
      ];
    }
    return [0.22, 0.74, 0.97];
  }

  function injectBeforeMainEnd(src, snippet) {
    const mainMatch = /void\s+main\s*\([^)]*\)\s*\{/.exec(src);
    if (!mainMatch) return null;
    let depth = 0;
    let inLineComment = false;
    let inBlockComment = false;
    const startIdx = mainMatch.index + mainMatch[0].length - 1;
    for (let i = startIdx; i < src.length; i++) {
      const ch = src[i];
      const next = src[i + 1];
      if (inLineComment) {
        if (ch === "\n") inLineComment = false;
        continue;
      }
      if (inBlockComment) {
        if (ch === "*" && next === "/") {
          inBlockComment = false;
          i++;
        }
        continue;
      }
      if (ch === "/" && next === "/") {
        inLineComment = true;
        i++;
        continue;
      }
      if (ch === "/" && next === "*") {
        inBlockComment = true;
        i++;
        continue;
      }
      if (ch === "{") depth++;
      else if (ch === "}") {
        depth--;
        if (depth === 0) {
          return src.slice(0, i) + "\n" + snippet + "\n" + src.slice(i);
        }
      }
    }
    return null;
  }

  function applyPointSizePatch(src) {
    if (
      typeof src === "string" &&
      !src.includes("uFusionPointMul") &&
      /gl_PointSize\s*=\s*([^;]+);/.test(src) &&
      /void\s+main\s*\(/.test(src)
    ) {
      viewBoostState.pointShaderPatched = true;
      return src
        .replace(/void\s+main\s*\(/, "uniform float uFusionPointMul;\nvoid main(")
        .replace(/gl_PointSize\s*=\s*([^;]+);/, "gl_PointSize = ($1) * uFusionPointMul;");
    }
    return src;
  }

  function patchPointsVertexForBoxColor(vsSrc) {
    if (!vsSrc || vsSrc.includes("uFusionBoxCount")) return vsSrc;
    const isEs3 = /^\s*#version\s+300\s+es/m.test(vsSrc);
    const qual = isEs3 ? "out" : "varying";
    const posExpr = /\bmodelMatrix\b/.test(vsSrc) ? "(modelMatrix * vec4(position, 1.0)).xyz" : "position";
    const decl = [
      `uniform int uFusionBoxCount;`,
      `uniform vec4 uFusionBoxCenterYaw[${MAX_GPU_BOXES}];`,
      `uniform vec4 uFusionBoxHalfSize[${MAX_GPU_BOXES}];`,
      `uniform vec4 uFusionBoxColor[${MAX_GPU_BOXES}];`,
      `${qual} vec4 vFusionBoxColor;`
    ].join("\n");

    const body = [
      `  vFusionBoxColor = vec4(0.0);`,
      `  if (uFusionBoxCount > 0) {`,
      `    vec3 _fp = ${posExpr};`,
      `    for (int _fi = 0; _fi < ${MAX_GPU_BOXES}; _fi++) {`,
      `      if (_fi >= uFusionBoxCount) break;`,
      `      vec4 _fcy = uFusionBoxCenterYaw[_fi];`,
      `      vec4 _fhs = uFusionBoxHalfSize[_fi];`,
      `      vec3 _fd = _fp - _fcy.xyz;`,
      `      if (abs(_fd.z) <= _fhs.z) {`,
      `        float _fc = cos(-_fcy.w);`,
      `        float _fs = sin(-_fcy.w);`,
      `        float _flx = _fd.x * _fc - _fd.y * _fs;`,
      `        float _fly = _fd.x * _fs + _fd.y * _fc;`,
      `        if (abs(_flx) <= _fhs.x && abs(_fly) <= _fhs.y) {`,
      `          vFusionBoxColor = uFusionBoxColor[_fi];`,
      `          break;`,
      `        }`,
      `      }`,
      `    }`,
      `  }`
    ].join("\n");

    const withDecl = vsSrc.replace(/void\s+main\s*\(/, `${decl}\nvoid main(`);
    return injectBeforeMainEnd(withDecl, body);
  }

  function patchPointsFragmentForBoxColor(fsSrc) {
    if (!fsSrc || fsSrc.includes("vFusionBoxColor")) return fsSrc;
    const isEs3 = /^\s*#version\s+300\s+es/m.test(fsSrc);
    const qual = isEs3 ? "in" : "varying";
    let outVar = null;
    if (/\bgl_FragColor\b/.test(fsSrc)) {
      outVar = "gl_FragColor";
    } else if (isEs3) {
      const m = fsSrc.match(/\bout\s+(?:(?:highp|mediump|lowp)\s+)?vec4\s+(\w+)\s*;/);
      if (m) outVar = m[1];
    }
    if (!outVar) return null;

    const decl = [
      `uniform float uFusionOutsideDim;`,
      `${qual} vec4 vFusionBoxColor;`
    ].join("\n");
    const body = [
      `  if (vFusionBoxColor.a > 0.01) {`,
      `    ${outVar}.rgb = mix(${outVar}.rgb, vFusionBoxColor.rgb, vFusionBoxColor.a);`,
      `  } else if (uFusionOutsideDim > 0.01 && uFusionOutsideDim < 0.99) {`,
      `    ${outVar}.rgb *= uFusionOutsideDim;`,
      `  }`
    ].join("\n");

    const withDecl = fsSrc.replace(/void\s+main\s*\(/, `${decl}\nvoid main(`);
    return injectBeforeMainEnd(withDecl, body);
  }

  function loadViewBoostFromLocalStorage() {
    try {
      const raw = localStorage.getItem(VIEW_BOOST_STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object") {
        const pm = Number(parsed.pointMul);
        const br = Number(parsed.brightness);
        if (Number.isFinite(pm) && pm >= 0.5 && pm <= 10) viewBoostState.pointMul = pm;
        if (Number.isFinite(br) && br >= 30 && br <= 300) viewBoostState.brightness = br;
        viewBoostState.brightenCam2d = Boolean(parsed.brightenCam2d);
        if (typeof parsed.camArrowShortcutEnabled === "boolean") {
          viewBoostState.camArrowShortcutEnabled = parsed.camArrowShortcutEnabled;
        }
        viewBoostState.colorPointsByCuboid = Boolean(parsed.colorPointsByCuboid);
        viewBoostState.dimOutsidePoints = Boolean(parsed.dimOutsidePoints);
        viewBoostState.colorSelectedOnly = Boolean(parsed.colorSelectedOnly);
        viewBoostState.showEgoDistance = Boolean(parsed.showEgoDistance);
      }
    } catch {}
  }

  function saveViewBoostToLocalStorage() {
    try {
      localStorage.setItem(
        VIEW_BOOST_STORAGE_KEY,
        JSON.stringify({
          pointMul: viewBoostState.pointMul,
          brightness: viewBoostState.brightness,
          brightenCam2d: viewBoostState.brightenCam2d,
          camArrowShortcutEnabled: viewBoostState.camArrowShortcutEnabled,
          colorPointsByCuboid: viewBoostState.colorPointsByCuboid,
          dimOutsidePoints: viewBoostState.dimOutsidePoints,
          colorSelectedOnly: viewBoostState.colorSelectedOnly,
          showEgoDistance: viewBoostState.showEgoDistance
        })
      );
    } catch {}
  }

  function applyViewBrightnessCss() {
    const applyNow = () => {
      const target = document.head || document.documentElement;
      if (!target) return;
      let styleEl = document.getElementById("fusion3d-view-boost-style");
      if (!styleEl) {
        styleEl = document.createElement("style");
        styleEl.id = "fusion3d-view-boost-style";
        target.appendChild(styleEl);
      }
      if (Math.abs(viewBoostState.brightness - 100) < 0.1) {
        styleEl.textContent = "";
        return;
      }
      const factor = (viewBoostState.brightness / 100).toFixed(2);
      const selectors = [
        ".cvat-canvas3d-perspective canvas",
        ".cvat-canvas3d-topview canvas",
        ".cvat-canvas3d-sideview canvas",
        ".cvat-canvas3d-frontview canvas",
        ".cvat-canvas3d-orthographic-view canvas",
        '[class*="cvat-canvas3d"] canvas'
      ];
      if (viewBoostState.brightenCam2d) {
        selectors.push(
          ".cvat-context-image-wrapper img",
          ".cvat-context-image-wrapper canvas",
          '[class*="context-image"] img',
          '[class*="context-image"] canvas'
        );
      }
      styleEl.textContent = `${selectors.join(",\n")} {\n  filter: brightness(${factor}) !important;\n}`;
    };

    if (document.head || document.documentElement) {
      applyNow();
    } else {
      document.addEventListener("DOMContentLoaded", applyNow, { once: true });
    }
  }

  function nudge3dCanvasRedraw() {
    try {
      const canvases = document.querySelectorAll(
        '.cvat-canvas3d-perspective canvas, .cvat-canvas3d-topview canvas, .cvat-canvas3d-sideview canvas, .cvat-canvas3d-frontview canvas, [class*="cvat-canvas3d"] canvas'
      );
      for (const c of canvases) {
        c.dispatchEvent(new MouseEvent("mousemove", { bubbles: true, clientX: 1, clientY: 1 }));
      }
    } catch {}
  }

  function patchWebGLPointSize(proto) {
    if (!proto || proto.__fusionPointPatched) return;
    proto.__fusionPointPatched = true;

    const shaderMeta = new WeakMap();
    const programShaders = new WeakMap();

    const origShaderSource = proto.shaderSource;
    const origCompileShader = proto.compileShader;
    const origAttachShader = proto.attachShader;
    const origLinkProgram = proto.linkProgram;

    proto.shaderSource = function (shader, src) {
      if (typeof src === "string") {
        const patchedSize = applyPointSizePatch(src);
        shaderMeta.set(shader, { origSrc: src, sizePatchedSrc: patchedSize });
        return origShaderSource.call(this, shader, patchedSize);
      }
      return origShaderSource.call(this, shader, src);
    };

    proto.attachShader = function (program, shader) {
      let arr = programShaders.get(program);
      if (!arr) {
        arr = [];
        programShaders.set(program, arr);
      }
      if (!arr.includes(shader)) arr.push(shader);
      return origAttachShader.call(this, program, shader);
    };

    proto.linkProgram = function (program) {
      const shaders = programShaders.get(program);
      if (Array.isArray(shaders) && shaders.length >= 2) {
        let vs = null;
        let fs = null;
        for (const s of shaders) {
          try {
            const st = this.getShaderParameter(s, this.SHADER_TYPE);
            if (st === this.VERTEX_SHADER) vs = s;
            else if (st === this.FRAGMENT_SHADER) fs = s;
          } catch {}
        }
        const vsMeta = vs ? shaderMeta.get(vs) : null;
        const fsMeta = fs ? shaderMeta.get(fs) : null;
        if (
          vs &&
          fs &&
          vsMeta?.origSrc &&
          fsMeta?.origSrc &&
          /gl_PointSize\s*=/.test(vsMeta.origSrc) &&
          /\bposition\b/.test(vsMeta.origSrc)
        ) {
          const patchedVs = patchPointsVertexForBoxColor(vsMeta.sizePatchedSrc);
          const patchedFs = patchPointsFragmentForBoxColor(fsMeta.origSrc);
          if (patchedVs && patchedFs) {
            origShaderSource.call(this, vs, patchedVs);
            origCompileShader.call(this, vs);
            origShaderSource.call(this, fs, patchedFs);
            origCompileShader.call(this, fs);
            origLinkProgram.call(this, program);
            if (this.getProgramParameter(program, this.LINK_STATUS)) {
              viewBoostState.pointColorShaderPatched = true;
              return;
            }
            // Fallback an toan tuyet doi neu GPU/driver khong chap nhan shader da patch
            console.warn("[Fusion3D] Point color shader link fallback:", this.getProgramInfoLog(program));
            origShaderSource.call(this, vs, vsMeta.sizePatchedSrc);
            origCompileShader.call(this, vs);
            origShaderSource.call(this, fs, fsMeta.origSrc);
            origCompileShader.call(this, fs);
          }
        }
      }
      return origLinkProgram.call(this, program);
    };

    const locs = new WeakMap();
    const applyPointUniforms = (glCtx) => {
      const prog = glCtx.getParameter(glCtx.CURRENT_PROGRAM);
      if (!prog) return;
      let u = locs.get(prog);
      if (u === undefined) {
        u = {
          mul: glCtx.getUniformLocation(prog, "uFusionPointMul") || false,
          count: glCtx.getUniformLocation(prog, "uFusionBoxCount") || false,
          cy: glCtx.getUniformLocation(prog, "uFusionBoxCenterYaw") || false,
          hs: glCtx.getUniformLocation(prog, "uFusionBoxHalfSize") || false,
          col: glCtx.getUniformLocation(prog, "uFusionBoxColor") || false,
          dim: glCtx.getUniformLocation(prog, "uFusionOutsideDim") || false,
          proj: glCtx.getUniformLocation(prog, "projectionMatrix") || false,
          view: glCtx.getUniformLocation(prog, "viewMatrix") || false,
          mv: glCtx.getUniformLocation(prog, "modelViewMatrix") || false
        };
        locs.set(prog, u);
      }
      if (u.mul) glCtx.uniform1f(u.mul, viewBoostState.pointMul);
      if (u.count) {
        if (viewBoostState.colorPointsByCuboid && gpuBoxState.count > 0) {
          glCtx.uniform1i(u.count, gpuBoxState.count);
          if (u.cy) glCtx.uniform4fv(u.cy, gpuBoxState.centerYaw);
          if (u.hs) glCtx.uniform4fv(u.hs, gpuBoxState.halfSize);
          if (u.col) glCtx.uniform4fv(u.col, gpuBoxState.color);
          if (u.dim) glCtx.uniform1f(u.dim, viewBoostState.dimOutsidePoints ? 0.42 : 1.0);
        } else {
          glCtx.uniform1i(u.count, 0);
          if (u.dim) glCtx.uniform1f(u.dim, 1.0);
        }
      }
      if (viewBoostState.showEgoDistance && glCtx.canvas && u.proj && (u.view || u.mv)) {
        try {
          const projMat = glCtx.getUniform(prog, u.proj);
          const viewMat = (u.view && glCtx.getUniform(prog, u.view)) || glCtx.getUniform(prog, u.mv);
          if (projMat && viewMat && projMat.length === 16 && viewMat.length === 16) {
            egoDistState.viewports.set(glCtx.canvas, { projMat, viewMat });
            if (typeof scheduleEgoDistDraw === "function") scheduleEgoDistDraw();
          }
        } catch {}
      }
    };

    const origDrawArrays = proto.drawArrays;
    proto.drawArrays = function (mode, first, count) {
      if (mode === this.POINTS) applyPointUniforms(this);
      return origDrawArrays.call(this, mode, first, count);
    };

    const origDrawElements = proto.drawElements;
    if (typeof origDrawElements === "function") {
      proto.drawElements = function (mode, count, type, offset) {
        if (mode === this.POINTS) applyPointUniforms(this);
        return origDrawElements.call(this, mode, count, type, offset);
      };
    }
  }

  loadViewBoostFromLocalStorage();
  if (window.WebGLRenderingContext) patchWebGLPointSize(WebGLRenderingContext.prototype);
  if (window.WebGL2RenderingContext) patchWebGLPointSize(WebGL2RenderingContext.prototype);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", applyViewBrightnessCss, { once: true });
  } else {
    applyViewBrightnessCss();
  }

  let nonce = null;
  let cachedStore = null;

  document.addEventListener("fusion3d:init", () => {
    const v = document.documentElement.getAttribute("data-fusion3d-init");
    if (v) nonce = v;
  });
  document.dispatchEvent(new Event("fusion3d:ready"));

  function reply(payload) {
    if (nonce) {
      window.postMessage({ __fusion3d: true, nonce, source: "page", ...payload }, location.origin);
    }
  }

  function getStore() {
    if (cachedStore?.getState && cachedStore?.dispatch) return cachedStore;
    const els = [
      document.querySelector("#cvat-app"),
      document.querySelector("#root"),
      document.querySelector(".cvat-canvas3d-perspective"),
      document.querySelector(".cvat-canvas-container"),
      document.body
    ].filter(Boolean);

    for (const el of els) {
      for (const k of Object.keys(el)) {
        if (!k.startsWith("__reactFiber$") && !k.startsWith("__reactContainer$")) continue;
        let cur = el[k];
        let depth = 0;
        while (cur && depth++ < 300) {
          const s = cur.memoizedProps?.store || cur.stateNode?.store;
          if (s?.getState && s?.dispatch) return (cachedStore = s);
          cur = cur.return || cur.child;
        }
      }
    }
    return null;
  }

  function ctx() {
    const store = getStore();
    const state = store?.getState();
    return { store, state, job: state?.annotation?.job?.instance || null };
  }

  function parseJobIdFromUrl() {
    const m = location.pathname.match(/\/jobs\/(\d+)/);
    return m ? parseInt(m[1], 10) : null;
  }

  function parseTaskIdFromUrl() {
    const m = location.pathname.match(/\/tasks\/(\d+)/);
    return m ? parseInt(m[1], 10) : null;
  }

  function serializeObjectState(st) {
    if (!st) return null;
    return {
      clientID: st.clientID,
      serverID: st.serverID,
      frame: st.frame,
      objectType: st.objectType,
      shapeType: st.shapeType,
      labelName: st.label?.name || "",
      labelId: st.label?.id ?? null,
      points: Array.isArray(st.points) ? [...st.points] : [],
      occluded: Boolean(st.occluded),
      outside: Boolean(st.outside),
      keyframe: Boolean(st.keyframe),
      zOrder: st.zOrder ?? 0,
      attributes: st.attributes ? { ...st.attributes } : {}
    };
  }

  async function syncUi(frame, updatedStates = null) {
    const { store, state, job } = ctx();
    if (!store || !job) return;
    const f = typeof frame === "number" ? frame : state.annotation?.player?.frame?.number ?? 0;
    const showAll = !!state.settings?.workspace?.showAllInterpolationTracks;
    const filters = state.annotation?.annotations?.filters || [];
    const states = await job.annotations.get(f, showAll, filters);
    const history = await job.actions.get();
    const activeId = state.annotation?.annotations?.activatedStateID ?? null;

    if (Array.isArray(updatedStates) && updatedStates.length > 0) {
      store.dispatch({
        type: "UPDATE_ANNOTATIONS_SUCCESS",
        payload: { states: updatedStates, history }
      });
    }

    store.dispatch({
      type: "FETCH_ANNOTATIONS_SUCCESS",
      payload: { states, intervals: [], history }
    });

    const canvas = state.annotation?.canvas?.instance;
    const data = state.annotation?.player?.frame?.data;
    try {
      if (canvas?.setup && data) {
        canvas.setup(data, states);
      }
      if (canvas?.activate && activeId != null) {
        canvas.activate(null);
        canvas.activate(activeId);
      }
    } catch {}

    if (typeof renderCamOverlaysNow === "function" && camOverlayState?.enabled) {
      renderCamOverlaysNow();
    }
    if (typeof refreshGpuCuboidBoxes === "function" && viewBoostState.colorPointsByCuboid) {
      refreshGpuCuboidBoxes(true);
    }
    if (typeof refreshEgoDistanceData === "function" && viewBoostState.showEgoDistance) {
      refreshEgoDistanceData();
    }
  }

  function jumpToFrame(frame) {
    const frameNum = Number(frame);
    if (!Number.isFinite(frameNum)) throw new Error("Frame không hợp lệ");
    const { store, state, job } = ctx();
    if (job && (frameNum < job.startFrame || frameNum > job.stopFrame)) {
      throw new Error(`Frame ${frameNum} ngoài phạm vi job [${job.startFrame}..${job.stopFrame}]`);
    }

    const input =
      document.querySelector(".cvat-player-frame-selector input") ||
      document.querySelector('input[aria-label="Frame number"]') ||
      document.querySelector(".cvat-player-controls input.ant-input-number-input");

    if (input) {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
      if (setter) setter.call(input, String(frameNum));
      else input.value = String(frameNum);
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new Event("change", { bubbles: true }));
      input.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Enter", code: "Enter", keyCode: 13, bubbles: true })
      );
      input.blur();
      return { ok: true, method: "player_input", frame: frameNum };
    }

    if (store && state?.annotation?.player) {
      store.dispatch({ type: "CHANGE_FRAME_SUCCESS", payload: { number: frameNum } });
      return { ok: true, method: "redux_fallback", frame: frameNum };
    }
    throw new Error("Không tìm thấy ô chuyển frame trên giao diện CVAT 3D");
  }

  async function selectObjectById(serverId, clientId, frame) {
    if (typeof frame === "number") {
      jumpToFrame(frame);
      await new Promise((r) => setTimeout(r, 280));
    }

    const { store, state, job } = ctx();
    if (!store || !job) {
      return { ok: false, error: "Chưa kết nối được jobInstance" };
    }

    const currentFrame =
      typeof frame === "number" ? frame : state.annotation?.player?.frame?.number ?? 0;

    const states = await job.annotations.get(currentFrame, true);
    const matched = states.find(
      (st) =>
        (clientId != null && st.clientID === clientId) ||
        (serverId != null && st.serverID === serverId)
    );

    if (!matched) {
      return { ok: false, error: `Không tìm thấy đối tượng trên frame ${currentFrame}` };
    }

    const canvas = state.annotation?.canvas?.instance;
    if (canvas && typeof canvas.activate === "function") {
      try {
        canvas.activate(matched.clientID);
      } catch {}
    }

    store.dispatch({
      type: "ACTIVATE_OBJECT",
      payload: {
        activatedStateID: matched.clientID,
        activatedElementID: null,
        activatedAttributeID: null
      }
    });

    const sidebarCard = document.querySelector(`#cvat-objects-sidebar-state-item-${matched.clientID}`);
    if (sidebarCard && typeof sidebarCard.scrollIntoView === "function") {
      sidebarCard.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    return { ok: true, object: serializeObjectState(matched) };
  }

  async function findState(job, frame, clientID) {
    const states = await job.annotations.get(frame, true);
    return states.find((s) => s.clientID === clientID);
  }

  function ensure16Points(pts, allowPitchRoll = false) {
    if (!Array.isArray(pts) || pts.length < 9) throw new Error("Mảng points cuboid < 9 phần tử");
    const out = pts.length >= 16 ? pts.slice(0, 16) : [...pts.slice(0, 9), 0, 0, 0, 0, 0, 0, 0];
    if (!allowPitchRoll) {
      out[3] = 0;
      out[4] = 0;
    }
    return out;
  }

  async function injectTracks({ tracks, objectType = "track", allowPitchRoll = false }) {
    const { state, job } = ctx();
    if (!job) throw new Error("Chưa kết nối được jobInstance trên trang CVAT 3D");
    const OS = window.cvat?.classes?.ObjectState;
    if (!OS) throw new Error("Không tìm thấy window.cvat.classes.ObjectState");

    const mode = String(objectType || "track").toLowerCase() === "shape" ? "shape" : "track";
    const labels = job.labels || [];
    const minF = job.startFrame ?? 0;
    const maxF = job.stopFrame ?? Infinity;
    const report = { created: 0, keyframes: 0, outsideCount: 0, objectType: mode, skipped: [], clientIDs: [] };

    if (mode === "shape") {
      const shapesToCreate = [];
      for (const t of tracks) {
        const lbl = labels.find(
          (l) => l.name && l.name.toLowerCase() === String(t.className || "").toLowerCase()
        );
        if (!lbl) {
          report.skipped.push(`Box #${t.track_id}: job không có nhãn "${t.className}"`);
          continue;
        }
        const kfs = [...(t.keyframes || [])]
          .filter((k) => k.frame >= minF && k.frame <= maxF)
          .sort((a, b) => a.frame - b.frame);
        if (kfs.length === 0) {
          report.skipped.push(`Box #${t.track_id}: không có frame trong phạm vi job`);
          continue;
        }
        for (const kf of kfs) {
          shapesToCreate.push(
            new OS({
              frame: kf.frame,
              label: lbl,
              objectType: window.cvat?.enums?.ObjectType?.SHAPE || "shape",
              shapeType: window.cvat?.enums?.ShapeType?.CUBOID || "cuboid",
              source: window.cvat?.enums?.Source?.AUTO || "auto",
              points: ensure16Points(kf.points, allowPitchRoll),
              occluded: Boolean(kf.occluded),
              outside: false,
              attributes: {},
              zOrder: 0
            })
          );
        }
      }
      if (shapesToCreate.length > 0) {
        const ids = await job.annotations.put(shapesToCreate);
        report.created = shapesToCreate.length;
        report.keyframes = shapesToCreate.length;
        report.clientIDs.push(...ids);
      }
      await syncUi(state.annotation?.player?.frame?.number);
      return report;
    }

    for (const t of tracks) {
      const lbl = labels.find(
        (l) => l.name && l.name.toLowerCase() === String(t.className || "").toLowerCase()
      );
      if (!lbl) {
        report.skipped.push(`Track #${t.track_id}: job không có nhãn "${t.className}"`);
        continue;
      }

      const kfs = [...(t.keyframes || [])]
        .filter((k) => k.frame >= minF && k.frame <= maxF)
        .sort((a, b) => a.frame - b.frame);
      if (kfs.length === 0) {
        report.skipped.push(`Track #${t.track_id}: không có keyframe trong phạm vi job`);
        continue;
      }

      const first = kfs[0];
      const os = new OS({
        frame: first.frame,
        label: lbl,
        objectType: window.cvat?.enums?.ObjectType?.TRACK || "track",
        shapeType: window.cvat?.enums?.ShapeType?.CUBOID || "cuboid",
        source: window.cvat?.enums?.Source?.AUTO || "auto",
        points: ensure16Points(first.points, allowPitchRoll),
        occluded: Boolean(first.occluded),
        outside: false,
        attributes: {},
        zOrder: 0
      });

      const [clientID] = await job.annotations.put([os]);
      report.created++;
      report.keyframes++;
      report.clientIDs.push(clientID);

      const kfFrameSet = new Set(kfs.map((k) => k.frame));
      const events = [
        ...kfs.slice(1).map((k) => ({
          frame: k.frame,
          points: ensure16Points(k.points, allowPitchRoll),
          occluded: Boolean(k.occluded),
          outside: false
        })),
        ...(t.outside_frames || [])
          .filter((f) => f >= minF && f <= maxF && !kfFrameSet.has(f))
          .map((f) => ({ frame: f, outside: true }))
      ].sort((a, b) => a.frame - b.frame);

      for (const ev of events) {
        const st = await findState(job, ev.frame, clientID);
        if (!st) {
          report.skipped.push(`Track #${t.track_id} frame ${ev.frame}: không tìm thấy state`);
          continue;
        }
        if (ev.outside) {
          // Luật 7 v4: TUYỆT ĐỐI không bật occluded và outside cùng lúc!
          st.occluded = false;
          st.outside = true;
          await st.save();
          report.outsideCount++;
        } else {
          st.outside = false;
          st.occluded = Boolean(ev.occluded);
          st.points = ev.points;
          await st.save();
          report.keyframes++;
        }
      }
    }

    await syncUi(state.annotation?.player?.frame?.number);
    return report;
  }

  // --- V2 Controlled Assist: Ghi qua API chuẩn ObjectState.save() ---
  async function applySafeEdit(action) {
    const { state, job } = ctx();
    if (!job) throw new Error("Không tìm thấy jobInstance trong Redux state.");
    const currentFrame = state.annotation?.player?.frame?.number ?? 0;

    if (action.kind === "LOCK_TRACK_SIZE_KEYFRAMES") {
      const { serverId, clientId, targetDims, keyframes } = action;
      if (!Array.isArray(keyframes) || keyframes.length === 0) {
        throw new Error("Không có keyframe nào được chọn để khóa kích thước.");
      }

      const updatedFrames = [];
      const skippedFrames = [];

      for (const f of keyframes) {
        const states = await job.annotations.get(f, true);
        const st = states.find(
          (item) =>
            (clientId != null && item.clientID === clientId) ||
            (serverId != null && item.serverID === serverId)
        );
        if (!st) {
          skippedFrames.push({ frame: f, reason: "Không tìm thấy state" });
          continue;
        }
        if (!st.keyframe) {
          skippedFrames.push({ frame: f, reason: "Không phải keyframe thật (bỏ qua)" });
          continue;
        }
        if (st.outside) {
          skippedFrames.push({ frame: f, reason: "Frame đang outside=true" });
          continue;
        }

        const pts = [...st.points];
        if (pts.length < 9) {
          skippedFrames.push({ frame: f, reason: "Mảng points < 9 phần tử" });
          continue;
        }

        const oldH = pts[8];
        const newH = targetDims.dz;
        const deltaH = newH - oldH;
        pts[2] = pts[2] + deltaH / 2;
        pts[6] = targetDims.dx;
        pts[7] = targetDims.dy;
        pts[8] = newH;

        st.points = pts;
        await st.save();
        updatedFrames.push(f);
      }

      await syncUi(currentFrame);
      return { ok: true, updatedFrames, skippedFrames };
    }

    if (action.kind === "RESOLVE_OCCLUDED_OUTSIDE") {
      const { serverId, clientId, frame, turnOff } = action;
      const states = await job.annotations.get(frame, true);
      const st = states.find(
        (item) =>
          (clientId != null && item.clientID === clientId) ||
          (serverId != null && item.serverID === serverId)
      );
      if (!st) throw new Error(`Không tìm thấy đối tượng trên frame ${frame}`);

      if (turnOff === "occluded") st.occluded = false;
      else if (turnOff === "outside") st.outside = false;
      else throw new Error("Tham số turnOff không hợp lệ");

      await st.save();
      await syncUi(currentFrame);
      return { ok: true, frame, turnOff };
    }

    if (action.kind === "FLIP_YAW_180" || action.kind === "ROTATE_BARRIER_90") {
      const { serverId, clientId, frame } = action;
      const targetF = typeof frame === "number" ? frame : currentFrame;
      const states = await job.annotations.get(targetF, true);
      const st = states.find(
        (item) =>
          (clientId != null && item.clientID === clientId) ||
          (serverId != null && item.serverID === serverId)
      );
      if (!st) throw new Error(`Không tìm thấy đối tượng trên frame ${targetF}`);

      const pts = [...st.points];
      if (pts.length < 9) throw new Error("Cuboid points không hợp lệ (< 9 phần tử)");

      const wrapPi = (rad) => {
        let a = rad;
        while (a > Math.PI) a -= 2 * Math.PI;
        while (a < -Math.PI) a += 2 * Math.PI;
        return a;
      };

      const oldYawDeg = Number(((pts[5] * 180) / Math.PI).toFixed(1));
      const oldL = Number(pts[6].toFixed(2));
      const oldW = Number(pts[7].toFixed(2));

      if (action.kind === "FLIP_YAW_180") {
        pts[5] = wrapPi(pts[5] + Math.PI);
      } else {
        pts[5] = wrapPi(pts[5] + Math.PI / 2);
        const tmp = pts[6];
        pts[6] = pts[7];
        pts[7] = tmp;
      }

      st.points = pts;
      const saved = await st.save();
      await syncUi(currentFrame, saved ? [saved] : null);
      return {
        ok: true,
        frame: targetF,
        clientID: st.clientID,
        oldYawDeg,
        newYawDeg: Number(((pts[5] * 180) / Math.PI).toFixed(1)),
        oldDims: `${oldL}×${oldW}`,
        newDims: `${Number(pts[6].toFixed(2))}×${Number(pts[7].toFixed(2))}`,
        newPoints: pts
      };
    }

    if (action.kind === "SNAP_TO_GROUND") {
      const { serverId, clientId, frame, zGround } = action;
      if (!Number.isFinite(zGround)) throw new Error("zGround không hợp lệ");
      const targetF = typeof frame === "number" ? frame : currentFrame;
      const states = await job.annotations.get(targetF, true);
      const st = states.find(
        (item) =>
          (clientId != null && item.clientID === clientId) ||
          (serverId != null && item.serverID === serverId)
      );
      if (!st) throw new Error(`Không tìm thấy đối tượng trên frame ${targetF}`);

      const pts = [...st.points];
      if (pts.length < 9) throw new Error("Cuboid points không hợp lệ");
      pts[2] = zGround + pts[8] / 2;

      st.points = pts;
      await st.save();
      await syncUi(currentFrame);
      return { ok: true, frame: targetF, newZ: pts[2], zGround };
    }

    throw new Error(`Không hỗ trợ hành động V2: ${action.kind}`);
  }

  // --- Bộ kiểm chứng Day 0 trực tiếp trên job CVAT 3D ---
  async function runDay0Probe() {
    const { store, state, job } = ctx();
    const report = {
      timestamp: new Date().toISOString(),
      url: location.href,
      reduxStoreFound: Boolean(store),
      jobInstanceFound: Boolean(job),
      windowCvatFound: Boolean(window.cvat),
      objectStateClassFound: Boolean(window.cvat?.classes?.ObjectState),
      jobInfo: null,
      pointsSample: null,
      barrierSamples: [],
      allTracksComparison: null,
      objectStateMethods: []
    };

    if (!job) return report;

    const currentFrame = state.annotation?.player?.frame?.number ?? job.startFrame ?? 0;
    report.jobInfo = {
      id: job.id,
      taskId: job.taskId ?? parseTaskIdFromUrl(),
      dimension: job.dimension || "unknown",
      startFrame: job.startFrame,
      stopFrame: job.stopFrame,
      currentFrame,
      labels: (job.labels || []).map((l) => ({ id: l.id, name: l.name, type: l.type }))
    };

    try {
      const statesDefault = await job.annotations.get(currentFrame, false);
      const statesAll = await job.annotations.get(currentFrame, true);
      report.allTracksComparison = {
        frame: currentFrame,
        defaultCount: statesDefault.length,
        allTracksTrueCount: statesAll.length,
        outsideCountInAllTracks: statesAll.filter((s) => s.outside).length
      };

      const firstCuboid = statesAll.find((s) => Array.isArray(s.points) && s.points.length >= 9);
      if (firstCuboid) {
        report.pointsSample = {
          label: firstCuboid.label?.name,
          shapeType: firstCuboid.shapeType,
          objectType: firstCuboid.objectType,
          pointsLength: firstCuboid.points.length,
          points: firstCuboid.points.slice(0, 16),
          interpreted: {
            x: firstCuboid.points[0],
            y: firstCuboid.points[1],
            z: firstCuboid.points[2],
            rx: firstCuboid.points[3],
            ry: firstCuboid.points[4],
            rz: firstCuboid.points[5],
            dx: firstCuboid.points[6],
            dy: firstCuboid.points[7],
            dz: firstCuboid.points[8]
          }
        };
        report.objectStateMethods = Object.getOwnPropertyNames(
          Object.getPrototypeOf(firstCuboid)
        ).filter((m) => typeof firstCuboid[m] === "function");
      }

      const hasBarrierLabel = (job.labels || []).some((l) =>
        /barrier|barricade|guardrail|rao_chan/i.test(l.name || "")
      );
      if (hasBarrierLabel) {
        const barriers = statesAll.filter(
          (s) => /barrier|barricade|guardrail|rao_chan/i.test(s.label?.name || "") && s.points?.length >= 9
        );
        report.barrierSamples = barriers.slice(0, 5).map((b) => ({
          serverID: b.serverID,
          clientID: b.clientID,
          label: b.label?.name,
          dx: Number(b.points[6].toFixed(3)),
          dy: Number(b.points[7].toFixed(3)),
          dz: Number(b.points[8].toFixed(3)),
          rzDeg: Number(((b.points[5] * 180) / Math.PI).toFixed(1)),
          ratioDxOverDy: Number((b.points[6] / (b.points[7] || 1e-6)).toFixed(2))
        }));
      } else {
        delete report.barrierSamples;
      }
    } catch (err) {
      report.probeError = err.message;
    }

    return report;
  }

  // --- 7. Realtime Camera 2D Projection Overlay (trực tiếp trên các khung ảnh cam0 của CVAT) ---
  const BOX_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
  ];
  const FRONT_FACE_EDGES = new Set(["0-1", "4-5", "0-4", "1-5"]);
  const EDGE_STEPS = 10;
  const MIN_CAM_DEPTH = 0.3;

  const camOverlayState = {
    enabled: false,
    calibsRaw: {},       // { s01: rawJson, default: rawJson }
    calibsCompiled: {},  // { s01: { cam01: compiledCam, ... } }
    frameNames: {},      // { 0: "pointcloud/s01f39.pcd", ... }
    showLabels: true,
    selectedOnly: false,
    unsubscribe: null,
    pollTimer: null,
    rafId: null,
    lastStatus: { drawnPanels: 0, matchedCams: [], boxesDrawn: 0 }
  };

  function invertRigid4x4(t) {
    const r = [
      [t[0][0], t[1][0], t[2][0]],
      [t[0][1], t[1][1], t[2][1]],
      [t[0][2], t[1][2], t[2][2]]
    ];
    const p = [
      -(r[0][0] * t[0][3] + r[0][1] * t[1][3] + r[0][2] * t[2][3]),
      -(r[1][0] * t[0][3] + r[1][1] * t[1][3] + r[1][2] * t[2][3]),
      -(r[2][0] * t[0][3] + r[2][1] * t[1][3] + r[2][2] * t[2][3])
    ];
    return [
      [r[0][0], r[0][1], r[0][2], p[0]],
      [r[1][0], r[1][1], r[1][2], p[1]],
      [r[2][0], r[2][1], r[2][2], p[2]],
      [0, 0, 0, 1]
    ];
  }

  function computeMaxRadius2(dist, searchTo = 1e4) {
    const k1 = dist[0] || 0;
    const k2 = dist[1] || 0;
    const k3 = dist[4] || 0;
    const slope = (r2) => 1 + 3 * k1 * r2 + 5 * k2 * r2 * r2 + 7 * k3 * r2 * r2 * r2;
    const qa = 21 * k3;
    const qb = 10 * k2;
    const qc = 3 * k1;
    let crit = [];
    if (Math.abs(qa) > 1e-12) {
      const disc = qb * qb - 4 * qa * qc;
      if (disc >= 0) {
        const s = Math.sqrt(disc);
        crit = [(-qb - s) / (2 * qa), (-qb + s) / (2 * qa)];
      }
    } else if (Math.abs(qb) > 1e-12) {
      crit = [-qc / qb];
    }
    const ends = [...new Set([...crit.filter((c) => c > 0 && c < searchTo), searchTo])].sort((a, b) => a - b);
    let lo = 0;
    for (const hi of ends) {
      if (slope(hi) <= 1e-12) {
        let a = lo;
        let b = hi;
        for (let i = 0; i < 60; i++) {
          const mid = (a + b) * 0.5;
          if (slope(mid) > 1e-12) a = mid;
          else b = mid;
        }
        return a;
      }
      lo = hi;
    }
    return Infinity;
  }

  function compileCalibEntry(raw) {
    if (!raw || typeof raw !== "object") return {};
    const out = {};
    const imgSize = raw.meta_info?.image_size || [1920, 1020];
    const camDict = raw.calib || raw.cameras || raw;
    for (const [camName, cal] of Object.entries(camDict)) {
      if (!cal || typeof cal !== "object") continue;
      // Format 1: Data-VF (cam_intrinsic + cam_to_velo + distortion)
      if (Array.isArray(cal.cam_intrinsic) && Array.isArray(cal.cam_to_velo)) {
        const d = [...(cal.distortion || [0, 0, 0, 0, 0]), 0, 0, 0, 0, 0].slice(0, 5);
        out[camName] = {
          T: invertRigid4x4(cal.cam_to_velo),
          K: cal.cam_intrinsic,
          D: d,
          size: cal.image_size || imgSize,
          r2max: computeMaxRadius2(d)
        };
      }
      // Format 2: nuScenes / edufun.ai (lidar2cam / extrinsic + cam2img / intrinsic)
      else if ((Array.isArray(cal.lidar2cam) || Array.isArray(cal.extrinsic)) && (Array.isArray(cal.cam2img) || Array.isArray(cal.intrinsic))) {
        const T = cal.lidar2cam || cal.extrinsic;
        const K = cal.cam2img || cal.intrinsic;
        const d = [...(cal.distortion || [0, 0, 0, 0, 0]), 0, 0, 0, 0, 0].slice(0, 5);
        out[camName] = {
          T,
          K,
          D: d,
          size: cal.image_size || imgSize,
          r2max: computeMaxRadius2(d)
        };
      }
    }
    return out;
  }

  function projectPointToCam(pt, cam) {
    const T = cam.T;
    const xc = T[0][0] * pt[0] + T[0][1] * pt[1] + T[0][2] * pt[2] + T[0][3];
    const yc = T[1][0] * pt[0] + T[1][1] * pt[1] + T[1][2] * pt[2] + T[1][3];
    const zc = T[2][0] * pt[0] + T[2][1] * pt[1] + T[2][2] * pt[2] + T[2][3];
    if (zc < MIN_CAM_DEPTH) return null;
    const a = xc / zc;
    const b = yc / zc;
    const r2 = a * a + b * b;
    if (r2 > cam.r2max) return null;
    const [k1, k2, p1, p2, k3] = cam.D;
    const radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    const ad = a * radial + 2 * p1 * a * b + p2 * (r2 + 2 * a * a);
    const bd = b * radial + p1 * (r2 + 2 * b * b) + 2 * p2 * a * b;
    const K = cam.K;
    return [K[0][0] * ad + K[0][1] * bd + K[0][2], K[1][1] * bd + K[1][2]];
  }

  function getCuboidCorners(pts) {
    const [x, y, z, , , rz, dx, dy, dz] = pts;
    const hl = Math.abs(dx) * 0.5;
    const hw = Math.abs(dy) * 0.5;
    const hh = Math.abs(dz) * 0.5;
    const c = Math.cos(rz);
    const s = Math.sin(rz);
    // 0: front-left-bottom, 1: front-right-bottom, 2: back-right-bottom, 3: back-left-bottom, 4..7: top
    const local = [
      [hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]
    ];
    const bot = local.map(([lx, ly]) => [x + c * lx - s * ly, y + s * lx + c * ly, z - hh]);
    const top = local.map(([lx, ly]) => [x + c * lx - s * ly, y + s * lx + c * ly, z + hh]);
    return [...bot, ...top];
  }

  function projectCuboidToCam(pts, cam) {
    const [width, height] = cam.size;
    const corners = getCuboidCorners(pts);
    const pieces = [];
    for (const [i, j] of BOX_EDGES) {
      const a = corners[i];
      const b = corners[j];
      const isFront = FRONT_FACE_EDGES.has(`${i}-${j}`);
      let line = [];
      for (let s = 0; s <= EDGE_STEPS; s++) {
        const f = s / EDGE_STEPS;
        const uv = projectPointToCam(
          [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])],
          cam
        );
        if (!uv) {
          if (line.length > 1) pieces.push({ pts: line, isFront });
          line = [];
        } else {
          line.push(uv);
        }
      }
      if (line.length > 1) pieces.push({ pts: line, isFront });
    }
    const margin = 0.5;
    let seen = false;
    let inside = false;
    let topAnchor = null;
    for (const piece of pieces) {
      for (const [u, v] of piece.pts) {
        if (u >= -margin * width && u <= (1 + margin) * width && v >= -margin * height && v <= (1 + margin) * height) {
          seen = true;
        }
        if (u >= 0 && u < width && v >= 0 && v < height) {
          inside = true;
          if (!topAnchor || v < topAnchor[1]) topAnchor = [u, v];
        }
      }
    }
    return seen && inside ? { pieces, topAnchor } : null;
  }

  function pickCalibForFrame(frameNum) {
    const compiled = camOverlayState.calibsCompiled;
    const keys = Object.keys(compiled);
    if (!keys.length) return null;
    const fname = String(camOverlayState.frameNames[frameNum] || "");
    const m = fname.match(/s(\d{2})f\d+/i) || fname.match(/sequence[_-]?(\d+)/i);
    if (m) {
      const skey = `s${String(Number(m[1])).padStart(2, "0")}`;
      if (compiled[skey]) return compiled[skey];
    }
    return compiled.default || compiled[keys[0]];
  }

  function findContextCameraPanels(calibCams) {
    const camNames = Object.keys(calibCams).sort();
    if (!camNames.length) return [];

    const candidates = [];

    // Cách 1: CVAT 3D đặt mỗi panel cam vào .react-grid-item (có góc kéo ⌟ ở dưới phải)
    const gridItems = Array.from(document.querySelectorAll(".react-grid-item"));
    for (const item of gridItems) {
      // Bỏ qua Perspective / Top / Side / Front
      if (item.querySelector(".cvat-canvas3d-perspective, .cvat-canvas3d-topview, .cvat-canvas3d-sideview, .cvat-canvas3d-frontview")) {
        continue;
      }
      const rawText = (item.textContent || "").trim();
      const lowerText = rawText.toLowerCase();
      if (/^(top|side|front)$/i.test(rawText)) continue;

      const media =
        item.querySelector("img") ||
        item.querySelector("canvas:not(.fusion3d-cam-overlay)");
      if (!media) continue;
      const r = media.getBoundingClientRect();
      if (r.width < 50 || r.height < 30) continue;

      const titleAttr =
        item.getAttribute("title") ||
        item.querySelector("[title]")?.getAttribute("title") ||
        media.getAttribute("alt") ||
        media.getAttribute("title") ||
        "";
      candidates.push({
        el: media,
        hint: `${lowerText} ${titleAttr.toLowerCase()}`,
        top: r.top,
        left: r.left
      });
    }

    // Cách 2: Fallback quét tất cả <img> hiển thị ngoài header nếu giao diện không dùng .react-grid-item
    if (candidates.length === 0) {
      const allImgs = Array.from(document.querySelectorAll("img")).filter((img) => {
        if (img.closest("header, .cvat-header, .ant-layout-header, .cvat-objects-sidebar")) return false;
        const r = img.getBoundingClientRect();
        return r.width >= 60 && r.height >= 40;
      });
      for (const img of allImgs) {
        const panel = img.parentElement?.parentElement || img.parentElement;
        const r = img.getBoundingClientRect();
        candidates.push({
          el: img,
          hint: `${panel?.textContent || ""} ${img.alt || ""} ${img.title || ""}`.toLowerCase(),
          top: r.top,
          left: r.left
        });
      }
    }

    if (!candidates.length) return [];

    // Sắp xếp theo thứ tự đọc từ trên xuống, trái sang phải
    candidates.sort((a, b) => (Math.abs(a.top - b.top) > 25 ? a.top - b.top : a.left - b.left));

    const matched = [];
    const usedCams = new Set();

    // Bước 1: Khớp theo tên cam trong text/title (vd: cam01, cam03, CAM_FRONT...)
    for (const c of candidates) {
      const foundCam = camNames.find((cn) => !usedCams.has(cn) && c.hint.includes(cn.toLowerCase()));
      if (foundCam) {
        usedCams.add(foundCam);
        matched.push({ img: c.el, camName: foundCam });
      } else {
        matched.push({ img: c.el, camName: null });
      }
    }

    // Bước 2: Fallback khớp theo thứ tự alphabet của calib
    const remainingCams = camNames.filter((cn) => !usedCams.has(cn));
    let remIdx = 0;
    for (const item of matched) {
      if (!item.camName && remIdx < remainingCams.length) {
        item.camName = remainingCams[remIdx++];
      }
    }

    return matched.filter((m) => Boolean(m.camName));
  }

  function clearCamOverlays() {
    document.querySelectorAll("canvas.fusion3d-cam-overlay").forEach((c) => c.remove());
  }

  function getOrCreateOverlayCanvas(img) {
    const parent = img.parentElement;
    if (!parent) return null;
    if (getComputedStyle(parent).position === "static") {
      parent.style.position = "relative";
    }
    let cvs = parent.querySelector("canvas.fusion3d-cam-overlay");
    if (!cvs) {
      cvs = document.createElement("canvas");
      cvs.className = "fusion3d-cam-overlay";
      cvs.style.position = "absolute";
      cvs.style.pointerEvents = "none";
      cvs.style.zIndex = "25";
      parent.appendChild(cvs);
    }
    return cvs;
  }

  function computeImageDrawRect(img, calibW, calibH) {
    const w = img.clientWidth;
    const h = img.clientHeight;
    const natW = img.naturalWidth || img.width || calibW || w;
    const natH = img.naturalHeight || img.height || calibH || h;
    const fit = getComputedStyle(img).objectFit || "fill";
    let drawW = w;
    let drawH = h;
    let offX = 0;
    let offY = 0;
    if (fit === "contain" && natW > 0 && natH > 0) {
      const scale = Math.min(w / natW, h / natH);
      drawW = natW * scale;
      drawH = natH * scale;
      offX = (w - drawW) * 0.5;
      offY = (h - drawH) * 0.5;
    }
    return {
      left: img.offsetLeft,
      top: img.offsetTop,
      w,
      h,
      offX,
      offY,
      sx: drawW / (calibW || natW || 1),
      sy: drawH / (calibH || natH || 1)
    };
  }

  async function renderCamOverlaysNow() {
    if (!camOverlayState.enabled) return;
    const { state, job } = ctx();
    if (!state || !job) return;

    const currentFrame = state.annotation?.player?.frame?.number ?? job.startFrame ?? 0;
    const calibCams = pickCalibForFrame(currentFrame);
    if (!calibCams) return;

    const panels = findContextCameraPanels(calibCams);
    if (!panels.length) return;

    let states = [];
    try {
      states = await job.annotations.get(currentFrame, true);
    } catch {
      return;
    }

    const activeId = state.annotation?.annotations?.activatedStateID ?? null;
    const boxes = states.filter((st) => {
      if (st.outside || !Array.isArray(st.points) || st.points.length < 9) return false;
      if (camOverlayState.selectedOnly && activeId != null && st.clientID !== activeId) return false;
      return true;
    });

    let totalDrawn = 0;
    const matchedNames = [];
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    for (const { img, camName } of panels) {
      const cam = calibCams[camName];
      if (!cam) continue;
      matchedNames.push(camName);
      const cvs = getOrCreateOverlayCanvas(img);
      if (!cvs) continue;

      const rect = computeImageDrawRect(img, cam.size[0], cam.size[1]);
      if (rect.w <= 0 || rect.h <= 0) continue;

      cvs.style.left = `${rect.left}px`;
      cvs.style.top = `${rect.top}px`;
      cvs.style.width = `${rect.w}px`;
      cvs.style.height = `${rect.h}px`;

      const targetW = Math.round(rect.w * dpr);
      const targetH = Math.round(rect.h * dpr);
      if (cvs.width !== targetW || cvs.height !== targetH) {
        cvs.width = targetW;
        cvs.height = targetH;
      }

      const g = cvs.getContext("2d");
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, rect.w, rect.h);

      for (const st of boxes) {
        const proj = projectCuboidToCam(st.points, cam);
        if (!proj) continue;
        totalDrawn++;
        const isSel = activeId != null && st.clientID === activeId;
        const baseColor = st.label?.color || "#38bdf8";

        for (const piece of proj.pieces) {
          g.beginPath();
          piece.pts.forEach(([u, v], idx) => {
            const px = rect.offX + u * rect.sx;
            const py = rect.offY + v * rect.sy;
            if (idx === 0) g.moveTo(px, py);
            else g.lineTo(px, py);
          });
          g.strokeStyle = isSel ? "#ffff00" : piece.isFront ? "#ffffff" : baseColor;
          g.lineWidth = isSel ? 2.4 : piece.isFront ? 1.8 : 1.4;
          g.stroke();
        }

        if (camOverlayState.showLabels && proj.topAnchor) {
          const lx = Math.max(2, Math.min(rect.w - 46, rect.offX + proj.topAnchor[0] * rect.sx));
          const ly = Math.max(11, Math.min(rect.h - 4, rect.offY + proj.topAnchor[1] * rect.sy - 3));
          const distSuffix = viewBoostState.showEgoDistance
            ? ` · ${Math.hypot(st.points[0] || 0, st.points[1] || 0).toFixed(1)}m`
            : "";
          const tag = `#${st.serverID ?? st.clientID} ${st.label?.name || ""}${distSuffix}`;
          g.font = "bold 10px sans-serif";
          const tw = g.measureText(tag).width + 6;
          g.fillStyle = isSel ? "rgba(234,179,8,0.9)" : "rgba(15,23,42,0.78)";
          g.fillRect(lx - 2, ly - 10, tw, 12);
          g.fillStyle = isSel ? "#000" : "#fff";
          g.fillText(tag, lx + 1, ly - 1);
        }
      }
    }

    camOverlayState.lastStatus = {
      drawnPanels: panels.length,
      matchedCams: matchedNames,
      boxesDrawn: totalDrawn
    };
  }

  function scheduleCamOverlayRender() {
    if (!camOverlayState.enabled || camOverlayState.rafId) return;
    camOverlayState.rafId = requestAnimationFrame(() => {
      camOverlayState.rafId = null;
      renderCamOverlaysNow();
    });
  }

  async function refreshGpuCuboidBoxes(forceRedraw = false) {
    if (!viewBoostState.colorPointsByCuboid) {
      if (gpuBoxState.count !== 0) {
        gpuBoxState.count = 0;
        gpuBoxState.lastSig = "";
        nudge3dCanvasRedraw();
      }
      return;
    }
    if (gpuBoxState.refreshPending) return;
    gpuBoxState.refreshPending = true;
    try {
      const { state, job } = ctx();
      if (!state || !job) return;
      const currentFrame = state.annotation?.player?.frame?.number ?? job.startFrame ?? 0;
      const activeId = state.annotation?.annotations?.activatedStateID ?? null;
      let states = [];
      try {
        states = await job.annotations.get(currentFrame, true);
      } catch {
        return;
      }

      let boxes = states.filter((st) => {
        if (st.outside || !Array.isArray(st.points) || st.points.length < 9) return false;
        if (viewBoostState.colorSelectedOnly && activeId != null && st.clientID !== activeId) {
          return false;
        }
        return true;
      });

      // Ưu tiên box đang chọn lên vị trí đầu tiên (nếu 2 box chồng nhau thì màu box đang chọn thắng), sau đó theo khoảng cách gần LiDAR
      boxes.sort((a, b) => {
        const aSel = activeId != null && a.clientID === activeId ? 0 : 1;
        const bSel = activeId != null && b.clientID === activeId ? 0 : 1;
        if (aSel !== bSel) return aSel - bSel;
        const da = a.points[0] * a.points[0] + a.points[1] * a.points[1];
        const db = b.points[0] * b.points[0] + b.points[1] * b.points[1];
        return da - db;
      });

      const count = Math.min(MAX_GPU_BOXES, boxes.length);
      const sigParts = [
        String(currentFrame),
        String(activeId ?? ""),
        String(viewBoostState.dimOutsidePoints ? 1 : 0),
        String(viewBoostState.colorSelectedOnly ? 1 : 0)
      ];

      for (let i = 0; i < count; i++) {
        const st = boxes[i];
        const pts = st.points;
        const x = Number(pts[0]) || 0;
        const y = Number(pts[1]) || 0;
        const z = Number(pts[2]) || 0;
        const rz = Number(pts[5]) || 0;
        const hx = Math.abs(Number(pts[6]) || 0) * 0.5;
        const hy = Math.abs(Number(pts[7]) || 0) * 0.5;
        const hz = Math.abs(Number(pts[8]) || 0) * 0.5;
        const [r, g, b] = parseColorToRgb01(st.label?.color);
        const isSel = activeId != null && st.clientID === activeId;
        const alpha = isSel ? 0.98 : 0.90;

        const base = i * 4;
        gpuBoxState.centerYaw[base] = x;
        gpuBoxState.centerYaw[base + 1] = y;
        gpuBoxState.centerYaw[base + 2] = z;
        gpuBoxState.centerYaw[base + 3] = rz;

        gpuBoxState.halfSize[base] = hx;
        gpuBoxState.halfSize[base + 1] = hy;
        gpuBoxState.halfSize[base + 2] = hz;
        gpuBoxState.halfSize[base + 3] = isSel ? 1.0 : 0.0;

        gpuBoxState.color[base] = r;
        gpuBoxState.color[base + 1] = g;
        gpuBoxState.color[base + 2] = b;
        gpuBoxState.color[base + 3] = alpha;

        sigParts.push(
          `${x.toFixed(2)},${y.toFixed(2)},${z.toFixed(2)},${rz.toFixed(2)},${hx.toFixed(2)},${hy.toFixed(2)},${hz.toFixed(2)},${st.label?.color || ""}`
        );
      }

      gpuBoxState.count = count;
      const sig = sigParts.join("|");
      if (forceRedraw || sig !== gpuBoxState.lastSig) {
        gpuBoxState.lastSig = sig;
        nudge3dCanvasRedraw();
      }
    } finally {
      gpuBoxState.refreshPending = false;
    }
  }

  async function syncGpuPointColorLoop(forceRedraw = false) {
    if (!viewBoostState.colorPointsByCuboid) {
      if (gpuBoxState.unsubscribe) {
        try {
          gpuBoxState.unsubscribe();
        } catch {}
        gpuBoxState.unsubscribe = null;
      }
      if (gpuBoxState.pollTimer) {
        clearInterval(gpuBoxState.pollTimer);
        gpuBoxState.pollTimer = null;
      }
      gpuBoxState.count = 0;
      gpuBoxState.lastSig = "";
      if (forceRedraw) nudge3dCanvasRedraw();
      return;
    }

    if (!gpuBoxState.unsubscribe) {
      const { store } = ctx();
      if (store?.subscribe) {
        gpuBoxState.unsubscribe = store.subscribe(() => {
          refreshGpuCuboidBoxes(false);
        });
      }
    }
    if (!gpuBoxState.pollTimer) {
      gpuBoxState.pollTimer = setInterval(() => {
        refreshGpuCuboidBoxes(false);
      }, 220);
    }
    await refreshGpuCuboidBoxes(forceRedraw);
  }

  // Tự động kích hoạt lại vòng cập nhật tô màu điểm nếu trước đó người dùng đang bật
  if (viewBoostState.colorPointsByCuboid) {
    setTimeout(() => syncGpuPointColorLoop(true), 1200);
  }

  // ============================================================================
  // EGO DISTANCE OVERLAY (Nhãn khoảng cách m ngay trên Perspective & Top view + HUD)
  // ============================================================================
  function clearEgoDistanceOverlays() {
    document.querySelectorAll("canvas.fusion3d-dist-overlay, #fusion3d-ego-dist-hud").forEach((el) => el.remove());
  }

  function updateEgoDistanceHud() {
    const persp = document.querySelector(".cvat-canvas3d-perspective") || document.querySelector(".cvat-canvas-container");
    if (!persp || !viewBoostState.showEgoDistance) {
      const old = document.getElementById("fusion3d-ego-dist-hud");
      if (old) old.remove();
      return;
    }
    if (getComputedStyle(persp).position === "static") {
      persp.style.position = "relative";
    }
    let hud = document.getElementById("fusion3d-ego-dist-hud");
    if (!hud) {
      hud = document.createElement("div");
      hud.id = "fusion3d-ego-dist-hud";
      hud.style.cssText = [
        "position:absolute",
        "top:8px",
        "right:10px",
        "z-index:30",
        "pointer-events:none",
        "background:rgba(15,23,42,0.84)",
        "border:1px solid rgba(56,189,248,0.55)",
        "border-radius:6px",
        "padding:5px 9px",
        "color:#f8fafc",
        "font:600 12px/1.35 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif",
        "box-shadow:0 2px 8px rgba(0,0,0,0.35)"
      ].join(";");
      persp.appendChild(hud);
    }

    const boxes = egoDistState.boxes;
    const sel = egoDistState.activeId != null ? boxes.find((b) => b.clientID === egoDistState.activeId) : null;
    if (sel) {
      hud.innerHTML =
        `<div style="color:#fde047">📍 #${sel.idLabel} ${sel.className}: <span style="font-size:14px">${sel.dist.toFixed(1)} m</span></div>` +
        `<div style="font-size:10.5px;color:#94a3b8;font-weight:500">Ego (x=${sel.x.toFixed(1)}, y=${sel.y.toFixed(1)}) · 3D: ${sel.dist3d.toFixed(1)} m</div>`;
    } else if (boxes.length > 0) {
      const minD = Math.min(...boxes.map((b) => b.dist));
      const maxD = Math.max(...boxes.map((b) => b.dist));
      hud.innerHTML =
        `<div>📍 Ego Distance · ${boxes.length} box</div>` +
        `<div style="font-size:10.5px;color:#94a3b8;font-weight:500">Gần nhất: ${minD.toFixed(1)} m · Xa nhất: ${maxD.toFixed(1)} m</div>`;
    } else {
      hud.innerHTML = `<div style="color:#94a3b8">📍 Ego Distance · 0 box trên frame</div>`;
    }
  }

  function drawEgoDistanceOverlaysNow() {
    if (!viewBoostState.showEgoDistance) return;
    updateEgoDistanceHud();

    const boxes = egoDistState.boxes;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    for (const [glCanvas, mats] of egoDistState.viewports.entries()) {
      if (!document.contains(glCanvas)) {
        egoDistState.viewports.delete(glCanvas);
        continue;
      }
      // Chỉ vẽ nhãn khoảng cách trên Perspective và Top View (tránh rối Side/Front)
      const isPersp = Boolean(glCanvas.closest(".cvat-canvas3d-perspective"));
      const isTop = Boolean(glCanvas.closest(".cvat-canvas3d-topview"));
      if (!isPersp && !isTop) continue;

      const parent = glCanvas.parentElement;
      if (!parent) continue;
      if (getComputedStyle(parent).position === "static") {
        parent.style.position = "relative";
      }

      let cvs = parent.querySelector("canvas.fusion3d-dist-overlay");
      if (!cvs) {
        cvs = document.createElement("canvas");
        cvs.className = "fusion3d-dist-overlay";
        cvs.style.position = "absolute";
        cvs.style.pointerEvents = "none";
        cvs.style.zIndex = "26";
        parent.appendChild(cvs);
      }

      const w = glCanvas.clientWidth;
      const h = glCanvas.clientHeight;
      if (w <= 0 || h <= 0) continue;
      cvs.style.left = `${glCanvas.offsetLeft}px`;
      cvs.style.top = `${glCanvas.offsetTop}px`;
      cvs.style.width = `${w}px`;
      cvs.style.height = `${h}px`;

      const tw = Math.round(w * dpr);
      const th = Math.round(h * dpr);
      if (cvs.width !== tw || cvs.height !== th) {
        cvs.width = tw;
        cvs.height = th;
      }

      const g = cvs.getContext("2d");
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);

      for (const b of boxes) {
        const anchorZ = isPersp ? b.z + b.dz * 0.5 + 0.25 : b.z + b.dz * 0.5;
        const eye = mulMat4Vec4(mats.viewMat, b.x, b.y, anchorZ, 1.0);
        const clip = mulMat4Vec4(mats.projMat, eye[0], eye[1], eye[2], eye[3]);
        if (clip[3] <= 0.001) continue;
        const ndcX = clip[0] / clip[3];
        const ndcY = clip[1] / clip[3];
        const ndcZ = clip[2] / clip[3];
        if (ndcX < -1.02 || ndcX > 1.02 || ndcY < -1.02 || ndcY > 1.02 || ndcZ < -1.0 || ndcZ > 1.0) {
          continue;
        }

        const px = (ndcX * 0.5 + 0.5) * w;
        const py = (0.5 - ndcY * 0.5) * h;
        const isSel = egoDistState.activeId != null && b.clientID === egoDistState.activeId;
        const text = isSel ? `#${b.idLabel} ${b.dist.toFixed(1)}m` : `${b.dist.toFixed(1)}m`;

        g.font = isSel ? "bold 11.5px sans-serif" : "600 10.5px sans-serif";
        const textW = g.measureText(text).width;
        const padX = 5;
        const boxW = textW + padX * 2;
        const boxH = isSel ? 17 : 15;
        const rx = Math.max(2, Math.min(w - boxW - 2, px - boxW * 0.5));
        const ry = Math.max(2, Math.min(h - boxH - 2, py - boxH - 2));

        g.fillStyle = isSel ? "rgba(234,179,8,0.94)" : "rgba(15,23,42,0.82)";
        g.fillRect(rx, ry, boxW, boxH);
        g.strokeStyle = isSel ? "#ffffff" : b.color || "#38bdf8";
        g.lineWidth = isSel ? 1.5 : 1.0;
        g.strokeRect(rx, ry, boxW, boxH);

        g.fillStyle = isSel ? "#0f172a" : "#f8fafc";
        g.fillText(text, rx + padX, ry + boxH - 4);
      }
    }
  }

  function scheduleEgoDistDraw() {
    if (!viewBoostState.showEgoDistance || egoDistState.rafId) return;
    egoDistState.rafId = requestAnimationFrame(() => {
      egoDistState.rafId = null;
      drawEgoDistanceOverlaysNow();
    });
  }

  async function refreshEgoDistanceData() {
    if (!viewBoostState.showEgoDistance) return;
    const { state, job } = ctx();
    if (!state || !job) return;
    const currentFrame = state.annotation?.player?.frame?.number ?? job.startFrame ?? 0;
    egoDistState.activeId = state.annotation?.annotations?.activatedStateID ?? null;
    try {
      const states = await job.annotations.get(currentFrame, true);
      egoDistState.boxes = states
        .filter((st) => !st.outside && Array.isArray(st.points) && st.points.length >= 9)
        .map((st) => {
          const [x, y, z, , , , , , dz] = st.points;
          return {
            clientID: st.clientID,
            idLabel: st.serverID ?? st.clientID,
            className: st.label?.name || "",
            color: st.label?.color || "#38bdf8",
            x: Number(x) || 0,
            y: Number(y) || 0,
            z: Number(z) || 0,
            dz: Math.abs(Number(dz) || 1.5),
            dist: Math.hypot(Number(x) || 0, Number(y) || 0),
            dist3d: Math.hypot(Number(x) || 0, Number(y) || 0, Number(z) || 0)
          };
        });
    } catch {}
    drawEgoDistanceOverlaysNow();
  }

  async function syncEgoDistanceLoop() {
    if (!viewBoostState.showEgoDistance) {
      if (egoDistState.unsubscribe) {
        try {
          egoDistState.unsubscribe();
        } catch {}
        egoDistState.unsubscribe = null;
      }
      if (egoDistState.pollTimer) {
        clearInterval(egoDistState.pollTimer);
        egoDistState.pollTimer = null;
      }
      if (egoDistState.rafId) {
        cancelAnimationFrame(egoDistState.rafId);
        egoDistState.rafId = null;
      }
      egoDistState.boxes = [];
      clearEgoDistanceOverlays();
      return;
    }

    if (!egoDistState.unsubscribe) {
      const { store } = ctx();
      if (store?.subscribe) {
        egoDistState.unsubscribe = store.subscribe(() => {
          refreshEgoDistanceData();
        });
      }
    }
    if (!egoDistState.pollTimer) {
      egoDistState.pollTimer = setInterval(() => {
        refreshEgoDistanceData();
      }, 250);
    }
    await refreshEgoDistanceData();
    nudge3dCanvasRedraw();
  }

  if (viewBoostState.showEgoDistance) {
    setTimeout(() => syncEgoDistanceLoop(), 1200);
  }

  function stopCamOverlayLoop() {
    if (camOverlayState.unsubscribe) {
      try {
        camOverlayState.unsubscribe();
      } catch {}
      camOverlayState.unsubscribe = null;
    }
    if (camOverlayState.pollTimer) {
      clearInterval(camOverlayState.pollTimer);
      camOverlayState.pollTimer = null;
    }
    if (camOverlayState.rafId) {
      cancelAnimationFrame(camOverlayState.rafId);
      camOverlayState.rafId = null;
    }
    clearCamOverlays();
  }

  async function setCamOverlay(payload) {
    if (payload.calibs && typeof payload.calibs === "object") {
      camOverlayState.calibsRaw = payload.calibs;
      camOverlayState.calibsCompiled = {};
      for (const [k, raw] of Object.entries(payload.calibs)) {
        camOverlayState.calibsCompiled[k] = compileCalibEntry(raw);
      }
    }
    if (payload.frameNames && typeof payload.frameNames === "object") {
      camOverlayState.frameNames = payload.frameNames;
    }
    if (typeof payload.showLabels === "boolean") camOverlayState.showLabels = payload.showLabels;
    if (typeof payload.selectedOnly === "boolean") camOverlayState.selectedOnly = payload.selectedOnly;

    camOverlayState.enabled = Boolean(payload.enabled);
    if (!camOverlayState.enabled) {
      stopCamOverlayLoop();
      return { ok: true, enabled: false };
    }

    if (!camOverlayState.unsubscribe) {
      const { store } = ctx();
      if (store?.subscribe) {
        camOverlayState.unsubscribe = store.subscribe(scheduleCamOverlayRender);
      }
    }
    if (!camOverlayState.pollTimer) {
      camOverlayState.pollTimer = setInterval(scheduleCamOverlayRender, 300);
    }
    await renderCamOverlaysNow();
    return {
      ok: true,
      enabled: true,
      calibKeys: Object.keys(camOverlayState.calibsCompiled),
      ...camOverlayState.lastStatus
    };
  }

  // Phím tắt [ và ]: chuyển qua lại giữa các Camera 2D trong cùng 1 frame
  function is3dViewItem(el) {
    return Boolean(
      el.querySelector(
        ".cvat-canvas3d-perspective, .cvat-canvas3d-topview, .cvat-canvas3d-sideview, .cvat-canvas3d-frontview"
      )
    );
  }

  function findContextCamContainers() {
    const containers = [];
    const seen = new Set();
    const add = (el) => {
      if (!el || seen.has(el)) return;
      if (is3dViewItem(el)) return;
      if (el.closest("header, .cvat-header, .ant-layout-header, .cvat-objects-sidebar, .cvat-player-controls")) {
        return;
      }
      seen.add(el);
      containers.push(el);
    };

    document.querySelectorAll(".cvat-context-image-wrapper, [class*='context-image']").forEach(add);
    document.querySelectorAll(".react-grid-item").forEach((item) => {
      if (item.querySelector("img, canvas:not(.fusion3d-cam-overlay)")) add(item);
    });
    return containers;
  }

  function getCamPanelName(panel) {
    const header = panel.querySelector("[class*='header'], .ant-card-head") || panel;
    const txt = (header.textContent || panel.textContent || "").trim();
    const m = txt.match(/\b(cam[_\-]?\d+|cam[_\-]?[a-z0-9]+)\b/i);
    if (m) return m[1].toLowerCase();
    return txt.slice(0, 32).toLowerCase();
  }

  function findPanelToggleExpandBtn(panel) {
    // Tìm nút expand/collapse của CVAT context image panel (thường ở góc trên bên trái)
    const explicit = panel.querySelector(
      '[aria-label="fullscreen"], [aria-label="fullscreen-exit"], [data-icon="fullscreen"], [data-icon="fullscreen-exit"], [data-icon="arrows-alt"], [data-icon="shrink"], .anticon-fullscreen, .anticon-fullscreen-exit, .anticon-arrows-alt, .anticon-shrink'
    );
    if (explicit) {
      const b = explicit.closest("button, [role='button'], span.anticon") || explicit;
      const r = b.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) return b;
    }

    // Fallback: nút/icon clickable nằm ở góc trên bên trái của panel (trừ nút settings/close bên phải)
    const panelRect = panel.getBoundingClientRect();
    const candidates = Array.from(panel.querySelectorAll("button, [role='button'], span.anticon, svg"));
    for (const el of candidates) {
      const r = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      if (r.top - panelRect.top <= 42 && r.left - panelRect.left <= 60) {
        return el.closest("button, [role='button']") || el;
      }
    }
    return null;
  }

  function listCamGridPanels() {
    const items = Array.from(document.querySelectorAll(".react-grid-item")).filter((item) => {
      if (is3dViewItem(item)) return false;
      const r = item.getBoundingClientRect();
      if (r.width < 50 || r.height < 40) return false;
      const hasMedia = Boolean(item.querySelector("img, canvas:not(.fusion3d-cam-overlay)"));
      const txt = (item.textContent || "").trim();
      if (/^(top|side|front|perspective)$/i.test(txt)) return false;
      return hasMedia || /\bcam/i.test(txt);
    });
    items.sort((a, b) =>
      getCamPanelName(a).localeCompare(getCamPanelName(b), undefined, { numeric: true })
    );
    return items;
  }

  async function switchContextCamera(direction) {
    const isNext = direction > 0;
    const containers = findContextCamContainers();

    // Cách 1: Nếu CVAT hiển thị nút mũi tên chuyển cam (< / >) hoặc <select>
    const iconSelectors = isNext
      ? [
          '[aria-label="right"]',
          '[data-icon="right"]',
          '[data-icon="caret-right"]',
          '[data-icon="step-forward"]',
          ".anticon-right",
          ".anticon-caret-right"
        ]
      : [
          '[aria-label="left"]',
          '[data-icon="left"]',
          '[data-icon="caret-left"]',
          '[data-icon="step-backward"]',
          ".anticon-left",
          ".anticon-caret-left"
        ];

    for (const box of containers) {
      for (const sel of iconSelectors) {
        const icon = box.querySelector(sel);
        const btn = icon?.closest("button, [role='button'], a, span.anticon") || icon;
        if (btn && !btn.disabled && btn.getAttribute("aria-disabled") !== "true") {
          const r = btn.getBoundingClientRect();
          if (r.width > 0 && r.height > 0) {
            btn.click();
            return true;
          }
        }
      }

      const buttons = Array.from(box.querySelectorAll("button, [role='button']"));
      for (const btn of buttons) {
        if (btn.disabled || btn.getAttribute("aria-disabled") === "true") continue;
        const r = btn.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) continue;
        const hint = `${btn.getAttribute("title") || ""} ${btn.getAttribute("aria-label") || ""} ${btn.textContent || ""}`
          .trim()
          .toLowerCase();
        const matched = isNext
          ? /\b(next|right)\b|^[>›→]$/.test(hint)
          : /\b(prev|previous|left)\b|^[<‹←]$/.test(hint);
        if (matched) {
          btn.click();
          return true;
        }
      }

      const sel = box.querySelector("select");
      if (sel && sel.options.length > 1) {
        const len = sel.options.length;
        sel.selectedIndex = (sel.selectedIndex + (isNext ? 1 : -1) + len) % len;
        sel.dispatchEvent(new Event("change", { bubbles: true }));
        return true;
      }
    }

    // Cách 2: CVAT chia mỗi cam thành 1 .react-grid-item có nút phóng to/thu nhỏ ở góc trên trái
    const panels = listCamGridPanels();
    if (!panels.length) return false;

    const isPanelMaximized = (p) => {
      const r = p.getBoundingClientRect();
      return r.width > window.innerWidth * 0.4 && r.height > window.innerHeight * 0.4;
    };

    const maximized = panels.find(isPanelMaximized);

    if (maximized) {
      const currentName = getCamPanelName(maximized);
      const collapseBtn = findPanelToggleExpandBtn(maximized);
      if (!collapseBtn) return false;
      collapseBtn.click();

      // Đợi React thu nhỏ panel hiện tại và render lại các panel cam khác
      let allPanels = [];
      for (let i = 0; i < 12; i++) {
        await new Promise((r) => setTimeout(r, 25));
        allPanels = listCamGridPanels();
        const stillMax = allPanels.find((p) => getCamPanelName(p) === currentName && isPanelMaximized(p));
        if (!stillMax && allPanels.length > 1) break;
      }
      if (allPanels.length <= 1) return true;

      const idx = allPanels.findIndex((p) => getCamPanelName(p) === currentName);
      const baseIdx = idx >= 0 ? idx : 0;
      const nextIdx = (baseIdx + (isNext ? 1 : -1) + allPanels.length) % allPanels.length;
      const expandBtn = findPanelToggleExpandBtn(allPanels[nextIdx]);
      if (expandBtn) expandBtn.click();
      return true;
    }

    // Chưa phóng to cam nào -> phóng to cam đầu (hoặc cuối nếu bấm [)
    const targetPanel = isNext ? panels[0] : panels[panels.length - 1];
    const expandBtn = findPanelToggleExpandBtn(targetPanel);
    if (expandBtn) {
      expandBtn.click();
      return true;
    }
    return false;
  }

  let camSwitchBusy = false;
  window.addEventListener(
    "keydown",
    async (e) => {
      if (!viewBoostState.camArrowShortcutEnabled) return;
      if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
      const isPrev = e.code === "BracketLeft" || e.key === "[";
      const isNext = e.code === "BracketRight" || e.key === "]";
      if (!isPrev && !isNext) return;
      const t = e.target;
      const tag = (t?.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select" || t?.isContentEditable) return;

      e.preventDefault();
      e.stopImmediatePropagation();
      if (camSwitchBusy) return;
      camSwitchBusy = true;
      try {
        await switchContextCamera(isNext ? 1 : -1);
      } finally {
        camSwitchBusy = false;
      }
    },
    true
  );

  // Phím tắt nhanh trực tiếp trên trang CVAT 3D: Alt+F = Flip Yaw 180° cho box đang chọn
  window.addEventListener("keydown", async (e) => {
    if (!e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    if (e.key.toLowerCase() !== "f") return;
    const tag = (e.target?.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return;
    const { state, job } = ctx();
    const activeId = state?.annotation?.annotations?.activatedStateID ?? null;
    if (!job || activeId == null) return;
    e.preventDefault();
    e.stopPropagation();
    try {
      const f = state.annotation?.player?.frame?.number ?? job.startFrame ?? 0;
      await applySafeEdit({ kind: "FLIP_YAW_180", clientId: activeId, frame: f });
    } catch (err) {
      console.warn("[Fusion3D Alt+F]", err);
    }
  });

  window.addEventListener("message", async (ev) => {
    const m = ev.data;
    if (
      ev.source !== window ||
      !m ||
      !m.__fusion3d ||
      !nonce ||
      m.nonce !== nonce ||
      m.source !== "content"
    ) {
      return;
    }

    try {
      let result;
      if (m.action === "GET_CONTEXT") {
        const { state, job } = ctx();
        const org =
          state?.organizations?.current?.slug ||
          state?.organizations?.current?.name ||
          state?.auth?.user?.organization ||
          new URLSearchParams(location.search).get("org") ||
          "";
        const currentFrame = state?.annotation?.player?.frame?.number ?? job?.startFrame ?? 0;
        const activeId = state?.annotation?.annotations?.activatedStateID ?? null;
        let activeObject = null;

        if (job && activeId != null) {
          try {
            const states = await job.annotations.get(currentFrame, true);
            const found = states.find((s) => s.clientID === activeId);
            if (found) activeObject = serializeObjectState(found);
          } catch {}
        }

        const labelObjects = (job?.labels || []).map((l) => ({
          id: l.id,
          name: l.name,
          type: l.type,
          color: l.color
        }));

        result = {
          connected: Boolean(job),
          url: location.href,
          origin: location.origin,
          jobId: job?.id ?? parseJobIdFromUrl(),
          taskId: job?.taskId ?? parseTaskIdFromUrl(),
          startFrame: job?.startFrame ?? 0,
          stopFrame: job?.stopFrame ?? 0,
          currentFrame,
          dimension: job?.dimension || "unknown",
          labels: labelObjects.map((l) => l.name),
          labelObjects,
          activeObject,
          org
        };
      } else if (m.action === "JUMP_FRAME") {
        result = jumpToFrame(m.payload.frame);
      } else if (m.action === "SELECT_OBJECT") {
        result = await selectObjectById(m.payload.serverId, m.payload.clientId, m.payload.frame);
      } else if (m.action === "INJECT_TRACKS") {
        result = await injectTracks(m.payload);
      } else if (m.action === "APPLY_V2_SAFE_EDIT") {
        result = await applySafeEdit(m.payload);
      } else if (m.action === "RUN_DAY0_PROBE") {
        result = await runDay0Probe();
      } else if (m.action === "SET_CAM_OVERLAY") {
        result = await setCamOverlay(m.payload || {});
      } else if (m.action === "SET_VIEW_BOOST") {
        const p = m.payload || {};
        if (p.pointMul !== undefined) {
          const pm = Number(p.pointMul);
          if (Number.isFinite(pm)) viewBoostState.pointMul = Math.min(10, Math.max(0.5, pm));
        }
        if (p.brightness !== undefined) {
          const br = Number(p.brightness);
          if (Number.isFinite(br)) viewBoostState.brightness = Math.min(300, Math.max(30, br));
        }
        if (p.brightenCam2d !== undefined) {
          viewBoostState.brightenCam2d = Boolean(p.brightenCam2d);
        }
        if (p.camArrowShortcutEnabled !== undefined) {
          viewBoostState.camArrowShortcutEnabled = Boolean(p.camArrowShortcutEnabled);
        }
        if (p.colorPointsByCuboid !== undefined) {
          viewBoostState.colorPointsByCuboid = Boolean(p.colorPointsByCuboid);
        }
        if (p.dimOutsidePoints !== undefined) {
          viewBoostState.dimOutsidePoints = Boolean(p.dimOutsidePoints);
        }
        if (p.colorSelectedOnly !== undefined) {
          viewBoostState.colorSelectedOnly = Boolean(p.colorSelectedOnly);
        }
        if (p.showEgoDistance !== undefined) {
          viewBoostState.showEgoDistance = Boolean(p.showEgoDistance);
        }
        saveViewBoostToLocalStorage();
        applyViewBrightnessCss();
        await syncGpuPointColorLoop(true);
        await syncEgoDistanceLoop();
        if (camOverlayState.enabled) renderCamOverlaysNow();
        nudge3dCanvasRedraw();
        result = { ok: true, ...viewBoostState, coloredBoxCount: gpuBoxState.count };
      } else if (m.action === "GET_VIEW_BOOST") {
        result = { ok: true, ...viewBoostState, coloredBoxCount: gpuBoxState.count };
      } else {
        throw new Error(`Unknown PageBridge action: ${m.action}`);
      }
      reply({ reqId: m.reqId, ok: true, result });
    } catch (err) {
      reply({ reqId: m.reqId, ok: false, error: err?.message || String(err) });
    }
  });
})();
