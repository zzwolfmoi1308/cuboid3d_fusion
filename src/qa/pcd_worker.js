/**
 * PCD Parser Web Worker (V1.5)
 * Giải mã file Point Cloud Data (.pcd) các định dạng:
 * - ascii
 * - binary
 * - binary_compressed (LZF decompression)
 * Chạy hoàn toàn trong Web Worker để không làm giật Canvas 3D.
 */

function lzfDecompress(src, outLen) {
  const dst = new Uint8Array(outLen);
  let iidx = 0;
  let oidx = 0;
  const inLen = src.length;

  while (iidx < inLen) {
    let ctrl = src[iidx++];
    if (ctrl < 32) {
      ctrl++;
      if (oidx + ctrl > outLen) throw new Error("LZF output overflow");
      for (let i = 0; i < ctrl; i++) {
        dst[oidx++] = src[iidx++];
      }
    } else {
      let len = ctrl >> 5;
      let ref = oidx - ((ctrl & 0x1f) << 8) - 1;
      if (len === 7) {
        len += src[iidx++];
      }
      ref -= src[iidx++];
      if (ref < 0) throw new Error("LZF invalid back-reference");
      len += 2;
      if (oidx + len > outLen) throw new Error("LZF output overflow");
      for (let i = 0; i < len; i++) {
        dst[oidx++] = dst[ref++];
      }
    }
  }
  return dst;
}

function parseHeader(bytes) {
  // Tìm dòng "DATA <format>\n"
  const decoder = new TextDecoder("ascii");
  const probeLen = Math.min(bytes.length, 4096);
  const headerText = decoder.decode(bytes.subarray(0, probeLen));
  const dataMatch = headerText.match(/^DATA\s+(\S+)\r?\n/m);
  if (!dataMatch) {
    throw new Error("Không tìm thấy trường DATA trong header PCD.");
  }

  const headerEndOffset = dataMatch.index + dataMatch[0].length;
  const lines = headerText.slice(0, headerEndOffset).split(/\r?\n/);

  const header = {
    fields: [],
    size: [],
    type: [],
    count: [],
    width: 0,
    height: 1,
    points: 0,
    data: dataMatch[1].toLowerCase(),
    headerEndOffset
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const parts = line.split(/\s+/);
    const key = parts[0].toUpperCase();
    const vals = parts.slice(1);

    if (key === "FIELDS") header.fields = vals.map((v) => v.toLowerCase());
    else if (key === "SIZE") header.size = vals.map((v) => parseInt(v, 10));
    else if (key === "TYPE") header.type = vals.map((v) => v.toUpperCase());
    else if (key === "COUNT") header.count = vals.map((v) => parseInt(v, 10));
    else if (key === "WIDTH") header.width = parseInt(vals[0], 10);
    else if (key === "HEIGHT") header.height = parseInt(vals[0], 10);
    else if (key === "POINTS") header.points = parseInt(vals[0], 10);
  }

  if (header.count.length === 0) {
    header.count = header.fields.map(() => 1);
  }
  if (!header.points) {
    header.points = header.width * header.height;
  }

  return header;
}

function readTypedValue(view, offset, type, size) {
  if (type === "F") {
    return size === 8 ? view.getFloat64(offset, true) : view.getFloat32(offset, true);
  }
  if (type === "I") {
    if (size === 1) return view.getInt8(offset);
    if (size === 2) return view.getInt16(offset, true);
    return view.getInt32(offset, true);
  }
  if (type === "U") {
    if (size === 1) return view.getUint8(offset);
    if (size === 2) return view.getUint16(offset, true);
    return view.getUint32(offset, true);
  }
  return view.getFloat32(offset, true);
}

export function parsePcdBuffer(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  const header = parseHeader(bytes);
  const xIdx = header.fields.indexOf("x");
  const yIdx = header.fields.indexOf("y");
  const zIdx = header.fields.indexOf("z");
  if (xIdx < 0 || yIdx < 0 || zIdx < 0) {
    throw new Error(`PCD thiếu trục x/y/z (FIELDS: ${header.fields.join(", ")})`);
  }

  const numPts = header.points;
  const xyz = new Float32Array(numPts * 3);
  let validCount = 0;

  if (header.data === "ascii") {
    const text = new TextDecoder("ascii").decode(bytes.subarray(header.headerEndOffset));
    const lines = text.split(/\r?\n/);
    for (let i = 0; i < lines.length && validCount < numPts; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      const cols = line.split(/\s+/);
      const x = parseFloat(cols[xIdx]);
      const y = parseFloat(cols[yIdx]);
      const z = parseFloat(cols[zIdx]);
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        xyz[validCount * 3] = x;
        xyz[validCount * 3 + 1] = y;
        xyz[validCount * 3 + 2] = z;
        validCount++;
      }
    }
  } else if (header.data === "binary") {
    const offsets = [];
    let stride = 0;
    for (let f = 0; f < header.fields.length; f++) {
      offsets.push(stride);
      stride += (header.size[f] || 4) * (header.count[f] || 1);
    }
    const view = new DataView(arrayBuffer, header.headerEndOffset);
    const maxPts = Math.min(numPts, Math.floor((bytes.length - header.headerEndOffset) / stride));
    for (let i = 0; i < maxPts; i++) {
      const base = i * stride;
      const x = readTypedValue(view, base + offsets[xIdx], header.type[xIdx], header.size[xIdx]);
      const y = readTypedValue(view, base + offsets[yIdx], header.type[yIdx], header.size[yIdx]);
      const z = readTypedValue(view, base + offsets[zIdx], header.type[zIdx], header.size[zIdx]);
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        xyz[validCount * 3] = x;
        xyz[validCount * 3 + 1] = y;
        xyz[validCount * 3 + 2] = z;
        validCount++;
      }
    }
  } else if (header.data === "binary_compressed") {
    const view = new DataView(arrayBuffer, header.headerEndOffset);
    const compressedSize = view.getUint32(0, true);
    const uncompressedSize = view.getUint32(4, true);
    const compData = bytes.subarray(
      header.headerEndOffset + 8,
      header.headerEndOffset + 8 + compressedSize
    );
    const raw = lzfDecompress(compData, uncompressedSize);
    const rawView = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);

    // Trong binary_compressed, mỗi field lưu liên tiếp toàn bộ N điểm (column-major)
    const fieldByteStart = [];
    let cursor = 0;
    for (let f = 0; f < header.fields.length; f++) {
      fieldByteStart.push(cursor);
      cursor += (header.size[f] || 4) * (header.count[f] || 1) * numPts;
    }

    const xSize = header.size[xIdx] || 4;
    const ySize = header.size[yIdx] || 4;
    const zSize = header.size[zIdx] || 4;

    for (let i = 0; i < numPts; i++) {
      const x = readTypedValue(rawView, fieldByteStart[xIdx] + i * xSize, header.type[xIdx], xSize);
      const y = readTypedValue(rawView, fieldByteStart[yIdx] + i * ySize, header.type[yIdx], ySize);
      const z = readTypedValue(rawView, fieldByteStart[zIdx] + i * zSize, header.type[zIdx], zSize);
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        xyz[validCount * 3] = x;
        xyz[validCount * 3 + 1] = y;
        xyz[validCount * 3 + 2] = z;
        validCount++;
      }
    }
  } else {
    throw new Error(`Định dạng DATA PCD chưa hỗ trợ: ${header.data}`);
  }

  return {
    format: header.data,
    fields: header.fields,
    declaredPoints: header.points,
    validPoints: validCount,
    xyz: xyz.subarray(0, validCount * 3)
  };
}

// Hỗ trợ chạy như Web Worker trong Extension
if (typeof self !== "undefined" && typeof self.postMessage === "function" && typeof window === "undefined") {
  self.onmessage = (e) => {
    const { id, buffer } = e.data || {};
    try {
      const parsed = parsePcdBuffer(buffer);
      // Transfer underlying ArrayBuffer của subarray nếu cần
      const copy = new Float32Array(parsed.xyz);
      self.postMessage(
        {
          id,
          ok: true,
          result: {
            format: parsed.format,
            fields: parsed.fields,
            declaredPoints: parsed.declaredPoints,
            validPoints: parsed.validPoints,
            xyz: copy
          }
        },
        [copy.buffer]
      );
    } catch (err) {
      self.postMessage({ id, ok: false, error: err?.message || String(err) });
    }
  };
}
