// Portrait analysis (MediaPipe) + dot-grid rendering.
import {
  FilesetResolver,
  ImageSegmenter,
  FaceDetector,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/vision_bundle.mjs";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm";
const SEG_MODEL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite";
const FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite";

// Longest side of the working copy used for analysis and sampling.
const WORK_SIZE = 1024;

let modelsPromise = null;

export function loadModels() {
  if (!modelsPromise) modelsPromise = createModels();
  return modelsPromise;
}

async function createModels() {
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  const make = async (delegate) => {
    const segmenter = await ImageSegmenter.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: SEG_MODEL, delegate },
      runningMode: "IMAGE",
      outputCategoryMask: false,
      outputConfidenceMasks: true,
    });
    const detector = await FaceDetector.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: FACE_MODEL, delegate },
      runningMode: "IMAGE",
      minDetectionConfidence: 0.4,
    });
    return { segmenter, detector };
  };
  try {
    return await make("GPU");
  } catch (err) {
    console.warn("GPU delegate unavailable, falling back to CPU", err);
    return make("CPU");
  }
}

export function loadImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Could not read ${file.name}`));
    img.src = url;
  });
}

// Runs segmentation + face detection once per image. The result is cached by
// the caller so slider changes only re-run the cheap grid/render steps.
export async function analyze(img) {
  const { segmenter, detector } = await loadModels();

  const scale = Math.min(1, WORK_SIZE / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.round(img.naturalWidth * scale);
  const h = Math.round(img.naturalHeight * scale);
  const work = document.createElement("canvas");
  work.width = w;
  work.height = h;
  work.getContext("2d").drawImage(img, 0, 0, w, h);

  // Foreground confidence = 1 - background confidence (category 0).
  const seg = segmenter.segment(work);
  const bg = seg.confidenceMasks[0].getAsFloat32Array();
  const mw = seg.confidenceMasks[0].width;
  const mh = seg.confidenceMasks[0].height;
  const fg = new Float32Array(bg.length);
  for (let i = 0; i < bg.length; i++) fg[i] = 1 - bg[i];
  seg.close();

  // Mask as an alpha canvas at working resolution so it can be cropped and
  // resampled with the same drawImage call as the photo.
  const mask = document.createElement("canvas");
  mask.width = w;
  mask.height = h;
  const mctx = mask.getContext("2d");
  const small = document.createElement("canvas");
  small.width = mw;
  small.height = mh;
  const sctx = small.getContext("2d");
  const id = sctx.createImageData(mw, mh);
  for (let i = 0; i < fg.length; i++) {
    id.data[i * 4 + 3] = Math.round(fg[i] * 255);
  }
  sctx.putImageData(id, 0, 0);
  mctx.drawImage(small, 0, 0, w, h);
  const maskData = mctx.getImageData(0, 0, w, h).data;

  const faces = detector.detect(work).detections;
  let face = null;
  for (const d of faces) {
    const b = d.boundingBox;
    if (!face || b.width * b.height > face.w * face.h) {
      face = { x: b.originX, y: b.originY, w: b.width, h: b.height };
    }
  }

  return { work, mask, w, h, face, autoCrop: computeCrop(maskData, w, h, face) };
}

// Square crop in working-image pixels: head near the top, shoulders filling
// the bottom, matching the reference framing.
function computeCrop(maskData, w, h, face) {
  const isFg = (x, y) => maskData[(y * w + x) * 4 + 3] > 127;

  // Bounding box of the foreground.
  let top = h, left = w, right = 0, bottom = 0;
  for (let y = 0; y < h; y += 2) {
    for (let x = 0; x < w; x += 2) {
      if (isFg(x, y)) {
        if (y < top) top = y;
        if (y > bottom) bottom = y;
        if (x < left) left = x;
        if (x > right) right = x;
      }
    }
  }
  if (top >= bottom) return { x: 0, y: 0, size: Math.min(w, h) };

  if (!face) {
    const size = Math.min(right - left, bottom - top) * 1.1;
    const cx = (left + right) / 2;
    return { x: cx - size / 2, y: top - size * 0.06, size };
  }

  // Top of the head: first row above the face with foreground over the face column.
  const x0 = Math.max(0, Math.floor(face.x));
  const x1 = Math.min(w - 1, Math.ceil(face.x + face.w));
  let headTop = face.y - face.h * 0.6;
  for (let y = 0; y < face.y; y++) {
    let count = 0;
    for (let x = x0; x <= x1; x++) if (isFg(x, y)) count++;
    if (count > (x1 - x0) * 0.08) {
      headTop = y;
      break;
    }
  }
  const chin = face.y + face.h;
  const headH = chin - headTop;
  const size = headH * 1.85;
  const cx = face.x + face.w / 2;
  return { x: cx - size / 2, y: headTop - size * 0.07, size };
}

// 8x8 Bayer matrix, normalized to (0, 1).
const BAYER8 = (() => {
  const m = [[0]];
  let n = 1;
  let mat = m;
  while (n < 8) {
    const next = [];
    for (let y = 0; y < n * 2; y++) next.push(new Array(n * 2));
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const v = mat[y][x] * 4;
        next[y][x] = v;
        next[y][x + n] = v + 2;
        next[y + n][x] = v + 3;
        next[y + n][x + n] = v + 1;
      }
    }
    mat = next;
    n *= 2;
  }
  return mat.map((row) => row.map((v) => (v + 0.5) / 64));
})();

// Samples the cropped photo into a cols x rows grid of tone levels.
// Returns Uint8Array where 0 = empty and 1..levels-1 = increasing ink.
export function buildGrid(analysis, adjust, opts) {
  const { cols, levels, contrast, brightness, gamma, dither, fadeRows, minInk, maxInk, detail } = opts;
  const rows = Math.round(cols * opts.aspect);

  // Crop window: zoom pivots around the face area so the head stays put.
  const base = analysis.autoCrop;
  const sizeW = (base.size / opts.zoom) * adjust.scale;
  const sizeH = sizeW * opts.aspect;
  const pivotY = base.y + base.size * 0.35;
  const sx = base.x + base.size / 2 + adjust.dx * base.size - sizeW / 2;
  const sy = pivotY - sizeW * 0.35 + (opts.offsetY + adjust.dy) * base.size;

  const sample = (source) => {
    // Halve repeatedly before the final resize for a proper area average.
    let w = Math.max(cols, Math.round(sizeW));
    let h = Math.max(rows, Math.round(sizeH));
    let cur = document.createElement("canvas");
    cur.width = w;
    cur.height = h;
    let ctx = cur.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(source, sx, sy, sizeW, sizeH, 0, 0, w, h);
    while (w > cols * 2 || h > rows * 2 || w !== cols || h !== rows) {
      const nw = w > cols * 2 ? Math.round(w / 2) : cols;
      const nh = h > rows * 2 ? Math.round(h / 2) : rows;
      const next = document.createElement("canvas");
      next.width = nw;
      next.height = nh;
      ctx = next.getContext("2d");
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(cur, 0, 0, nw, nh);
      cur = next;
      w = nw;
      h = nh;
    }
    return ctx.getImageData(0, 0, cols, rows).data;
  };

  const px = sample(analysis.work);
  const mk = sample(analysis.mask);

  const n = cols * rows;
  const lum = new Float32Array(n);
  const inside = new Uint8Array(n);
  const fgLum = [];
  for (let i = 0; i < n; i++) {
    const a = mk[i * 4 + 3] / 255;
    inside[i] = a > 0.5 ? 1 : 0;
    lum[i] = (0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2]) / 255;
    if (inside[i]) fgLum.push(lum[i]);
  }

  // Auto-levels on the subject only so every portrait lands in a similar range.
  fgLum.sort((a, b) => a - b);
  const lo = fgLum.length ? fgLum[Math.floor(fgLum.length * 0.02)] : 0;
  const hi = fgLum.length ? fgLum[Math.floor(fgLum.length * 0.98)] : 1;
  const range = Math.max(0.05, hi - lo);
  for (let i = 0; i < n; i++) lum[i] = (lum[i] - lo) / range;

  // Local contrast: push each cell away from its neighbourhood mean so eyes,
  // brows and mouth survive the coarse grid. Only subject cells are averaged
  // to avoid a halo against the empty background.
  if (detail > 0) {
    const r = Math.max(1, Math.round(cols / 32));
    const out = new Float32Array(n);
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const i = y * cols + x;
        if (!inside[i]) continue;
        let sum = 0, cnt = 0;
        for (let yy = Math.max(0, y - r); yy <= Math.min(rows - 1, y + r); yy++) {
          for (let xx = Math.max(0, x - r); xx <= Math.min(cols - 1, x + r); xx++) {
            const j = yy * cols + xx;
            if (inside[j]) { sum += lum[j]; cnt++; }
          }
        }
        out[i] = lum[i] + detail * (lum[i] - sum / cnt);
      }
    }
    lum.set(out);
  }

  // The fade is measured from where the subject ends in each column, which is
  // usually the photo's bottom edge rather than the bottom of the canvas.
  const bottom = new Int32Array(cols).fill(-1);
  for (let x = 0; x < cols; x++) {
    for (let y = rows - 1; y >= 0; y--) {
      if (inside[y * cols + x]) { bottom[x] = y; break; }
    }
  }
  // Use the lowest common edge so the fade reads as one horizontal band.
  const edges = [...bottom].filter((b) => b >= 0).sort((a, b) => a - b);
  const floorRow = edges.length ? edges[Math.floor(edges.length * 0.75)] : rows - 1;

  const grid = new Uint8Array(n);
  const steps = levels - 1;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const i = y * cols + x;
      if (!inside[i]) continue;
      const fromBottom = Math.min(floorRow, bottom[x]) - y;
      if (fromBottom < 0) continue;
      const fade = fromBottom < fadeRows ? (fromBottom + 0.5) / (fadeRows + 0.5) : 1;
      const t = BAYER8[y % 8][x % 8];
      if (fade < 1 && t > fade) continue;

      let v = lum[i];
      v = (v - 0.5) * contrast + 0.5 + brightness;
      v = Math.min(1, Math.max(0, v));
      v = Math.pow(v, gamma);
      // Compress into the ink range so highlights keep a light dot and
      // dark clothing reads as dense grey rather than solid black.
      const ink = minInk + (1 - v) * (maxInk - minInk);

      const threshold = 0.5 + (t - 0.5) * dither;
      const q = Math.floor(ink * steps + threshold);
      grid[i] = Math.min(steps, Math.max(0, q));
    }
  }
  return { grid, cols, rows };
}

function hexToRgb(hex) {
  const v = parseInt(hex.replace("#", ""), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

function levelColors(levels, ink, paper) {
  const a = hexToRgb(ink);
  const b = hexToRgb(paper);
  const out = [null];
  for (let k = 1; k < levels; k++) {
    const t = k / (levels - 1);
    const c = a.map((v, j) => Math.round(b[j] + (v - b[j]) * t));
    out.push(`rgb(${c[0]},${c[1]},${c[2]})`);
  }
  return out;
}

export function renderCanvas(canvas, { grid, cols, rows }, opts, pitch) {
  canvas.width = Math.round(cols * pitch);
  canvas.height = Math.round(rows * pitch);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!opts.transparent) {
    ctx.fillStyle = opts.paper;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
  const colors = levelColors(opts.levels, opts.ink, opts.paper);
  const s = pitch * opts.dotSize;
  const off = (pitch - s) / 2;
  for (let k = 1; k < opts.levels; k++) {
    ctx.fillStyle = colors[k];
    ctx.beginPath();
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        if (grid[y * cols + x] !== k) continue;
        const px = x * pitch + off;
        const py = y * pitch + off;
        if (opts.shape === "circle") {
          ctx.moveTo(px + s, py + s / 2);
          ctx.arc(px + s / 2, py + s / 2, s / 2, 0, Math.PI * 2);
        } else {
          ctx.rect(px, py, s, s);
        }
      }
    }
    ctx.fill();
  }
}

export function toSVG({ grid, cols, rows }, opts) {
  const pitch = 10;
  const s = +(pitch * opts.dotSize).toFixed(3);
  const off = +((pitch - s) / 2).toFixed(3);
  const colors = levelColors(opts.levels, opts.ink, opts.paper);
  const W = cols * pitch;
  const H = rows * pitch;
  const parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`,
  ];
  if (!opts.transparent) parts.push(`<rect width="${W}" height="${H}" fill="${opts.paper}"/>`);
  for (let k = 1; k < opts.levels; k++) {
    parts.push(`<g fill="${colors[k]}">`);
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        if (grid[y * cols + x] !== k) continue;
        const px = x * pitch + off;
        const py = y * pitch + off;
        parts.push(
          opts.shape === "circle"
            ? `<circle cx="${px + s / 2}" cy="${py + s / 2}" r="${s / 2}"/>`
            : `<rect x="${px}" y="${py}" width="${s}" height="${s}"/>`
        );
      }
    }
    parts.push(`</g>`);
  }
  parts.push(`</svg>`);
  return parts.join("");
}
