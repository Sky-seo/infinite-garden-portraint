import { loadModels, loadImage, analyze, buildGrid, renderCanvas, toSVG } from "./halftone.js";

const $ = (id) => document.getElementById(id);
const gallery = $("gallery");
const statusEl = $("status");
const tpl = $("cardTpl");

const NUMERIC = ["zoom", "offsetY", "aspect", "cols", "dotSize", "levels", "contrast", "brightness", "gamma", "detail", "minInk", "maxInk", "dither", "fadeRows"];
const OTHER = ["shape", "ink", "paper", "transparent", "exportSize"];
const DEFAULTS = {};
for (const id of [...NUMERIC, ...OTHER]) {
  const el = $(id);
  DEFAULTS[id] = el.type === "checkbox" ? el.checked : el.value;
}

const PREVIEW_WIDTH = 720;
const items = [];
let queue = Promise.resolve();

function readOpts() {
  const o = {};
  for (const id of NUMERIC) o[id] = parseFloat($(id).value);
  o.shape = $("shape").value;
  o.ink = $("ink").value;
  o.paper = $("paper").value;
  o.transparent = $("transparent").checked;
  o.exportSize = parseInt($("exportSize").value, 10) || 1600;
  return o;
}

function syncOutputs() {
  for (const id of NUMERIC) {
    const out = $(id).parentElement.querySelector("output");
    if (out) out.textContent = $(id).value;
  }
}

let frame = 0;
function scheduleRender() {
  cancelAnimationFrame(frame);
  frame = requestAnimationFrame(() => {
    syncOutputs();
    const opts = readOpts();
    for (const item of items) drawItem(item, opts);
  });
}

function drawItem(item, opts = readOpts()) {
  if (!item.analysis) return;
  item.grid = buildGrid(item.analysis, item.adjust, opts);
  renderCanvas(item.canvas, item.grid, opts, PREVIEW_WIDTH / opts.cols);
}

function updateBulkButtons() {
  const ready = items.some((i) => i.analysis);
  $("allPng").disabled = !ready;
  $("allSvg").disabled = !ready;
  $("empty").hidden = items.length > 0;
}

function baseName(item) {
  return item.name.replace(/\.[^.]+$/, "") + "-dots";
}

function download(blob, filename) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function exportPNG(item) {
  const opts = readOpts();
  const grid = buildGrid(item.analysis, item.adjust, opts);
  const c = document.createElement("canvas");
  renderCanvas(c, grid, opts, opts.exportSize / opts.cols);
  return new Promise((resolve) =>
    c.toBlob((blob) => {
      download(blob, baseName(item) + ".png");
      resolve();
    }, "image/png")
  );
}

function exportSVG(item) {
  const opts = readOpts();
  const svg = toSVG(buildGrid(item.analysis, item.adjust, opts), opts);
  download(new Blob([svg], { type: "image/svg+xml" }), baseName(item) + ".svg");
}

function bindCard(item) {
  const card = item.card;
  const stage = card.querySelector(".stage");

  card.querySelector(".actions").addEventListener("click", (e) => {
    const act = e.target.dataset.act;
    if (!act) return;
    if (act === "remove") {
      items.splice(items.indexOf(item), 1);
      card.remove();
      updateBulkButtons();
      return;
    }
    if (!item.analysis) return;
    if (act === "png") exportPNG(item);
    if (act === "svg") exportSVG(item);
    if (act === "recenter") {
      item.adjust = { dx: 0, dy: 0, scale: 1 };
      drawItem(item);
    }
  });

  // Drag to pan, scroll to zoom, double-click to reset the crop.
  let drag = null;
  stage.addEventListener("pointerdown", (e) => {
    if (!item.analysis) return;
    stage.setPointerCapture(e.pointerId);
    drag = { x: e.clientX, y: e.clientY };
  });
  stage.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const opts = readOpts();
    const k = item.adjust.scale / opts.zoom / stage.clientWidth;
    item.adjust.dx -= (e.clientX - drag.x) * k;
    item.adjust.dy -= (e.clientY - drag.y) * k;
    drag = { x: e.clientX, y: e.clientY };
    drawItem(item, opts);
  });
  const endDrag = () => (drag = null);
  stage.addEventListener("pointerup", endDrag);
  stage.addEventListener("pointercancel", endDrag);
  stage.addEventListener(
    "wheel",
    (e) => {
      if (!item.analysis) return;
      e.preventDefault();
      const s = item.adjust.scale * Math.exp(e.deltaY * 0.0015);
      item.adjust.scale = Math.min(3, Math.max(0.3, s));
      drawItem(item);
    },
    { passive: false }
  );
  stage.addEventListener("dblclick", () => {
    if (!item.analysis) return;
    item.adjust = { dx: 0, dy: 0, scale: 1 };
    drawItem(item);
  });
}

function addFiles(files) {
  for (const file of files) {
    // Some files (e.g. HEIC from Finder) arrive with an empty MIME type.
    const isImage = file.type.startsWith("image/") || /\.(jpe?g|png|webp|gif|bmp|avif|heic|heif)$/i.test(file.name);
    if (!isImage) continue;
    const card = tpl.content.firstElementChild.cloneNode(true);
    const item = {
      name: file.name,
      card,
      canvas: card.querySelector("canvas"),
      analysis: null,
      grid: null,
      adjust: { dx: 0, dy: 0, scale: 1 },
    };
    card.querySelector(".name").textContent = file.name;
    card.classList.add("loading");
    gallery.appendChild(card);
    items.push(item);
    bindCard(item);
    updateBulkButtons();

    // Analyze one image at a time; the models are shared.
    queue = queue.then(async () => {
      try {
        const img = await loadImage(file);
        item.analysis = await analyze(img);
        URL.revokeObjectURL(img.src);
        card.classList.remove("loading");
        if (!item.analysis.face) card.title = "No face detected — framed from the silhouette.";
        drawItem(item);
      } catch (err) {
        console.error(err);
        card.classList.remove("loading");
        card.classList.add("failed");
        card.querySelector(".busy").textContent = /\.hei[cf]$/i.test(file.name)
          ? "HEIC isn't supported in this browser — export as JPG"
          : `Could not process image: ${err.message}`;
      }
      updateBulkButtons();
    });
  }
}

// Inputs
for (const id of [...NUMERIC, ...OTHER]) {
  $(id).addEventListener("input", scheduleRender);
}
$("file").addEventListener("change", (e) => {
  addFiles(e.target.files);
  e.target.value = "";
});
const drop = $("drop");
for (const ev of ["dragenter", "dragover"]) {
  document.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.add("over");
  });
}
for (const ev of ["dragleave", "drop"]) {
  document.addEventListener(ev, (e) => {
    e.preventDefault();
    if (ev === "drop" || e.relatedTarget === null) drop.classList.remove("over");
  });
}
document.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
document.addEventListener("paste", (e) => {
  const files = [...e.clipboardData.items].filter((i) => i.kind === "file").map((i) => i.getAsFile());
  addFiles(files);
});

$("allPng").addEventListener("click", async () => {
  for (const item of items) if (item.analysis) await exportPNG(item);
});
$("allSvg").addEventListener("click", () => {
  for (const item of items) if (item.analysis) exportSVG(item);
});
$("reset").addEventListener("click", () => {
  for (const [id, v] of Object.entries(DEFAULTS)) {
    const el = $(id);
    if (el.type === "checkbox") el.checked = v;
    else el.value = v;
  }
  scheduleRender();
});

syncOutputs();
loadModels().then(
  () => {
    statusEl.textContent = "Ready";
    statusEl.classList.add("ready");
  },
  (err) => {
    console.error(err);
    statusEl.textContent = "Failed to load models — check your connection";
    statusEl.classList.add("error");
  }
);

// Expose for debugging / automated checks.
window.__dotPortrait = { items, addFiles };
