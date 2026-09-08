"use strict";

const PRODUCT_NODES_URL = "../product_info/product_nodes.json";
const DOCUMENT_NODES_URL = "../document_info/document_nodes.json";

const BOX_W = 200;
const BOX_H = 44;
const COL_W = 260;
const ROW_H = 56;

const state = {
  nodesById: new Map(),
  childrenByParent: new Map(),
  docsByOwner: new Map(),
  childItemsCache: new Map(), // node_id -> item[] (children + own doc leaves)
  expanded: new Set(["root"]),
  selectedId: null,
  root: null,
  pan: { x: 40, y: 40 },
  scale: 1,
};

init();

async function init() {
  let productData, documentData;
  try {
    [productData, documentData] = await Promise.all([
      fetchJson(PRODUCT_NODES_URL),
      fetchJson(DOCUMENT_NODES_URL),
    ]);
  } catch (err) {
    showLoadError(err);
    return;
  }

  buildIndexes(productData, documentData);
  wirePanZoom();
  document.getElementById("reset-view").addEventListener("click", fitToView);
  render();
  fitToView();
}

function fetchJson(url) {
  return fetch(url).then((res) => {
    if (!res.ok) throw new Error(`${url} → HTTP ${res.status}`);
    return res.json();
  });
}

function showLoadError(err) {
  document.querySelector(".viewport").innerHTML = `
    <div class="load-error">
      <strong>Veri yüklenemedi.</strong>
      <p>Bu ekran <code>product_nodes.json</code> ve <code>document_nodes.json</code>
      dosyalarını <code>fetch</code> ile okur; tarayıcılar dosya sisteminden (file://)
      doğrudan fetch'e genelde izin vermez. <code>chatbot-corpus</code> klasöründe basit
      bir HTTP sunucusu başlatıp oradan açın:</p>
      <code>cd chatbot-corpus<br>python -m http.server 8000</code>
      <p>Sonra tarayıcıda: <code>http://localhost:8000/viz/</code></p>
      <p style="color:#a33">${escapeHtml(err.message || String(err))}</p>
    </div>`;
}

// ---------------------------------------------------------------------
// Data indexing
// ---------------------------------------------------------------------

function buildIndexes(productData, documentData) {
  const nodes = productData.product_nodes;
  for (const n of nodes) state.nodesById.set(n.node_id, n);

  for (const n of nodes) {
    const key = n.parent_id;
    if (!state.childrenByParent.has(key)) state.childrenByParent.set(key, []);
    state.childrenByParent.get(key).push(n);
  }
  for (const arr of state.childrenByParent.values()) arr.sort(bySourceRef);

  for (const doc of documentData.documents) {
    for (const ownerId of doc.links.owner_ids) {
      if (!state.docsByOwner.has(ownerId)) state.docsByOwner.set(ownerId, []);
      state.docsByOwner.get(ownerId).push(doc);
    }
  }

  const roots = state.childrenByParent.get(null) || [];
  state.root = {
    kind: "root",
    id: "root",
    node: null,
    childItems: roots.map((n) => ({ kind: "node", id: n.node_id, node: n })),
  };
}

function bySourceRef(a, b) {
  const ka = a.source_ref.split(".").map(Number);
  const kb = b.source_ref.split(".").map(Number);
  const len = Math.max(ka.length, kb.length);
  for (let i = 0; i < len; i++) {
    const x = ka[i] ?? -1;
    const y = kb[i] ?? -1;
    if (x !== y) return x - y;
  }
  return 0;
}

function nodeLabel(node) {
  if (node.type === "product") {
    return (node.product && (node.product.display_name || node.product.product_code)) || "(isimsiz ürün)";
  }
  return node[node.type] || node.source_ref;
}

function nodeCode(node) {
  if (node.product) return node.product.acme_code || node.product.product_code || null;
  if (node.category) return node.category.reference_code || null;
  return null;
}

function nodeSubtitle(node) {
  if (node.type === "product") {
    const code = nodeCode(node);
    if (node.product && node.product.price_on_request) return code ? `${code} · fiyat sorunuz` : "fiyat sorunuz";
    if (node.product && node.product.list_price != null) {
      const price = new Intl.NumberFormat("tr-TR", { maximumFractionDigits: 0 }).format(node.product.list_price);
      return code ? `${code} · ${price}` : price;
    }
    return code || "";
  }
  const kidCount = (state.childrenByParent.get(node.node_id) || []).length;
  return kidCount ? `${kidCount} öğe` : "";
}

// item children: subcategory/product nodes, then this node's own documents as leaves
function getChildItems(item) {
  if (item.kind === "root") return item.childItems;
  if (item.kind === "doc") return [];

  if (state.childItemsCache.has(item.id)) return state.childItemsCache.get(item.id);

  const subNodes = state.childrenByParent.get(item.id) || [];
  const docs = state.docsByOwner.get(item.id) || [];
  const items = [
    ...subNodes.map((n) => ({ kind: "node", id: n.node_id, node: n })),
    ...docs.map((d) => ({ kind: "doc", id: "doc:" + d.identity.doc_id, doc: d })),
  ];
  state.childItemsCache.set(item.id, items);
  return items;
}

// ---------------------------------------------------------------------
// Layout (left-to-right collapsible tree)
// ---------------------------------------------------------------------

function layout() {
  const positions = new Map(); // id -> {x, y, item, parentId}
  const edges = []; // {from, to}
  let cursorY = 0;

  function place(item, depth, parentId) {
    const expandable = item.kind !== "doc" && getChildItems(item).length > 0;
    const isOpen = expandable && state.expanded.has(item.id);
    const kids = isOpen ? getChildItems(item) : [];

    let y;
    if (kids.length === 0) {
      y = cursorY * ROW_H;
      cursorY++;
    } else {
      const childYs = kids.map((k) => place(k, depth + 1, item.id));
      y = (childYs[0] + childYs[childYs.length - 1]) / 2;
    }

    positions.set(item.id, { x: depth * COL_W, y, item, expandable, isOpen });
    if (parentId) edges.push({ from: parentId, to: item.id });
    return y;
  }

  place(state.root, 0, null);
  return { positions, edges };
}

// ---------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------

function render() {
  const { positions, edges } = layout();

  const nodesEl = document.getElementById("nodes");
  const linesEl = document.getElementById("lines");
  const worldEl = document.getElementById("world");

  let maxX = 0;
  let maxY = 0;
  for (const p of positions.values()) {
    maxX = Math.max(maxX, p.x + BOX_W);
    maxY = Math.max(maxY, p.y + BOX_H);
  }
  worldEl.style.width = maxX + 60 + "px";
  worldEl.style.height = maxY + 60 + "px";
  linesEl.setAttribute("width", maxX + 60);
  linesEl.setAttribute("height", maxY + 60);

  nodesEl.innerHTML = "";
  for (const p of positions.values()) {
    nodesEl.appendChild(buildBox(p));
  }

  linesEl.innerHTML = edges
    .map((e) => {
      const a = positions.get(e.from);
      const b = positions.get(e.to);
      if (!a || !b) return "";
      const x1 = a.x + BOX_W;
      const y1 = a.y + BOX_H / 2;
      const x2 = b.x;
      const y2 = b.y + BOX_H / 2;
      const midX = (x1 + x2) / 2;
      return `<path d="M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}" />`;
    })
    .join("");
}

function buildBox(p) {
  const item = p.item;
  const box = document.createElement("div");
  box.style.left = p.x + "px";
  box.style.top = p.y + "px";
  box.dataset.id = item.id;

  if (item.kind === "root") {
    box.className = "node-box root";
    box.innerHTML = `<div class="title">ACME</div><div class="sub">Ürün Aile Ağacı</div>`;
  } else if (item.kind === "doc") {
    const d = item.doc;
    const isImage = d.doc_type === "PRODUCT_IMAGE";
    box.className = "node-box doc" + (isImage ? " image" : "");
    box.title = d.location.file_path;

    if (isImage) {
      const img = document.createElement("img");
      img.className = "thumb";
      img.src = `/thumb?path=${encodeURIComponent(d.location.file_path)}`;
      img.alt = "";
      img.loading = "lazy";
      img.addEventListener("error", () => { img.style.visibility = "hidden"; });
      box.appendChild(img);
      const title = document.createElement("div");
      title.className = "title";
      title.textContent = d.identity.file_name;
      box.appendChild(title);
    } else {
      const ext = (d.identity.extension || "").replace(".", "").toUpperCase() || "?";
      const title = document.createElement("div");
      title.className = "title";
      const extSpan = document.createElement("span");
      extSpan.className = "ext";
      extSpan.textContent = ext;
      title.appendChild(extSpan);
      title.appendChild(document.createTextNode(d.identity.file_name));
      box.appendChild(title);
    }

    box.addEventListener("click", () => openOrCopy(d.location.file_path, d.identity.file_name));
  } else {
    const n = item.node;
    box.className = `node-box ${n.type}`;
    box.title = nodeLabel(n);
    const sub = nodeSubtitle(n);
    box.innerHTML = `<div class="title">${escapeHtml(nodeLabel(n))}</div>${sub ? `<div class="sub">${escapeHtml(sub)}</div>` : ""}`;
    box.addEventListener("click", () => toggle(item.id));
  }

  if (item.id === state.selectedId) box.classList.add("selected");

  if (p.expandable) {
    const dot = document.createElement("div");
    dot.className = "toggle-dot";
    dot.textContent = p.isOpen ? "−" : "+";
    box.appendChild(dot);
  }

  return box;
}

function toggle(id) {
  state.selectedId = id;
  if (state.expanded.has(id)) state.expanded.delete(id);
  else state.expanded.add(id);
  render();
}

// ---------------------------------------------------------------------
// Pan / zoom
// ---------------------------------------------------------------------

function applyTransform() {
  const world = document.getElementById("world");
  world.style.transform = `translate(${state.pan.x}px, ${state.pan.y}px) scale(${state.scale})`;
}

function wirePanZoom() {
  const viewport = document.getElementById("viewport");
  let dragging = false;
  let moved = false;
  let startX, startY, startPanX, startPanY;

  viewport.addEventListener("mousedown", (e) => {
    dragging = true;
    moved = false;
    viewport.classList.add("dragging");
    startX = e.clientX;
    startY = e.clientY;
    startPanX = state.pan.x;
    startPanY = state.pan.y;
  });

  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) moved = true;
    state.pan.x = startPanX + dx;
    state.pan.y = startPanY + dy;
    applyTransform();
  });

  window.addEventListener("mouseup", () => {
    dragging = false;
    viewport.classList.remove("dragging");
  });

  viewport.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const rect = viewport.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      const worldX = (cx - state.pan.x) / state.scale;
      const worldY = (cy - state.pan.y) / state.scale;
      const factor = e.deltaY < 0 ? 1.1 : 0.9;
      const newScale = Math.min(2, Math.max(0.25, state.scale * factor));
      state.pan.x = cx - worldX * newScale;
      state.pan.y = cy - worldY * newScale;
      state.scale = newScale;
      applyTransform();
    },
    { passive: false }
  );
}

function fitToView() {
  const viewport = document.getElementById("viewport");
  const world = document.getElementById("world");
  const vw = viewport.clientWidth;
  const vh = viewport.clientHeight;
  const ww = parseFloat(world.style.width) || vw;
  const wh = parseFloat(world.style.height) || vh;
  const scale = Math.min(1, (vw - 60) / ww, (vh - 60) / wh);
  state.scale = scale;
  state.pan.x = 30;
  state.pan.y = Math.max(30, (vh - wh * scale) / 2);
  applyTransform();
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

async function openOrCopy(filePath, fileName) {
  try {
    const res = await fetch(`/open?path=${encodeURIComponent(filePath)}`);
    if (res.ok) {
      showToast("Açılıyor: " + fileName);
      return;
    }
  } catch (err) {
    // server unreachable -- fall back to copying below
  }
  navigator.clipboard.writeText(filePath).then(
    () => showToast("Açılamadı (serve.py çalışmıyor olabilir) — yol kopyalandı"),
    () => showToast("Açılamadı")
  );
}

let toastHandle = null;
function showToast(msg) {
  const toast = document.getElementById("toast");
  toast.textContent = msg;
  toast.classList.add("show");
  clearTimeout(toastHandle);
  toastHandle = setTimeout(() => toast.classList.remove("show"), 2200);
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
