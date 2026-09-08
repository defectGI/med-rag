const STAGE_LABELS = { parse: "Parse", chunk: "Chunk", facts: "Facts (specs.db)", vectorize: "Vectorize" };

async function getJSON(path) {
  const res = await fetch(path);
  return res.json();
}

function fmtDate(iso) {
  if (!iso) return "—";
  return iso.replace("T", " ").slice(0, 19);
}

const OUTCOME_LABELS = { full_success: "Tam başarılı", partial: "Kısmi", failed: "Başarısız" };
const OUTCOME_CLASS = { full_success: "fresh", partial: "stale", failed: "failed" };

// Landing page -- last night + four counters. The panel computes NOTHING here,
// it shows the ready-made numbers returned by `/api/dashboard` (that endpoint
// also only reads the nightly report, see dashboard.py).
async function loadDashboard() {
  const data = await getJSON("/api/dashboard");
  const badge = document.getElementById("outcome-badge");
  const dateEl = document.getElementById("outcome-date");
  if (!data.last_night) {
    badge.textContent = "Henüz gecelik koşu yok";
    badge.className = "outcome-badge notrun";
    dateEl.textContent = "";
  } else {
    const outcome = data.last_night.outcome;
    badge.textContent = OUTCOME_LABELS[outcome] || outcome || "Bilinmiyor";
    badge.className = "outcome-badge " + (OUTCOME_CLASS[outcome] || "notrun");
    dateEl.textContent = `Koşu: ${fmtDate(data.last_night.started_at)} (${data.last_night.run_id || ""})`;
  }
  const c = data.counters;
  document.getElementById("dashboard-counters").innerHTML = `
    <div class="counter"><div class="counter-value">${c.stale_source_changed}</div><div class="counter-label">Bayat — kaynak değişti</div></div>
    <div class="counter"><div class="counter-value">${c.stale_stage_changed}</div><div class="counter-label">Bayat — aşama değişti</div></div>
    <div class="counter"><div class="counter-value">${c.ownerless}</div><div class="counter-label">Sahipsiz</div></div>
    <div class="counter"><div class="counter-value">${c.missing}</div><div class="counter-label">Eksik (üretilmemiş)</div></div>
    <div class="counter"><div class="counter-value">${c.open_issues}</div><div class="counter-label">Açık issue</div></div>
  `;
}

function currentIssueFilters() {
  return {
    stage: document.getElementById("issue-filter-stage").value,
    severity: document.getElementById("issue-filter-severity").value,
    source_doc: document.getElementById("issue-filter-source-doc").value.trim(),
    reason: document.getElementById("issue-filter-reason").value,
  };
}

// List narrowed by stage/severity/source_doc/reason filters + a per-row
// "incelendi/çözüldü" mark (it calls the single write endpoint that requires a
// write token, and triggers nothing else).
async function loadIssues(filters) {
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(filters || {})) {
    if (v) params.set(k, v);
  }
  const q = params.toString() ? `?${params}` : "";
  const data = await getJSON(`/api/issues${q}`);
  const tbody = document.querySelector("#issues-table tbody");
  tbody.innerHTML = "";
  const issues = data.issues || [];
  for (const it of issues) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${it.id}</td><td>${it.stage}</td><td>${it.severity}</td>
      <td>${it.source_doc || ""}</td><td>${it.entity_ref || ""}</td>
      <td>${it.reason}</td>
      <td>${(it.detail || "").replace(/</g, "&lt;")}</td>
      <td>${fmtDate(it.created_at)}</td>
      <td>${it.resolved_at ? `çözüldü (${it.resolved_by || ""})` : "açık"}</td>
      <td>${it.resolved_at ? "" : `<button class="resolve-btn" data-id="${it.id}">incelendi/çözüldü</button>`}</td>
    `;
    tbody.appendChild(tr);
  }
  if (issues.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" style="color:var(--muted)">Kayıt yok.</td></tr>`;
  }
  tbody.querySelectorAll(".resolve-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const token = document.getElementById("write-token").value;
      const resolvedBy = window.prompt("Kim çözdü/inceledi? (resolved_by)");
      if (!resolvedBy) return;
      const res = await fetch(`/api/issues/${btn.dataset.id}/resolve`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Panel-Write-Token": token },
        body: JSON.stringify({ resolved_by: resolvedBy }),
      });
      if (res.ok) {
        loadIssues(currentIssueFilters());
      } else {
        const body = await res.json().catch(() => ({}));
        window.alert(`Hata: ${body.error || res.status}`);
      }
    });
  });
}

async function loadStages() {
  const data = await getJSON("/api/stages");
  document.getElementById("ingest-meta").textContent =
    data.generated_from_ingest_at
      ? `Son ingest: ${fmtDate(data.generated_from_ingest_at)}`
      : "Henüz ingest çalıştırılmamış (python ingest.py).";

  const container = document.getElementById("stage-cards");
  container.innerHTML = "";
  for (const s of data.stages) {
    const card = document.createElement("div");
    card.className = "card";
    let statusClass = "notrun";
    let statusText = "veri yok";
    if (s.total > 0) {
      if (s.stale > 0) { statusClass = "stale"; statusText = `${s.stale}/${s.total} bayat`; }
      else if (s.not_run === s.total) { statusClass = "notrun"; statusText = "hiçbiri çalışmadı"; }
      else { statusClass = "fresh"; statusText = `${s.total} güncel`; }
    }
    card.innerHTML = `
      <div class="stage-name">${STAGE_LABELS[s.stage] || s.stage}</div>
      <div class="stage-status"><span class="dot ${statusClass}"></span>${statusText}</div>
      <div class="stage-detail">son çalışma: ${fmtDate(s.last_ran_at)}</div>
    `;
    container.appendChild(card);
  }
}

async function loadConflicts(productCode) {
  const q = productCode ? `?product_code=${encodeURIComponent(productCode)}` : "";
  const data = await getJSON(`/api/conflicts${q}`);
  const tbody = document.querySelector("#conflicts-table tbody");
  tbody.innerHTML = "";
  for (const c of data.conflicts) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${c.product_code || ""}</td>
      <td>${c.block || ""}</td>
      <td>${c.key || ""}</td>
      <td>${(c.detail || "").replace(/</g, "&lt;")}</td>
      <td>${fmtDate(c.flagged_at)}</td>
    `;
    tbody.appendChild(tr);
  }
  if (data.conflicts.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="color:var(--muted)">Kayıt yok.</td></tr>`;
  }
}

function drawGraph(nodeType, nodeId, upstream, downstream) {
  const svg = document.getElementById("graph-svg");
  svg.innerHTML = "";
  const NS = "http://www.w3.org/2000/svg";

  function makeNode(x, y, label, isCenter) {
    const g = document.createElementNS(NS, "g");
    g.setAttribute("class", "graph-node" + (isCenter ? " center" : ""));
    const w = Math.min(220, Math.max(90, label.length * 6.5 + 20));
    const rect = document.createElementNS(NS, "rect");
    rect.setAttribute("x", x - w / 2);
    rect.setAttribute("y", y - 16);
    rect.setAttribute("width", w);
    rect.setAttribute("height", 32);
    rect.setAttribute("rx", 6);
    g.appendChild(rect);
    const text = document.createElementNS(NS, "text");
    text.setAttribute("x", x);
    text.setAttribute("y", y + 4);
    text.setAttribute("text-anchor", "middle");
    text.textContent = label.length > 30 ? label.slice(0, 28) + "…" : label;
    g.appendChild(text);
    svg.appendChild(g);
    return { x, y, w };
  }

  function makeEdge(x1, y1, x2, y2) {
    const line = document.createElementNS(NS, "line");
    line.setAttribute("class", "graph-edge");
    line.setAttribute("x1", x1); line.setAttribute("y1", y1);
    line.setAttribute("x2", x2); line.setAttribute("y2", y2);
    svg.appendChild(line);
  }

  const centerX = 450, centerY = 180;
  const center = makeNode(centerX, centerY, `${nodeType}:${nodeId}`, true);

  const upN = upstream.length || 1;
  upstream.forEach((e, i) => {
    const x = 100 + (800 / (upN + 1)) * (i + 1);
    const y = 50;
    const n = makeNode(x, y, `${e.src_type}:${e.src_id}`, false);
    makeEdge(n.x, n.y + 16, center.x, center.y - 16);
  });

  const downN = downstream.length || 1;
  downstream.forEach((e, i) => {
    const x = 100 + (800 / (downN + 1)) * (i + 1);
    const y = 310;
    const n = makeNode(x, y, `${e.dst_type}:${e.dst_id}`, false);
    makeEdge(center.x, center.y + 16, n.x, n.y - 16);
  });
}

async function loadGraph(nodeType, nodeId) {
  const data = await getJSON(`/api/graph?node_type=${encodeURIComponent(nodeType)}&node_id=${encodeURIComponent(nodeId)}`);
  if (data.error) {
    document.getElementById("graph-detail").textContent = data.error;
    return;
  }
  drawGraph(nodeType, nodeId, data.upstream, data.downstream);
  document.getElementById("graph-detail").textContent =
    `Yukarı (nereden türedi): ${data.upstream.length} kenar · Aşağı (neyi besledi): ${data.downstream.length} kenar. Bir node'a tıklamak istersen node_id kutusuna yazıp "Göster"e bas.`;
}

async function runCascade(stage, nodeId) {
  const data = await getJSON(`/api/cascade?stage=${encodeURIComponent(stage)}&node_id=${encodeURIComponent(nodeId)}`);
  const el = document.getElementById("cascade-result");
  if (data.error) { el.textContent = data.error; return; }
  if (data.would_go_stale.length === 0) {
    el.innerHTML = `<span class="ok">Bu doküman için "${stage}" sonrasında henüz çalışmış bir aşama yok — cascade etkisi yok.</span>`;
    return;
  }
  el.innerHTML = `<span class="warn">"${stage}" yeniden çalıştırılırsa, bu doküman için şu aşamalar bayatlayacak:</span><ul>` +
    data.would_go_stale.map(s => `<li>${STAGE_LABELS[s.stage] || s.stage} (şu an: ${s.current_status || "?"}, son çalışma ${fmtDate(s.current_ran_at)})</li>`).join("") +
    `</ul><p>${data.note}</p>`;
}

document.getElementById("conflict-filter-btn").addEventListener("click", () => {
  loadConflicts(document.getElementById("conflict-filter").value.trim());
});
document.getElementById("graph-lookup-btn").addEventListener("click", () => {
  const nodeType = document.getElementById("graph-node-type").value;
  const nodeId = document.getElementById("graph-node-id").value.trim();
  if (nodeId) loadGraph(nodeType, nodeId);
});
document.getElementById("cascade-btn").addEventListener("click", () => {
  const stage = document.getElementById("cascade-stage").value;
  const nodeId = document.getElementById("cascade-node-id").value.trim();
  if (nodeId) runCascade(stage, nodeId);
});

document.getElementById("issue-filter-btn").addEventListener("click", () => {
  loadIssues(currentIssueFilters());
});

loadDashboard();
loadIssues({});
loadStages();
loadConflicts("");
