import {
  health,
  getKM,
  getAnomalies,
  getVistaSummary,
  getSegment,
  getMarkovMatrix,
  getMarkovAbsorption,
} from "./api.js";

const STATE_LABELS = [
  "Cancelled",
  "In Legal Processing",
  "Legal Agreement Effective",
  "Disbursed",
  "Closed",
];

// ---------------------
// Small utils
// ---------------------
function fmtNumber(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return new Intl.NumberFormat().format(x);
}

function fmtMoney(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  // Keep generic (no currency symbol) since GCF has mixed reporting; adjust if needed
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(x);
}

function fmtPct(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(2)}%`;
}

function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

// ---------------------
// KPI rendering
// ---------------------
function renderKpis(vistaSummary, absorption) {
  // VISTA summary
  const mean = vistaSummary?.mean;
  const min = vistaSummary?.min;
  const max = vistaSummary?.max;

  document.getElementById("kpiVistaMean").textContent =
    mean !== undefined ? mean.toFixed(3) : "—";
  document.getElementById("kpiVistaRange").textContent =
    (min !== undefined && max !== undefined)
      ? `range: ${min.toFixed(3)} → ${max.toFixed(3)}`
      : "range: —";

  // Markov absorption from transient state 1 (Legal Processing)
  // absorption.absorption_probabilities["1"] => {"0":..., "4":...}
  const probs = absorption?.absorption_probabilities?.["1"];
  const steps = absorption?.expected_steps_to_absorption?.["1"];

  const pCancel = probs ? probs["0"] : undefined;
  const pClosed = probs ? probs["4"] : undefined;

  document.getElementById("kpiMarkovClosedFromLegal").textContent =
    pClosed !== undefined ? fmtPct(pClosed) : "—";
  document.getElementById("kpiMarkovCancelFromLegal").textContent =
    pCancel !== undefined ? fmtPct(pCancel) : "—";
  document.getElementById("kpiMarkovStepsFromLegal").textContent =
    steps !== undefined ? steps.toFixed(2) : "—";
}

// ---------------------
// KM chart (line)
// ---------------------
function drawKMChart(selector, data) {
  const el = document.querySelector(selector);
  clear(el);

  const width = el.clientWidth;
  const height = el.clientHeight;

  const margin = { top: 18, right: 14, bottom: 36, left: 46 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const svg = d3.select(el)
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  const g = svg.append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  // Data
  const x = d3.scaleLinear()
    .domain(d3.extent(data, d => +d.timeline))
    .nice()
    .range([0, w]);

  const y = d3.scaleLinear()
    .domain([0, 1])
    .range([h, 0]);

  // Axes
  g.append("g")
    .attr("transform", `translate(0,${h})`)
    .call(d3.axisBottom(x).ticks(6))
    .call(styleAxis);

  g.append("g")
    .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")))
    .call(styleAxis);

  // Grid
  g.append("g")
    .attr("class", "grid")
    .call(d3.axisLeft(y).ticks(5).tickSize(-w).tickFormat(""))
    .call(styleGrid);

  // Line
  const line = d3.line()
    .x(d => x(+d.timeline))
    .y(d => y(+d.survival_prob));

  g.append("path")
    .datum(data)
    .attr("fill", "none")
    .attr("stroke-width", 2.2)
    .attr("stroke", "rgba(45,212,191,0.95)")
    .attr("d", line);

  // Labels
  g.append("text")
    .attr("x", 0)
    .attr("y", -4)
    .attr("fill", "rgba(230,243,251,0.85)")
    .attr("font-size", 12)
    .text("Survival probability over time (days since approval)");
}

// ---------------------
// Markov Heatmap (5x5)
// ---------------------
function drawMarkovHeatmap(selector, P) {
  const el = document.querySelector(selector);
  clear(el);

  const width = el.clientWidth;
  const height = el.clientHeight;

  const margin = { top: 34, right: 18, bottom: 90, left: 150 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const svg = d3.select(el)
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  const g = svg.append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  const n = P.length;

  const x = d3.scaleBand()
    .domain(d3.range(n))
    .range([0, w])
    .paddingInner(0.05);

  const y = d3.scaleBand()
    .domain(d3.range(n))
    .range([0, h])
    .paddingInner(0.05);

  const flat = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      flat.push({ i, j, v: +P[i][j] });
    }
  }

  const vmax = d3.max(flat, d => d.v) || 1;

  // teal/blue-ish continuous
  const color = d3.scaleLinear()
    .domain([0, vmax])
    .range(["rgba(56,189,248,0.05)", "rgba(45,212,191,0.95)"]);

  // Axis labels (state names)
  const xAxis = g.append("g")
    .attr("transform", `translate(0,${h})`)
    .call(d3.axisBottom(x).tickFormat(i => STATE_LABELS[i]))
    .call(styleAxis);

  xAxis.selectAll("text")
    .style("text-anchor", "end")
    .attr("transform", "rotate(-35)")
    .attr("dx", "-0.6em")
    .attr("dy", "0.2em");

  g.append("g")
    .call(d3.axisLeft(y).tickFormat(i => STATE_LABELS[i]))
    .call(styleAxis);

  // Cells
  const cell = g.selectAll("rect")
    .data(flat)
    .enter()
    .append("rect")
    .attr("x", d => x(d.j))
    .attr("y", d => y(d.i))
    .attr("width", x.bandwidth())
    .attr("height", y.bandwidth())
    .attr("rx", 6)
    .attr("fill", d => color(d.v))
    .attr("stroke", "rgba(255,255,255,0.08)")
    .attr("stroke-width", 1);

  // Values
  g.selectAll("text.cellv")
    .data(flat)
    .enter()
    .append("text")
    .attr("class", "cellv")
    .attr("x", d => x(d.j) + x.bandwidth() / 2)
    .attr("y", d => y(d.i) + y.bandwidth() / 2 + 4)
    .attr("text-anchor", "middle")
    .attr("fill", d => (d.v > 0.35 ? "rgba(7,26,36,0.9)" : "rgba(230,243,251,0.85)"))
    .attr("font-size", 12)
    .attr("font-weight", 650)
    .text(d => (d.v === 0 ? "" : d.v.toFixed(2)));

  // Title
  svg.append("text")
    .attr("x", margin.left)
    .attr("y", 18)
    .attr("fill", "rgba(230,243,251,0.85)")
    .attr("font-size", 12)
    .text("P(next_state | current_state)");
}

// ---------------------
// Region closure rate chart (bar)
// ---------------------
function drawRegionChart(selector, rows) {
  const el = document.querySelector(selector);
  clear(el);

  // Sort by closure_rate desc
  const data = [...rows].sort((a, b) => (b.closure_rate ?? 0) - (a.closure_rate ?? 0));

  const width = el.clientWidth;
  const height = el.clientHeight;

  const margin = { top: 18, right: 14, bottom: 40, left: 46 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const svg = d3.select(el)
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  const g = svg.append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  const x = d3.scaleBand()
    .domain(data.map(d => d.Region))
    .range([0, w])
    .padding(0.22);

  const y = d3.scaleLinear()
    .domain([0, d3.max(data, d => +d.closure_rate) || 1])
    .nice()
    .range([h, 0]);

  g.append("g")
    .attr("transform", `translate(0,${h})`)
    .call(d3.axisBottom(x))
    .call(styleAxis);

  g.append("g")
    .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")))
    .call(styleAxis);

  g.append("g")
    .attr("class", "grid")
    .call(d3.axisLeft(y).ticks(5).tickSize(-w).tickFormat(""))
    .call(styleGrid);

  g.selectAll("rect")
    .data(data)
    .enter()
    .append("rect")
    .attr("x", d => x(d.Region))
    .attr("y", d => y(+d.closure_rate))
    .attr("width", x.bandwidth())
    .attr("height", d => h - y(+d.closure_rate))
    .attr("rx", 10)
    .attr("fill", "rgba(56,189,248,0.65)")
    .attr("stroke", "rgba(255,255,255,0.10)");

  // Value labels
  g.selectAll("text.val")
    .data(data)
    .enter()
    .append("text")
    .attr("class", "val")
    .attr("x", d => x(d.Region) + x.bandwidth() / 2)
    .attr("y", d => y(+d.closure_rate) - 6)
    .attr("text-anchor", "middle")
    .attr("fill", "rgba(230,243,251,0.85)")
    .attr("font-size", 12)
    .text(d => fmtPct(+d.closure_rate));
}

// ---------------------
// Anomalies table
// ---------------------
function renderAnomaliesTable(rows) {
  const tbody = document.querySelector("#anomaliesTable tbody");
  tbody.innerHTML = "";

  rows.forEach((r, idx) => {
    const tr = document.createElement("tr");

    const stagn = r.stagnation_flag ? "Yes" : "No";
    const bottleneck = r.bottleneck_flag ? "Yes" : "No";

    const cells = [
      idx + 1,
      r.Status ?? "—",
      fmtMoney(+r.financing),
      fmtNumber(+r.duration_days),
      r.velocity !== undefined && r.velocity !== null ? (+r.velocity).toFixed(2) : "—",
      stagn,
      bottleneck,
      r.anomaly_score !== undefined && r.anomaly_score !== null ? (+r.anomaly_score).toFixed(3) : "—",
    ];

    cells.forEach(c => {
      const td = document.createElement("td");
      td.textContent = String(c);
      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });
}

// ---------------------
// Styling helpers for axes/grid
// ---------------------
function styleAxis(g) {
  g.selectAll("path").attr("stroke", "rgba(230,243,251,0.25)");
  g.selectAll("line").attr("stroke", "rgba(230,243,251,0.18)");
  g.selectAll("text").attr("fill", "rgba(230,243,251,0.75)").attr("font-size", 11);
  return g;
}

function styleGrid(g) {
  g.selectAll("line").attr("stroke", "rgba(230,243,251,0.08)");
  g.selectAll("path").attr("stroke", "rgba(0,0,0,0)");
  return g;
}

// ---------------------
// Boot
// ---------------------
async function boot() {
  // health
  try {
    const h = await health();
    const chip = document.getElementById("healthChip");
    chip.textContent = `API: ok • ${h.files?.length ?? 0} outputs`;
    chip.style.borderColor = "rgba(52,211,153,0.45)";
    chip.style.background = "rgba(52,211,153,0.10)";
  } catch (e) {
    const chip = document.getElementById("healthChip");
    chip.textContent = "API: offline";
    chip.style.borderColor = "rgba(251,113,133,0.55)";
    chip.style.background = "rgba(251,113,133,0.10)";
  }

  // Load data (parallel)
  const [km, anomalies, vistaSummary, regionStats, P, absorption] = await Promise.all([
    getKM(),
    getAnomalies(25),
    getVistaSummary(),
    getSegment("Region"),
    getMarkovMatrix(),
    getMarkovAbsorption(),
  ]);

  renderKpis(vistaSummary, absorption);
  drawKMChart("#kmChart", km);
  drawMarkovHeatmap("#markovHeatmap", P);
  drawRegionChart("#regionChart", regionStats);
  renderAnomaliesTable(anomalies);

  // Redraw on resize (basic)
  let t = null;
  window.addEventListener("resize", () => {
    if (t) clearTimeout(t);
    t = setTimeout(() => {
      drawKMChart("#kmChart", km);
      drawMarkovHeatmap("#markovHeatmap", P);
      drawRegionChart("#regionChart", regionStats);
    }, 120);
  });
}

boot().catch(err => {
  console.error(err);
});