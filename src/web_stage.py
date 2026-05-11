"""HTML/CSS/JS stage for a polished interactive simulation view."""

from __future__ import annotations

import html
import json

import networkx as nx
import numpy as np

from src.simulation import SimulationResult


def build_stage_html(
    graph: nx.Graph,
    result: SimulationResult,
    pos: dict[int, tuple[float, float]],
    topology: str,
    visual_theme: str = "Auto",
) -> str:
    """Return a self-contained interactive canvas stage."""
    payload = _stage_payload(graph, result, pos, topology, visual_theme)
    data_json = json.dumps(payload, separators=(",", ":"))

    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
:root {{
  color-scheme: light dark;
  --blue: #5ac8fa;
  --amber: #ffcc00;
  --red: #ff453a;
  --gray: #8e8e93;
}}
body[data-theme="light"] {{
  --stage-bg:
    radial-gradient(circle at 18% 10%, rgba(90, 200, 250, 0.30), transparent 30%),
    radial-gradient(circle at 76% 4%, rgba(255, 69, 58, 0.18), transparent 32%),
    linear-gradient(145deg, #f8fbff 0%, #eef4fb 46%, #f6f1f4 100%);
  --stage-border: rgba(0, 0, 0, 0.08);
  --stage-shadow: 0 28px 70px rgba(22, 27, 37, 0.15);
  --grid-a: rgba(14, 23, 38, 0.075);
  --grid-b: rgba(14, 23, 38, 0.055);
  --text: #111827;
  --muted: rgba(17, 24, 39, 0.70);
  --panel: rgba(255, 255, 255, 0.62);
  --panel-strong: rgba(255, 255, 255, 0.76);
  --line: rgba(15, 23, 42, 0.12);
  --button-bg: #111827;
  --button-text: #ffffff;
  --secondary-bg: rgba(255, 255, 255, 0.62);
  --secondary-text: #111827;
}}
body[data-theme="dark"] {{
  --stage-bg:
    radial-gradient(circle at 20% 12%, rgba(90, 200, 250, 0.22), transparent 28%),
    radial-gradient(circle at 75% 0%, rgba(255, 69, 58, 0.22), transparent 30%),
    linear-gradient(145deg, #050814 0%, #0a1020 45%, #111827 100%);
  --stage-border: rgba(255, 255, 255, 0.13);
  --stage-shadow: 0 32px 70px rgba(0, 0, 0, 0.34);
  --grid-a: rgba(255, 255, 255, 0.035);
  --grid-b: rgba(255, 255, 255, 0.025);
  --text: #f9fafb;
  --muted: rgba(229, 231, 235, 0.82);
  --panel: rgba(17, 24, 39, 0.72);
  --panel-strong: rgba(8, 13, 25, 0.74);
  --line: rgba(255, 255, 255, 0.13);
  --button-bg: #f8fafc;
  --button-text: #030712;
  --secondary-bg: rgba(255, 255, 255, 0.13);
  --secondary-text: #ffffff;
}}
* {{ box-sizing: border-box; }}
html, body {{
  margin: 0;
  padding: 0;
  overflow: hidden;
  height: 100%;
  background: transparent;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", "Segoe UI", sans-serif;
}}
.stage {{
  position: relative;
  height: 100%;
  min-height: 460px;
  border-radius: 30px;
  overflow: hidden;
  background: var(--stage-bg);
  border: 1px solid var(--stage-border);
  box-shadow: var(--stage-shadow);
}}
.stage::before {{
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background-image:
    linear-gradient(var(--grid-a) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid-b) 1px, transparent 1px);
  background-size: 42px 42px;
  mask-image: linear-gradient(to bottom, rgba(0,0,0,0.9), transparent 72%);
}}
.stage-top {{
  position: absolute;
  z-index: 4;
  left: 24px;
  right: 24px;
  top: 18px;
  display: flex;
  justify-content: space-between;
  gap: 20px;
  align-items: flex-start;
}}
.title-block {{
  max-width: 520px;
}}
.kicker {{
  color: rgba(255, 204, 0, 0.96);
  text-transform: uppercase;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.13em;
}}
h1 {{
  margin: 6px 0 5px;
  color: var(--text);
  font-size: clamp(26px, 3.5vw, 42px);
  line-height: 0.96;
  letter-spacing: -0.04em;
  font-weight: 860;
}}
.subtitle {{
  margin: 0;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.25;
}}
.live-pill {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 11px;
  border-radius: 999px;
  background: var(--panel);
  border: 1px solid var(--line);
  color: var(--text);
  font-size: 13px;
  font-weight: 740;
  backdrop-filter: blur(18px);
  white-space: nowrap;
}}
.live-dot {{
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--red);
  box-shadow: 0 0 22px var(--red);
}}
.metrics {{
  position: absolute;
  z-index: 4;
  left: 24px;
  right: 24px;
  top: 116px;
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 9px;
}}
.metric {{
  min-height: 50px;
  padding: 8px 10px;
  border-radius: 16px;
  background: var(--panel);
  border: 1px solid var(--line);
  backdrop-filter: blur(20px);
}}
.metric .label {{
  color: var(--muted);
  font-size: 10px;
  font-weight: 760;
  letter-spacing: 0.01em;
}}
.metric .value {{
  color: var(--text);
  font-size: 20px;
  font-weight: 820;
  line-height: 1.1;
  letter-spacing: -0.03em;
  margin-top: 4px;
}}
.canvas-wrap {{
  position: absolute;
  inset: 0;
}}
#networkCanvas {{
  width: 100%;
  height: 100%;
  display: block;
  cursor: grab;
  touch-action: none;
}}
#networkCanvas.is-dragging {{
  cursor: grabbing;
}}
.stage-hint {{
  position: absolute;
  z-index: 4;
  left: 24px;
  bottom: 86px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 11px;
  border-radius: 999px;
  color: var(--muted);
  background: var(--panel);
  border: 1px solid var(--line);
  backdrop-filter: blur(20px);
  font-size: 11px;
  font-weight: 720;
}}
.side-panel {{
  position: absolute;
  z-index: 4;
  right: 22px;
  bottom: 88px;
  width: min(286px, calc(100% - 44px));
  padding: 12px;
  border-radius: 22px;
  background: var(--panel-strong);
  border: 1px solid var(--line);
  backdrop-filter: blur(24px);
}}
.panel-title {{
  color: var(--text);
  font-size: 13px;
  font-weight: 790;
  margin-bottom: 10px;
}}
#curveCanvas {{
  width: 100%;
  height: 112px;
  display: block;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.12);
}}
.state-mix {{
  display: flex;
  width: 100%;
  height: 10px;
  overflow: hidden;
  border-radius: 999px;
  margin-top: 11px;
  background: rgba(255, 255, 255, 0.08);
}}
.mix-segment {{ height: 100%; }}
.legend {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
  margin-top: 10px;
  color: var(--muted);
  font-size: 11px;
  font-weight: 650;
}}
.legend.directed {{
  grid-template-columns: repeat(2, 1fr);
}}
.legend span {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}
.swatch {{
  width: 9px;
  height: 9px;
  border-radius: 50%;
  display: inline-block;
}}
.influencer-swatch {{
  background: transparent;
  border: 2px solid var(--amber);
}}
.controls {{
  position: absolute;
  z-index: 5;
  left: 24px;
  right: 24px;
  bottom: 16px;
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 14px;
  align-items: center;
  padding: 10px;
  border-radius: 22px;
  background: var(--panel);
  border: 1px solid var(--line);
  backdrop-filter: blur(24px);
}}
button {{
  appearance: none;
  border: 0;
  border-radius: 999px;
  padding: 9px 16px;
  background: var(--button-bg);
  color: var(--button-text);
  font-weight: 800;
  font-size: 14px;
  cursor: pointer;
  box-shadow: 0 14px 34px rgba(0, 0, 0, 0.28);
}}
button.secondary {{
  width: 36px;
  height: 36px;
  padding: 0;
  background: var(--secondary-bg);
  color: var(--secondary-text);
  border: 1px solid var(--line);
  box-shadow: none;
}}
.range-wrap {{
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 12px;
  align-items: center;
}}
input[type="range"] {{
  width: 100%;
  accent-color: var(--button-bg);
}}
.step-readout {{
  color: var(--text);
  font-size: 13px;
  font-weight: 760;
  min-width: 80px;
  text-align: right;
}}
.speed {{
  display: inline-flex;
  align-items: center;
  gap: 10px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}}
.speed input {{ width: 92px; }}
@media (max-width: 680px) {{
  .stage {{ min-height: 760px; border-radius: 24px; }}
  .stage-top {{ flex-direction: column; }}
  .metrics {{
    top: 220px;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}
  .metric:nth-child(5) {{ grid-column: span 2; }}
  .side-panel {{
    left: 20px;
    right: 20px;
    width: auto;
    bottom: 118px;
  }}
  .stage-hint {{
    display: none;
  }}
  .controls {{
    left: 20px;
    right: 20px;
    grid-template-columns: 1fr;
  }}
  h1 {{ font-size: 42px; }}
}}
</style>
</head>
<body>
<section class="stage">
  <div class="canvas-wrap">
    <canvas id="networkCanvas"></canvas>
  </div>

  <div class="stage-top">
    <div class="title-block">
      <div class="kicker">{html.escape(payload["kicker"])}</div>
      <h1>Social Contagion</h1>
      <p class="subtitle">{html.escape(payload["subtitle"])}</p>
    </div>
    <div class="live-pill"><span class="live-dot"></span><span id="statusText">Live cascade</span></div>
  </div>

  <div class="metrics" id="metrics"></div>
  <div class="stage-hint">Drag to move · Scroll to zoom · Double-click to reset</div>

  <aside class="side-panel">
    <div class="panel-title">Temporal Wave</div>
    <canvas id="curveCanvas"></canvas>
    <div class="state-mix" id="stateMix"></div>
    <div class="legend" id="legend">
      <span><i class="swatch" style="background: var(--blue)"></i>S</span>
      <span><i class="swatch" style="background: var(--amber)"></i>E</span>
      <span><i class="swatch" style="background: var(--red)"></i>I</span>
      <span><i class="swatch" style="background: var(--gray)"></i>R</span>
      <span id="influencerLegend"><i class="swatch influencer-swatch"></i>Influencer</span>
    </div>
  </aside>

  <div class="controls">
    <div>
      <button id="playButton">Play</button>
      <button class="secondary" id="prevButton">‹</button>
      <button class="secondary" id="nextButton">›</button>
      <button class="secondary" id="zoomOutButton">−</button>
      <button class="secondary" id="zoomInButton">+</button>
      <button class="secondary" id="resetViewButton">↺</button>
    </div>
    <div class="range-wrap">
      <input id="stepRange" type="range" min="0" max="{payload["max_step"]}" value="{payload["peak_step"]}" />
      <div class="step-readout" id="stepReadout"></div>
    </div>
    <label class="speed">Speed <input id="speedRange" type="range" min="40" max="800" value="140" /></label>
  </div>
</section>

<script>
const data = {data_json};
const colors = {{
  0: "#5ac8fa",
  1: "#ff453a",
  2: "#8e8e93",
  3: "#ffcc00"
}};
const labels = {{0: "S", 1: "I", 2: "R", 3: "E"}};
const stateOrder = [
  ["S", 0, "#5ac8fa"],
  ["E", 3, "#ffcc00"],
  ["I", 1, "#ff453a"],
  ["R", 2, "#8e8e93"]
];
const networkCanvas = document.getElementById("networkCanvas");
const networkCtx = networkCanvas.getContext("2d");
const curveCanvas = document.getElementById("curveCanvas");
const curveCtx = curveCanvas.getContext("2d");
const playButton = document.getElementById("playButton");
const prevButton = document.getElementById("prevButton");
const nextButton = document.getElementById("nextButton");
const stepRange = document.getElementById("stepRange");
const speedRange = document.getElementById("speedRange");
const stepReadout = document.getElementById("stepReadout");
const metrics = document.getElementById("metrics");
const stateMix = document.getElementById("stateMix");
const statusText = document.getElementById("statusText");
const legend = document.getElementById("legend");
const influencerLegend = document.getElementById("influencerLegend");
const zoomOutButton = document.getElementById("zoomOutButton");
const zoomInButton = document.getElementById("zoomInButton");
const resetViewButton = document.getElementById("resetViewButton");

const palettes = {{
  light: {{
    backgroundGlow: ["rgba(255,255,255,0.32)", "rgba(90,200,250,0.10)", "rgba(255,255,255,0)"],
    grid: "rgba(15,23,42,0.08)",
    edgeIdle: "rgba(30,41,59,0.17)",
    edgeActive: "rgba(15,23,42,0.34)",
    nodeStroke: "rgba(255,255,255,0.95)",
    curveBg: "rgba(255,255,255,0.54)",
    curveGrid: "rgba(15,23,42,0.12)",
    curveMarker: "rgba(17,24,39,0.72)"
  }},
  dark: {{
    backgroundGlow: ["rgba(255,255,255,0.045)", "rgba(90,200,250,0.030)", "rgba(0,0,0,0)"],
    grid: "rgba(255,255,255,0.08)",
    edgeIdle: "rgba(148,163,184,0.12)",
    edgeActive: "rgba(255,255,255,0.22)",
    nodeStroke: "rgba(255,255,255,0.88)",
    curveBg: "rgba(255,255,255,0.025)",
    curveGrid: "rgba(255,255,255,0.10)",
    curveMarker: "rgba(255,255,255,0.75)"
  }}
}};

function resolveTheme() {{
  if (data.theme === "Dark") return "dark";
  if (data.theme === "Light") return "light";
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}}

let activeTheme = resolveTheme();
let palette = palettes[activeTheme];
document.body.dataset.theme = activeTheme;

let step = Number(stepRange.value);
let playing = false;
let playbackTimer = null;
let canvasW = 0;
let canvasH = 0;
let curveW = 0;
let curveH = 0;
let view = {{ scale: 1, x: 0, y: 0 }};
let dragging = false;
let lastPointer = {{ x: 0, y: 0 }};

if (data.directed) {{
  legend.classList.add("directed");
}} else {{
  influencerLegend.style.display = "none";
}}

function resizeCanvas(canvas, ctx) {{
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [rect.width, rect.height];
}}

function resize() {{
  [canvasW, canvasH] = resizeCanvas(networkCanvas, networkCtx);
  [curveW, curveH] = resizeCanvas(curveCanvas, curveCtx);
}}

function baseNodePoint(node) {{
  const marginX = Math.min(190, canvasW * 0.15);
  const marginTop = Math.min(188, canvasH * 0.34);
  const marginBottom = Math.min(92, canvasH * 0.16);
  const usableW = Math.max(520, canvasW - marginX * 0.92);
  const usableH = Math.max(245, canvasH - marginTop - marginBottom);
  return [
    marginX * 0.24 + node.x * usableW,
    marginTop + node.y * usableH
  ];
}}

function nodePoint(node) {{
  const [x, y] = baseNodePoint(node);
  const cx = canvasW / 2;
  const cy = canvasH / 2;
  return [
    cx + (x - cx) * view.scale + view.x,
    cy + (y - cy) * view.scale + view.y
  ];
}}

function clamp(value, min, max) {{
  return Math.max(min, Math.min(max, value));
}}

function zoomAt(clientX, clientY, factor) {{
  const rect = networkCanvas.getBoundingClientRect();
  const px = clientX - rect.left;
  const py = clientY - rect.top;
  const cx = canvasW / 2;
  const cy = canvasH / 2;
  const oldScale = view.scale;
  const nextScale = clamp(oldScale * factor, 0.45, 4.0);
  const worldX = (px - cx - view.x) / oldScale;
  const worldY = (py - cy - view.y) / oldScale;
  view.scale = nextScale;
  view.x = px - cx - worldX * nextScale;
  view.y = py - cy - worldY * nextScale;
  drawNetwork();
}}

function zoomFromCenter(factor) {{
  zoomAt(canvasW / 2, canvasH / 2, factor);
}}

function resetView() {{
  view = {{ scale: 1, x: 0, y: 0 }};
  drawNetwork();
}}

function drawBackground() {{
  networkCtx.clearRect(0, 0, canvasW, canvasH);
  const g = networkCtx.createRadialGradient(canvasW * 0.52, canvasH * 0.52, 40, canvasW * 0.52, canvasH * 0.52, canvasW * 0.74);
  g.addColorStop(0, palette.backgroundGlow[0]);
  g.addColorStop(0.38, palette.backgroundGlow[1]);
  g.addColorStop(1, palette.backgroundGlow[2]);
  networkCtx.fillStyle = g;
  networkCtx.fillRect(0, 0, canvasW, canvasH);

  networkCtx.save();
  networkCtx.strokeStyle = palette.grid;
  for (let x = 0; x < canvasW; x += 42) {{
    networkCtx.beginPath();
    networkCtx.moveTo(x, 0);
    networkCtx.lineTo(x, canvasH);
    networkCtx.stroke();
  }}
  for (let y = 0; y < canvasH; y += 42) {{
    networkCtx.beginPath();
    networkCtx.moveTo(0, y);
    networkCtx.lineTo(canvasW, y);
    networkCtx.stroke();
  }}
  networkCtx.restore();
}}

function drawArrowHead(x1, y1, x2, y2, color, targetRadius) {{
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const tipX = x2 - Math.cos(angle) * (targetRadius + 2);
  const tipY = y2 - Math.sin(angle) * (targetRadius + 2);
  const size = 5.8;

  networkCtx.save();
  networkCtx.fillStyle = color;
  networkCtx.beginPath();
  networkCtx.moveTo(tipX, tipY);
  networkCtx.lineTo(
    tipX - Math.cos(angle - Math.PI / 7) * size,
    tipY - Math.sin(angle - Math.PI / 7) * size
  );
  networkCtx.lineTo(
    tipX - Math.cos(angle + Math.PI / 7) * size,
    tipY - Math.sin(angle + Math.PI / 7) * size
  );
  networkCtx.closePath();
  networkCtx.fill();
  networkCtx.restore();
}}

function drawNetwork() {{
  const states = data.states[step];
  drawBackground();

  networkCtx.save();
  networkCtx.lineCap = "round";
  for (const [a, b] of data.edges) {{
    const source = data.nodes[a];
    const target = data.nodes[b];
    const [x1, y1] = nodePoint(source);
    const [x2, y2] = nodePoint(target);
    const isActive = states[a] === 1 || states[b] === 1 || states[a] === 3 || states[b] === 3;
    const edgeColor = isActive ? palette.edgeActive : palette.edgeIdle;
    networkCtx.strokeStyle = edgeColor;
    networkCtx.lineWidth = isActive ? 1.05 : 0.7;
    networkCtx.beginPath();
    networkCtx.moveTo(x1, y1);
    networkCtx.lineTo(x2, y2);
    networkCtx.stroke();
    if (data.directed) {{
      const targetRadius = 3.0 + target.degreeScale * 7.0;
      drawArrowHead(x1, y1, x2, y2, edgeColor, targetRadius);
    }}
  }}
  networkCtx.restore();

  for (const node of data.nodes) {{
    const state = states[node.id];
    const [x, y] = nodePoint(node);
    const base = 3.0 + node.degreeScale * 7.0;
    if (node.influencer) {{
      networkCtx.strokeStyle = "#ffcc00";
      networkCtx.lineWidth = 2.4;
      networkCtx.beginPath();
      networkCtx.arc(x, y, base + 3.0, 0, Math.PI * 2);
      networkCtx.stroke();
    }}
    networkCtx.fillStyle = colors[state];
    networkCtx.strokeStyle = node.influencer ? "#ffcc00" : palette.nodeStroke;
    networkCtx.lineWidth = node.influencer ? 1.7 : state === 1 ? 1.6 : 0.9;
    networkCtx.beginPath();
    networkCtx.arc(x, y, base, 0, Math.PI * 2);
    networkCtx.fill();
    networkCtx.stroke();
  }}
}}

function drawCurve() {{
  curveCtx.clearRect(0, 0, curveW, curveH);
  curveCtx.fillStyle = palette.curveBg;
  curveCtx.fillRect(0, 0, curveW, curveH);

  const pad = 26;
  const maxY = Math.max(1, data.node_count);
  curveCtx.strokeStyle = palette.curveGrid;
  curveCtx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {{
    const y = pad + (curveH - pad * 2) * i / 4;
    curveCtx.beginPath();
    curveCtx.moveTo(pad, y);
    curveCtx.lineTo(curveW - pad, y);
    curveCtx.stroke();
  }}

  function point(index, value) {{
    const x = pad + (curveW - pad * 2) * index / data.max_step;
    const y = curveH - pad - (curveH - pad * 2) * value / maxY;
    return [x, y];
  }}

  for (const [name, id, color] of stateOrder) {{
    curveCtx.strokeStyle = color;
    curveCtx.lineWidth = name === "I" ? 2.4 : 1.8;
    curveCtx.globalAlpha = name === "E" && data.history.E.every(v => v === 0) ? 0.28 : 0.96;
    curveCtx.beginPath();
    data.history[name].forEach((value, index) => {{
      const [x, y] = point(index, value);
      if (index === 0) curveCtx.moveTo(x, y);
      else curveCtx.lineTo(x, y);
    }});
    curveCtx.stroke();
  }}
  curveCtx.globalAlpha = 1;

  const x = pad + (curveW - pad * 2) * step / data.max_step;
  curveCtx.strokeStyle = palette.curveMarker;
  curveCtx.setLineDash([5, 5]);
  curveCtx.beginPath();
  curveCtx.moveTo(x, pad - 6);
  curveCtx.lineTo(x, curveH - pad + 6);
  curveCtx.stroke();
  curveCtx.setLineDash([]);
}}

function updateMetrics() {{
  const row = data.rows[step];
  metrics.innerHTML = [
    ["Cascade", `${{(data.metrics.final_cascade_size * 100).toFixed(1)}}%`],
    ["Peak I", `${{(data.metrics.peak_infected_fraction * 100).toFixed(1)}}%`],
    ["Step", `${{step}} / ${{data.max_step}}`],
    ["Active E+I", `${{(((row.E + row.I) / data.node_count) * 100).toFixed(1)}}%`],
    ["Edges", data.edge_count.toLocaleString()]
  ].map(([label, value]) => `<div class="metric"><div class="label">${{label}}</div><div class="value">${{value}}</div></div>`).join("");

  stateMix.innerHTML = stateOrder.map(([name, id, color]) => {{
    const width = row[name] / data.node_count * 100;
    return `<div class="mix-segment" style="width:${{width}}%; background:${{color}}"></div>`;
  }}).join("");

  stepReadout.textContent = `Step ${{step}}`;
  stepRange.value = step;
  statusText.textContent = `${{labels[data.states[step].includes(1) ? 1 : data.states[step].includes(3) ? 3 : 2]}} dominant wave`;
}}

function setStep(nextStep) {{
  const boundedStep = Math.max(0, Math.min(data.max_step, nextStep));
  if (boundedStep === step) return;
  step = boundedStep;
  updateMetrics();
  drawCurve();
  drawNetwork();
}}

function advancePlayback() {{
  setStep(step >= data.max_step ? 0 : step + 1);
}}

function stopPlayback() {{
  playing = false;
  playButton.textContent = "Play";
  if (playbackTimer !== null) {{
    window.clearInterval(playbackTimer);
    playbackTimer = null;
  }}
}}

function startPlayback() {{
  stopPlayback();
  playing = true;
  playButton.textContent = "Pause";
  advancePlayback();
  playbackTimer = window.setInterval(advancePlayback, Number(speedRange.value));
}}

playButton.addEventListener("click", () => {{
  if (playing) stopPlayback();
  else startPlayback();
}});
prevButton.addEventListener("click", () => setStep(step - 1));
nextButton.addEventListener("click", () => setStep(step + 1));
zoomOutButton.addEventListener("click", () => zoomFromCenter(0.82));
zoomInButton.addEventListener("click", () => zoomFromCenter(1.22));
resetViewButton.addEventListener("click", resetView);
stepRange.addEventListener("input", (event) => setStep(Number(event.target.value)));
networkCanvas.addEventListener("pointerdown", (event) => {{
  dragging = true;
  lastPointer = {{ x: event.clientX, y: event.clientY }};
  networkCanvas.classList.add("is-dragging");
  networkCanvas.setPointerCapture(event.pointerId);
}});
networkCanvas.addEventListener("pointermove", (event) => {{
  if (!dragging) return;
  view.x += event.clientX - lastPointer.x;
  view.y += event.clientY - lastPointer.y;
  lastPointer = {{ x: event.clientX, y: event.clientY }};
  drawNetwork();
}});
networkCanvas.addEventListener("pointerup", (event) => {{
  dragging = false;
  networkCanvas.classList.remove("is-dragging");
  networkCanvas.releasePointerCapture(event.pointerId);
}});
networkCanvas.addEventListener("pointercancel", () => {{
  dragging = false;
  networkCanvas.classList.remove("is-dragging");
}});
networkCanvas.addEventListener("dblclick", resetView);
networkCanvas.addEventListener("wheel", (event) => {{
  event.preventDefault();
  const factor = event.deltaY > 0 ? 0.90 : 1.11;
  zoomAt(event.clientX, event.clientY, factor);
}}, {{ passive: false }});
speedRange.addEventListener("input", () => {{
  if (playing) {{
    startPlayback();
  }}
}});
window.addEventListener("resize", () => {{
  resize();
  drawCurve();
  drawNetwork();
}});
if (window.matchMedia) {{
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {{
    activeTheme = resolveTheme();
    palette = palettes[activeTheme];
    document.body.dataset.theme = activeTheme;
    drawCurve();
    drawNetwork();
  }});
}}

resize();
updateMetrics();
drawCurve();
drawNetwork();
</script>
</body>
</html>
"""


def _stage_payload(
    graph: nx.Graph,
    result: SimulationResult,
    pos: dict[int, tuple[float, float]],
    topology: str,
    visual_theme: str,
) -> dict:
    xs = np.array([pos[node][0] for node in graph.nodes], dtype=float)
    ys = np.array([pos[node][1] for node in graph.nodes], dtype=float)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_span = max(1e-9, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)
    max_degree = max((_visual_degree(graph, node) for node in graph.nodes), default=1)

    nodes = [
        {
            "id": int(node),
            "x": float((pos[node][0] - x_min) / x_span),
            "y": float((pos[node][1] - y_min) / y_span),
            "degree": int(_visual_degree(graph, node)),
            "degreeScale": float(_visual_degree(graph, node) / max_degree),
            "influencer": _is_influencer_node(graph, node),
        }
        for node in graph.nodes
    ]

    history = result.history
    rows = [
        {
            "S": int(row.S),
            "E": int(row.E),
            "I": int(row.I),
            "R": int(row.R),
            "ever_infected": int(row.ever_infected),
        }
        for row in history.itertuples(index=False)
    ]

    return {
        "nodes": nodes,
        "edges": [[int(u), int(v)] for u, v in graph.edges],
        "states": [[int(value) for value in state.tolist()] for state in result.states],
        "rows": rows,
        "history": {
            "S": [int(value) for value in history["S"].tolist()],
            "E": [int(value) for value in history["E"].tolist()],
            "I": [int(value) for value in history["I"].tolist()],
            "R": [int(value) for value in history["R"].tolist()],
        },
        "metrics": result.metrics,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "directed": graph.is_directed(),
        "theme": visual_theme,
        "max_step": len(result.states) - 1,
        "peak_step": int(result.metrics["peak_index"]),
        "kicker": _stage_kicker(graph, topology, result.variant),
        "subtitle": _stage_subtitle(graph),
    }


def _visual_degree(graph: nx.Graph, node: int) -> int:
    if graph.is_directed():
        return int(graph.out_degree[node])
    return int(graph.degree[node])


def _is_influencer_node(graph: nx.Graph, node: int) -> bool:
    return graph.nodes[node].get("role") == "influencer"


def _stage_kicker(graph: nx.Graph, topology: str, variant: str) -> str:
    if graph.is_directed():
        return f"Directed influence / {variant}"
    return f"{topology} network / {variant}"


def _stage_subtitle(graph: nx.Graph) -> str:
    if graph.is_directed():
        return (
            "Arrows show who can transmit influence; influencer nodes broadcast "
            "without incoming social ties."
        )
    return (
        "A real-time view of adoption pressure, exposure and recovery across a "
        f"{graph.number_of_nodes()} node network."
    )
