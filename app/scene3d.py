"""3D data scene for the overview hero band: real month-7 applications in model space.

Axes: credit_risk_score (x), name_email_similarity (depth), fraud probability on a log scale (height).
A translucent plane marks the alert threshold, so red points above it are caught fraud, red points below
are missed fraud, and blue points above it are false alarms. Drawn with a small canvas renderer (no
library, works offline), rotates slowly, drag to orbit, hover for details.
Motion rules (ui-animation skill): the entrance plays once per mount (illustrative, <= 1000 ms), auto-rotation
pauses off-screen and when the tab is hidden, and reduced motion gets a still, angled view that still drags.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ui import load_samples, load_scores

X_RANGE = (0.0, 330.0)        # credit_risk_score, covers the 1st-99th percentile of month 7
Z_LOG_RANGE = (-4.0, 0.0)     # log10 fraud probability: 0.0001 .. 1


def _norm(v, lo, hi):
    return np.clip((np.asarray(v, float) - lo) / (hi - lo), 0, 1) * 2 - 1


@st.cache_data(show_spinner=False)
def payload(threshold: float, n_legit: int = 3500, seed: int = 7) -> str:
    scores = load_scores()
    raw7 = load_samples().query("month == 7").reset_index(drop=True)
    sc7 = scores[scores.split == "test"].reset_index(drop=True)
    df = raw7[["credit_risk_score", "name_email_similarity", "fraud_bool"]].assign(score=sc7.lgbm.to_numpy())
    fraud = df[df.fraud_bool == 1]
    legit = df[df.fraud_bool == 0].sample(n_legit, random_state=seed)
    d = pd.concat([legit, fraud]).sample(frac=1, random_state=seed)
    z = _norm(np.log10(np.clip(d.score, 1e-4, 1)), *Z_LOG_RANGE)
    caught = int(((d.fraud_bool == 1) & (d.score >= threshold)).sum())
    data = {
        "x": np.round(_norm(d.credit_risk_score, *X_RANGE), 3).tolist(),
        "y": np.round(_norm(d.name_email_similarity, 0, 1), 3).tolist(),
        "z": np.round(z, 3).tolist(),
        "f": d.fraud_bool.astype(int).tolist(),
        "s": np.round(d.score, 4).tolist(),
        "cr": d.credit_risk_score.astype(int).tolist(),
        "ne": np.round(d.name_email_similarity, 2).tolist(),
        "thr": float(np.round(_norm(np.log10(threshold), *Z_LOG_RANGE), 4)),
        "thrText": f"{threshold:.4f}",
        "nFraud": int(len(fraud)), "nLegit": int(n_legit), "caught": caught,
    }
    return json.dumps(data, separators=(",", ":"))


def render(threshold: float, height: int = 440) -> dict:
    data = payload(threshold)
    components.html(_HTML.replace("__DATA__", data), height=height)
    return json.loads(data)


_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%;background:transparent;overflow:hidden;color-scheme:normal;
  font:12px "IBM Plex Sans","Segoe UI",system-ui,sans-serif}
canvas{display:block;width:100%;height:100%;cursor:grab;touch-action:none;outline:none;border-radius:12px}
canvas:focus-visible{box-shadow:0 0 0 2px rgba(255,255,255,.6)}
canvas.drag{cursor:grabbing}
#tip{position:absolute;pointer-events:none;background:rgba(9,20,48,.94);color:#eef2ff;padding:7px 10px;border-radius:8px;
  line-height:1.5;box-shadow:0 0 0 1px rgba(255,255,255,.14),0 8px 20px -8px rgba(0,0,0,.6);opacity:0;
  transition:opacity 120ms ease;white-space:nowrap;font-size:12px}
#tip b{font-weight:600}
</style></head><body>
<canvas id="c" tabindex="0" role="img"
  aria-label="Biểu đồ 3D các hồ sơ tháng 7: trục ngang là điểm rủi ro tín dụng và độ giống tên - email, trục đứng là xác suất gian lận; mặt phẳng là ngưỡng cảnh báo. Kéo hoặc dùng phím mũi tên để xoay."></canvas>
<div id="tip"></div>
<script>
(() => {
const D = __DATA__;
const N = D.x.length;
const cv = document.getElementById('c'), ctx = cv.getContext('2d'), tip = document.getElementById('tip');
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const COL = { legit: 'rgba(134,182,239,0.62)', legitTop: 'rgba(190,216,250,0.95)', fraud: '#ff6b6b',
  halo: 'rgba(255,107,107,0.20)', plane: 'rgba(255,255,255,0.09)', planeEdge: 'rgba(255,255,255,0.45)',
  grid: 'rgba(255,255,255,0.07)', box: 'rgba(255,255,255,0.16)', label: 'rgba(238,242,255,0.78)' };

let W = 0, H = 0;
function resize() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  W = cv.clientWidth; H = cv.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  wake();
}

// Camera: yaw around the vertical axis, pitch looks down from above.
const HOME = { yaw: -0.75, pitch: 0.38 };
let yaw = HOME.yaw, pitch = HOME.pitch, vyaw = 0, dragging = false, lastMove = 0, visible = true;
const AUTO = reduce ? 0 : 0.12;               // rad/s, slow enough to read
const delay = new Float32Array(N).map(() => Math.random() * 0.35);
const born = performance.now();
const ENTER_MS = reduce ? 0 : 900;
const expoOut = t => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t));

function proj(x, y, z) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const X = x * cy - y * sy, Zd = x * sy + y * cy;      // rotate in the ground plane
  const Y2 = z * cp - Zd * sp, Z2 = z * sp + Zd * cp;   // tilt toward the viewer
  const dist = 3.6, k = Math.min(W, H) * 0.29 * dist / (dist + Z2);
  return [W / 2 + X * k, H / 2 + 0.03 * H - Y2 * k, Z2, k];
}

const px = new Float32Array(N), py = new Float32Array(N), pd = new Float32Array(N), pk = new Float32Array(N);
const order = new Uint32Array(N).map((_, i) => i);

function line(a, b, color, width = 1) {
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
  ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
}
function text(s, p, align = 'center', dy = 0) {
  ctx.fillStyle = COL.label; ctx.textAlign = align; ctx.fillText(s, p[0], p[1] + dy);
}

function drawPlane() {
  const z = D.thr, c = [[-1,-1],[1,-1],[1,1],[-1,1]].map(([x, y]) => proj(x, y, z));
  ctx.fillStyle = COL.plane; ctx.beginPath();
  c.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.closePath(); ctx.fill();
  for (let g = -0.5; g <= 0.5; g += 0.5) {
    line(proj(g, -1, z), proj(g, 1, z), COL.grid); line(proj(-1, g, z), proj(1, g, z), COL.grid);
  }
  ctx.strokeStyle = COL.planeEdge; ctx.lineWidth = 1.2; ctx.beginPath();
  c.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.closePath(); ctx.stroke();
  ctx.font = '600 12px "IBM Plex Sans","Segoe UI",sans-serif';
  text('Ngưỡng ' + D.thrText, proj(1, 1, z), 'left', -6);
}

function drawBox() {
  const e = [[[-1,-1,-1],[1,-1,-1]],[[-1,-1,-1],[-1,1,-1]],[[1,-1,-1],[1,1,-1]],[[-1,1,-1],[1,1,-1]],
             [[-1,-1,-1],[-1,-1,1]]];
  e.forEach(([a, b]) => line(proj(...a), proj(...b), COL.box));
  ctx.font = '12px "IBM Plex Sans","Segoe UI",sans-serif';
  text('Điểm rủi ro tín dụng', proj(0, -1.18, -1), 'center', 14);
  text('Độ giống tên - email', proj(-1.18, 0, -1), 'center', 14);
  text('Xác suất gian lận', proj(-1, -1, 1), 'center', -10);
}

function point(i) {
  const fraud = D.f[i] === 1, k = pk[i], r = fraud ? Math.max(2.2, k * 0.012) : Math.max(1.4, k * 0.0075);
  if (fraud) { ctx.fillStyle = COL.halo; ctx.beginPath(); ctx.arc(px[i], py[i], r * 2.6, 0, 6.2832); ctx.fill(); }
  ctx.fillStyle = fraud ? COL.fraud : (D.z[i] >= D.thr ? COL.legitTop : COL.legit);
  ctx.beginPath(); ctx.arc(px[i], py[i], r, 0, 6.2832); ctx.fill();
}

let raf = 0, last = 0;
function frame(now) {
  raf = 0;
  if (!W) return;
  const dt = last ? Math.min(0.05, (now - last) / 1000) : 0; last = now;
  if (!dragging) {
    if (Math.abs(vyaw) > 1e-4) { yaw += vyaw; vyaw *= 0.94; }
    else if (now - lastMove > 2500) yaw += AUTO * dt;
  }
  ctx.clearRect(0, 0, W, H);
  drawBox();
  const age = now - born;
  for (let i = 0; i < N; i++) {
    let z = D.z[i];
    if (age < ENTER_MS + 350) {
      const t = Math.max(0, Math.min(1, (age - delay[i] * 1000) / ENTER_MS));
      z = -1 + (z + 1) * expoOut(t);
    }
    const p = proj(D.x[i], D.y[i], z); px[i] = p[0]; py[i] = p[1]; pd[i] = p[2]; pk[i] = p[3];
  }
  order.sort((a, b) => pd[b] - pd[a]);                       // far first (painter's order)
  const planeDepth = proj(0, 0, D.thr)[2];
  let planeDrawn = false;
  for (let n = 0; n < N; n++) {
    const i = order[n];
    if (!planeDrawn && pd[i] < planeDepth) { drawPlane(); planeDrawn = true; }
    point(i);
  }
  if (!planeDrawn) drawPlane();
  if (visible && !document.hidden && needsMotion()) raf = requestAnimationFrame(frame);
  else last = 0;
}
function needsMotion() {
  return dragging || Math.abs(vyaw) > 1e-4 || AUTO > 0 || performance.now() - born < ENTER_MS + 400;
}
function wake() { if (!raf) raf = requestAnimationFrame(frame); }

// Pause when scrolled off-screen or the tab is hidden (ui-animation: looping motion pauses off-screen).
new IntersectionObserver(es => { visible = es[0].isIntersecting; if (visible) wake(); }).observe(cv);
document.addEventListener('visibilitychange', () => { if (!document.hidden) wake(); });
new ResizeObserver(resize).observe(cv);

// Orbit: locked to the pointer while dragging, inertia only after release.
let lx = 0, ly = 0, vx = 0;
cv.addEventListener('pointerdown', e => { dragging = true; cv.setPointerCapture(e.pointerId); cv.classList.add('drag');
  lx = e.clientX; ly = e.clientY; vx = 0; vyaw = 0; tip.style.opacity = 0; wake(); });
cv.addEventListener('pointermove', e => {
  if (dragging) {
    const dx = e.clientX - lx, dy = e.clientY - ly; lx = e.clientX; ly = e.clientY;
    yaw += dx * 0.008; vx = dx * 0.008; pitch = Math.max(0.05, Math.min(1.2, pitch + dy * 0.006));
    lastMove = performance.now(); return;
  }
  hover(e);
});
const end = () => { if (!dragging) return; dragging = false; cv.classList.remove('drag'); vyaw = reduce ? 0 : vx;
  lastMove = performance.now(); wake(); };
cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end);
cv.addEventListener('pointerleave', () => { tip.style.opacity = 0; });
cv.addEventListener('dblclick', () => { yaw = HOME.yaw; pitch = HOME.pitch; vyaw = 0; lastMove = performance.now(); wake(); });
cv.addEventListener('keydown', e => {
  const step = { ArrowLeft: [-0.12, 0], ArrowRight: [0.12, 0], ArrowUp: [0, -0.08], ArrowDown: [0, 0.08] }[e.key];
  if (!step) return; e.preventDefault();
  yaw += step[0]; pitch = Math.max(0.05, Math.min(1.2, pitch + step[1])); lastMove = performance.now(); wake();
});

const fmt = (v, d) => v.toLocaleString('vi-VN', { minimumFractionDigits: d, maximumFractionDigits: d });
function hover(e) {
  const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  let best = -1, bd = 100;                                    // nearest point within 10px (squared distance)
  for (let i = 0; i < N; i++) { const dx = px[i] - mx, dy = py[i] - my, d2 = dx * dx + dy * dy; if (d2 < bd) { bd = d2; best = i; } }
  if (best < 0) { tip.style.opacity = 0; return; }
  const fraud = D.f[best] === 1, above = D.z[best] >= D.thr;
  const outcome = fraud ? (above ? 'Gian lận, bị chặn' : 'Gian lận, bỏ lọt') : (above ? 'Hợp lệ, báo động giả' : 'Hợp lệ, được duyệt');
  tip.innerHTML = '<b>' + outcome + '</b><br>Xác suất ' + fmt(D.s[best] * 100, 2) + '%<br>Điểm rủi ro tín dụng ' +
    D.cr[best] + ' · Giống tên - email ' + fmt(D.ne[best], 2);
  tip.style.left = Math.min(mx + 14, W - tip.offsetWidth - 6) + 'px';
  tip.style.top = Math.max(6, my - tip.offsetHeight - 10) + 'px';
  tip.style.opacity = 1;
}

resize();
})();
</script></body></html>"""
