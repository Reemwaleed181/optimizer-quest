// ============================================================
//  arena.js — Animated training arena canvas
// ============================================================

let arenaAnimId = null;
let arenaT      = 0;
let arenaRunning = false;
let arenaLossProgress = 0; // 0→1 from real training

function startArenaAnimation() {
  arenaRunning = true;
}
function stopArenaAnimation() {
  arenaRunning = false;
}
function setArenaProgress(p) {
  arenaLossProgress = Math.max(0, Math.min(1, p));
}

function initArena() {
  const canvas = document.getElementById('arenaCanvas');
  new ResizeObserver(() => {
    canvas.width  = canvas.offsetWidth  || 680;
    canvas.height = 230;
  }).observe(canvas);
  canvas.width  = canvas.offsetWidth  || 680;
  canvas.height = 230;
  arenaLoop();
}

function arenaLoop() {
  arenaT += arenaRunning ? 0.022 : 0.007;
  const canvas = document.getElementById('arenaCanvas');
  if (canvas) drawArenaFrame(canvas, arenaT, arenaLossProgress);
  arenaAnimId = requestAnimationFrame(arenaLoop);
}

function drawArenaFrame(canvas, t, progress) {
  const W = canvas.width, H = canvas.height;
  if (!W || !H) return;
  const ctx  = canvas.getContext('2d');
  const info = typeof window.activeOpt !== 'undefined' ? OPT_INFO[window.activeOpt] : OPT_INFO['Adam'];
  const col  = info.color;

  ctx.clearRect(0, 0, W, H);

  // ── Sky ─────────────────────────────────────────────────────
  const sky = ctx.createLinearGradient(0, 0, 0, H * 0.65);
  sky.addColorStop(0,   '#040c1c');
  sky.addColorStop(0.5, '#071526');
  sky.addColorStop(1,   '#091e3a');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, H);

  // Stars
  const STARS = [
    [0.05,0.07],[0.13,0.04],[0.24,0.12],[0.37,0.03],[0.50,0.08],
    [0.62,0.03],[0.73,0.11],[0.84,0.06],[0.94,0.13],
    [0.10,0.20],[0.33,0.21],[0.56,0.17],[0.78,0.19],[0.44,0.15],
  ];
  for (const [sx, sy] of STARS) {
    const flicker = 0.4 + 0.5 * Math.abs(Math.sin(t * 0.8 + sx * 18));
    ctx.globalAlpha = flicker * 0.7;
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(sx * W, sy * H, 1, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  // ── Left cliff ──────────────────────────────────────────────
  ctx.fillStyle = '#18120a';
  ctx.beginPath();
  ctx.moveTo(0, H);
  ctx.lineTo(0, H * 0.52); ctx.lineTo(W * 0.04, H * 0.37); ctx.lineTo(W * 0.08, H * 0.46);
  ctx.lineTo(W * 0.13, H * 0.34); ctx.lineTo(W * 0.17, H * 0.50); ctx.lineTo(W * 0.21, H * 0.63);
  ctx.lineTo(W * 0.21, H); ctx.closePath(); ctx.fill();
  ctx.fillStyle = '#241b08';
  ctx.beginPath();
  ctx.moveTo(0, H); ctx.lineTo(0, H * 0.58); ctx.lineTo(W * 0.05, H * 0.45);
  ctx.lineTo(W * 0.10, H * 0.54); ctx.lineTo(W * 0.15, H * 0.44);
  ctx.lineTo(W * 0.19, H * 0.58); ctx.lineTo(W * 0.19, H); ctx.closePath(); ctx.fill();

  // ── Right cliff ─────────────────────────────────────────────
  ctx.fillStyle = '#18120a';
  ctx.beginPath();
  ctx.moveTo(W, H); ctx.lineTo(W, H * 0.52); ctx.lineTo(W * 0.96, H * 0.35);
  ctx.lineTo(W * 0.91, H * 0.45); ctx.lineTo(W * 0.86, H * 0.34);
  ctx.lineTo(W * 0.82, H * 0.51); ctx.lineTo(W * 0.79, H * 0.63);
  ctx.lineTo(W * 0.79, H); ctx.closePath(); ctx.fill();
  ctx.fillStyle = '#241b08';
  ctx.beginPath();
  ctx.moveTo(W, H); ctx.lineTo(W, H * 0.58); ctx.lineTo(W * 0.95, H * 0.44);
  ctx.lineTo(W * 0.90, H * 0.54); ctx.lineTo(W * 0.85, H * 0.43);
  ctx.lineTo(W * 0.81, H * 0.57); ctx.lineTo(W * 0.81, H); ctx.closePath(); ctx.fill();

  // ── Valley floor ────────────────────────────────────────────
  const vg = ctx.createLinearGradient(0, H * 0.5, 0, H);
  vg.addColorStop(0, '#183a25'); vg.addColorStop(0.5, '#235835'); vg.addColorStop(1, '#2e7045');
  ctx.fillStyle = vg;
  ctx.beginPath();
  ctx.moveTo(W * 0.19, H * 0.71);
  ctx.bezierCurveTo(W * 0.28, H * 0.85, W * 0.40, H * 0.88, W * 0.50, H * 0.86);
  ctx.bezierCurveTo(W * 0.60, H * 0.84, W * 0.72, H * 0.78, W * 0.81, H * 0.70);
  ctx.lineTo(W * 0.81, H); ctx.lineTo(W * 0.19, H); ctx.closePath(); ctx.fill();

  // ── Loss landscape surface ───────────────────────────────────
  const sg = ctx.createLinearGradient(0, H * 0.26, 0, H * 0.78);
  sg.addColorStop(0,    '#c4a015');
  sg.addColorStop(0.2,  '#a8bb25');
  sg.addColorStop(0.5,  '#40a85c');
  sg.addColorStop(0.85, '#208640');
  sg.addColorStop(1,    '#145e2c');
  ctx.fillStyle = sg;
  ctx.beginPath();
  ctx.moveTo(W * 0.15, H * 0.53);
  ctx.bezierCurveTo(W * 0.20, H * 0.34, W * 0.26, H * 0.57, W * 0.34, H * 0.51);
  ctx.bezierCurveTo(W * 0.38, H * 0.47, W * 0.43, H * 0.56, W * 0.50, H * 0.68);
  ctx.bezierCurveTo(W * 0.57, H * 0.78, W * 0.63, H * 0.73, W * 0.69, H * 0.66);
  ctx.bezierCurveTo(W * 0.75, H * 0.57, W * 0.79, H * 0.51, W * 0.85, H * 0.49);
  ctx.lineTo(W * 0.85, H); ctx.lineTo(W * 0.15, H); ctx.closePath(); ctx.fill();

  // Highlight ridge
  const hg = ctx.createLinearGradient(0, H * 0.26, 0, H * 0.52);
  hg.addColorStop(0,   'rgba(235,210,55,.62)');
  hg.addColorStop(0.6, 'rgba(185,215,50,.18)');
  hg.addColorStop(1,   'rgba(100,195,100,0)');
  ctx.fillStyle = hg;
  ctx.beginPath();
  ctx.moveTo(W * 0.15, H * 0.53);
  ctx.bezierCurveTo(W * 0.20, H * 0.34, W * 0.26, H * 0.57, W * 0.34, H * 0.51);
  ctx.bezierCurveTo(W * 0.44, H * 0.45, W * 0.56, H * 0.43, W * 0.68, H * 0.47);
  ctx.bezierCurveTo(W * 0.76, H * 0.49, W * 0.81, H * 0.49, W * 0.85, H * 0.49);
  ctx.lineTo(W * 0.85, H * 0.54);
  ctx.bezierCurveTo(W * 0.80, H * 0.54, W * 0.74, H * 0.55, W * 0.67, H * 0.53);
  ctx.bezierCurveTo(W * 0.56, H * 0.50, W * 0.44, H * 0.53, W * 0.34, H * 0.57);
  ctx.bezierCurveTo(W * 0.26, H * 0.61, W * 0.21, H * 0.45, W * 0.15, H * 0.58);
  ctx.closePath(); ctx.fill();

  // Contour lines
  ctx.strokeStyle = 'rgba(255,255,140,.09)';
  ctx.lineWidth = 1;
  for (let i = 0; i < 7; i++) {
    const dy = i * H * 0.038;
    ctx.beginPath();
    ctx.moveTo(W * 0.15, H * 0.53 + dy);
    ctx.bezierCurveTo(W * 0.28, H * 0.41 + dy, W * 0.42, H * 0.59 + dy, W * 0.50, H * 0.68 + dy);
    ctx.bezierCurveTo(W * 0.58, H * 0.76 + dy, W * 0.70, H * 0.63 + dy, W * 0.85, H * 0.49 + dy);
    ctx.stroke();
  }

  // Trees
  const drawTree = (x, y, h, c) => {
    ctx.fillStyle = c;
    ctx.beginPath(); ctx.moveTo(x, y - h); ctx.lineTo(x + h * 0.42, y); ctx.lineTo(x - h * 0.42, y);
    ctx.closePath(); ctx.fill();
    ctx.fillStyle = '#30190a';
    ctx.fillRect(x - h * 0.09, y, h * 0.18, h * 0.22);
  };
  drawTree(W * 0.24, H * 0.73, H * 0.058, '#1c5228');
  drawTree(W * 0.29, H * 0.76, H * 0.048, '#226030');
  drawTree(W * 0.73, H * 0.73, H * 0.055, '#1c5228');
  drawTree(W * 0.77, H * 0.76, H * 0.046, '#226030');

  // ── Target (convergence point) ───────────────────────────────
  const TX = W * 0.50, TY = H * 0.67;
  const sg2 = ctx.createRadialGradient(TX, TY + 5, 0, TX, TY + 5, 26);
  sg2.addColorStop(0, 'rgba(0,0,0,.4)'); sg2.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = sg2;
  ctx.beginPath(); ctx.ellipse(TX, TY + 7, 26, 10, 0, 0, Math.PI * 2); ctx.fill();

  const ringR  = [22, 14, 7];
  const ringC  = ['rgba(255,255,255,.15)', 'rgba(255,255,255,.30)', 'rgba(255,255,255,.65)'];
  for (let i = 0; i < 3; i++) {
    const pulse = 1 + 0.07 * Math.sin(t * 2.5 + i * 1.2);
    ctx.beginPath(); ctx.arc(TX, TY + 4, ringR[i] * pulse, 0, Math.PI * 2);
    ctx.strokeStyle = ringC[i]; ctx.lineWidth = 1.5; ctx.stroke();
  }
  // Flag pole + flag
  ctx.strokeStyle = '#ccc'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(TX, TY + 2); ctx.lineTo(TX, TY - 30); ctx.stroke();
  ctx.fillStyle = '#dc2626';
  ctx.beginPath(); ctx.moveTo(TX, TY - 30); ctx.lineTo(TX + 16, TY - 22); ctx.lineTo(TX, TY - 14);
  ctx.closePath(); ctx.fill();

  // ── Optimizer Ball ─────────────────────────────────────────
  // Ball position driven by real training progress
  const wobble = arenaRunning ? Math.sin(t * 3.2) * H * 0.014 : 0;
  const BX = W * 0.27 + (TX - W * 0.27) * progress;
  const BY = H * 0.38 + (TY - H * 0.38) * progress + wobble;

  // Trail
  for (let i = 1; i <= 7; i++) {
    const tp  = Math.max(0, progress - i * 0.05);
    const tx2 = W * 0.27 + (TX - W * 0.27) * tp;
    const ty2 = H * 0.38 + (TY - H * 0.38) * tp;
    ctx.globalAlpha = (1 - i / 8) * 0.22;
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(tx2, ty2, 5 - i * 0.5, 0, Math.PI * 2); ctx.fill();
  }
  ctx.globalAlpha = 1;

  // Glow
  const bg = ctx.createRadialGradient(BX, BY, 0, BX, BY, 20);
  bg.addColorStop(0, hexToRgbaArena(col, 0.50));
  bg.addColorStop(1, hexToRgbaArena(col, 0.00));
  ctx.fillStyle = bg;
  ctx.beginPath(); ctx.arc(BX, BY, 20, 0, Math.PI * 2); ctx.fill();

  // Ball body
  const bb = ctx.createRadialGradient(BX - 3, BY - 3, 1, BX, BY, 10);
  bb.addColorStop(0,   '#ddeeff');
  bb.addColorStop(0.4, '#8aaccf');
  bb.addColorStop(1,   '#1e3450');
  ctx.fillStyle = bb;
  ctx.beginPath(); ctx.arc(BX, BY, 10, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = 'rgba(255,255,255,.55)';
  ctx.beginPath(); ctx.arc(BX - 3, BY - 3, 3.5, 0, Math.PI * 2); ctx.fill();
}

function hexToRgbaArena(hex, a) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${a})`;
}
