// ============================================================
//  charts.js — Loss chart, Radar chart, Decision Boundary
// ============================================================

let lossChartInst  = null;
let radarChartInst = null;

function hexToRgba(hex, a) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${a})`;
}

// ── Loss Chart ────────────────────────────────────────────────
function initLossChart(activeOpt) {
  if (lossChartInst) { lossChartInst.destroy(); lossChartInst = null; }
  const col = OPT_INFO[activeOpt].color;
  const ctx = document.getElementById('lossChart').getContext('2d');
  lossChartInst = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Train Loss',
        data: [],
        borderColor: col,
        backgroundColor: hexToRgba(col, 0.08),
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.35,
        fill: true,
      }, {
        label: 'Validation Loss',
        data: [],
        borderColor: hexToRgba(col, 0.45),
        borderDash: [5, 4],
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.35,
        fill: false,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0f1e35',
          borderColor: '#1a3556',
          borderWidth: 1,
          titleColor: '#8ba4c8',
          bodyColor: '#e0ecff',
          padding: 8,
          callbacks: {
            title: items => 'Epoch ' + items[0].label,
            label: item => item.dataset.label + ': ' + item.raw.toFixed(4),
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(26,53,86,.4)', drawBorder: false },
          ticks: { color: '#3a5a88', font: { size: 10 }, maxTicksLimit: 8 },
          title: { display: true, text: 'Epoch', color: '#3a5a88', font: { size: 10 } },
        },
        y: {
          min: 0,
          grid: { color: 'rgba(26,53,86,.4)', drawBorder: false },
          ticks: { color: '#3a5a88', font: { size: 10 }, maxTicksLimit: 5, callback: v => v.toFixed(3) },
          title: { display: true, text: 'Loss', color: '#3a5a88', font: { size: 10 } },
        }
      }
    }
  });
}

function pushLossPoint(epoch, trainLoss, testLoss) {
  if (!lossChartInst) return;
  lossChartInst.data.labels.push(epoch);
  lossChartInst.data.datasets[0].data.push(+trainLoss.toFixed(5));
  lossChartInst.data.datasets[1].data.push(+testLoss.toFixed(5));
  lossChartInst.update('none');
}

function resetLossChart(activeOpt) {
  if (!lossChartInst) return;
  const col = OPT_INFO[activeOpt].color;
  lossChartInst.data.labels = [];
  lossChartInst.data.datasets[0].data = [];
  lossChartInst.data.datasets[0].borderColor = col;
  lossChartInst.data.datasets[0].backgroundColor = hexToRgba(col, 0.08);
  lossChartInst.data.datasets[1].data = [];
  lossChartInst.data.datasets[1].borderColor = hexToRgba(col, 0.45);
  lossChartInst.update('none');
}

// ── Radar Chart ───────────────────────────────────────────────
const RADAR_LABELS = ['Speed', 'Accuracy', 'Stability', 'Generalization', 'Efficiency'];

function initRadarChart(activeOpt) {
  if (radarChartInst) { radarChartInst.destroy(); radarChartInst = null; }
  const ctx = document.getElementById('radarChart').getContext('2d');
  const datasets = Object.entries(OPT_INFO).map(([name, info]) => ({
    label: name,
    data: info.radar,
    borderColor: info.color,
    backgroundColor: hexToRgba(info.color, name === activeOpt ? 0.12 : 0.04),
    borderWidth: name === activeOpt ? 2.5 : 1,
    pointRadius: name === activeOpt ? 4 : 2,
    pointBackgroundColor: info.color,
  }));
  radarChartInst = new Chart(ctx, {
    type: 'radar',
    data: { labels: RADAR_LABELS, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0f1e35', borderColor: '#1a3556', borderWidth: 1,
          titleColor: '#8ba4c8', bodyColor: '#e0ecff',
        }
      },
      scales: {
        r: {
          min: 0, max: 100,
          grid: { color: 'rgba(26,53,86,.5)' },
          angleLines: { color: 'rgba(26,53,86,.5)' },
          ticks: { display: false, stepSize: 20 },
          pointLabels: { color: '#6a8ab0', font: { size: 10 } },
        }
      }
    }
  });
}

function updateRadarActive(activeOpt) {
  if (!radarChartInst) return;
  radarChartInst.data.datasets.forEach(ds => {
    const isActive = ds.label === activeOpt;
    ds.borderWidth = isActive ? 2.5 : 1;
    ds.pointRadius = isActive ? 4 : 2;
    ds.backgroundColor = hexToRgba(OPT_INFO[ds.label].color, isActive ? 0.12 : 0.04);
  });
  radarChartInst.update('none');
}

// ── Decision Boundary ─────────────────────────────────────────
function drawBoundary(net, dataset) {
  const canvas = document.getElementById('boundaryCanvas');
  if (!canvas || !net || !dataset) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width  = canvas.offsetWidth  || 240;
  const H = canvas.height = canvas.offsetHeight || 160;

  const RES = 40; // grid resolution
  const imgData = ctx.createImageData(W, H);

  // Draw heatmap
  for (let px = 0; px < W; px++) {
    for (let py = 0; py < H; py++) {
      // Map pixel to data space [-3, 3]
      const x1 = (px / W) * 6 - 3;
      const x2 = (py / H) * 6 - 3;
      const out = net.rawOutput([x1, x2]);
      const idx = (py * W + px) * 4;
      // Class 0 = blue, Class 1 = orange
      if (out > 0.5) {
        imgData.data[idx]     = 251; // orange tint
        imgData.data[idx + 1] = 146;
        imgData.data[idx + 2] = 60;
        imgData.data[idx + 3] = Math.round(40 + 120 * (out - 0.5) * 2);
      } else {
        imgData.data[idx]     = 59;  // blue tint
        imgData.data[idx + 1] = 130;
        imgData.data[idx + 2] = 246;
        imgData.data[idx + 3] = Math.round(40 + 120 * (0.5 - out) * 2);
      }
    }
  }
  ctx.putImageData(imgData, 0, 0);

  // Decision boundary line (contour at 0.5)
  ctx.strokeStyle = 'rgba(255,255,255,0.6)';
  ctx.lineWidth = 1.5;
  for (let px = 0; px < W - 1; px++) {
    for (let py = 0; py < H - 1; py++) {
      const x1a = (px / W) * 6 - 3,  y1a = (py / H) * 6 - 3;
      const x1b = ((px+1) / W) * 6 - 3;
      const x1c = (px / W) * 6 - 3,  y1c = ((py+1) / H) * 6 - 3;
      const v00 = net.rawOutput([x1a, y1a]);
      const v10 = net.rawOutput([x1b, y1a]);
      const v01 = net.rawOutput([x1a, y1c]);
      if ((v00 > 0.5) !== (v10 > 0.5) || (v00 > 0.5) !== (v01 > 0.5)) {
        ctx.beginPath();
        ctx.arc(px, py, 0.7, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }

  // Draw data points
  const pts = Math.min(dataset.X.length, 120);
  for (let i = 0; i < pts; i++) {
    const [x1, x2] = dataset.X[i];
    const px = ((x1 + 3) / 6) * W;
    const py = ((x2 + 3) / 6) * H;
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fillStyle = dataset.Y[i] === 0 ? '#60a5fa' : '#fb923c';
    ctx.strokeStyle = 'rgba(0,0,0,0.5)';
    ctx.lineWidth = 0.5;
    ctx.fill();
    ctx.stroke();
  }
}

function initCharts(activeOpt) {
  initLossChart(activeOpt);
  initRadarChart(activeOpt);
}
