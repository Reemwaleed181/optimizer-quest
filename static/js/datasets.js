// ============================================================
//  datasets.js — Five real 2D classification datasets
// ============================================================

function randn2() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ── Two Moons ─────────────────────────────────────────────────
// Two interleaving half-circles, classic non-linear benchmark
function makeMoons(n = 300, noise = 0.15) {
  const X = [], Y = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const t = (Math.PI * i) / (half - 1);
    X.push([Math.cos(t) + randn2() * noise, Math.sin(t) + randn2() * noise]);
    Y.push(0);
  }
  for (let i = 0; i < n - half; i++) {
    const t = (Math.PI * i) / (n - half - 1);
    X.push([1 - Math.cos(t) + randn2() * noise, 1 - Math.sin(t) - 0.5 + randn2() * noise]);
    Y.push(1);
  }
  return normalize(shuffle(X, Y));
}

// ── Circles ───────────────────────────────────────────────────
// Inner vs outer ring
function makeCircles(n = 300, noise = 0.1, factor = 0.5) {
  const X = [], Y = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const t = (2 * Math.PI * i) / half;
    X.push([Math.cos(t) + randn2() * noise, Math.sin(t) + randn2() * noise]);
    Y.push(0);
  }
  for (let i = 0; i < n - half; i++) {
    const t = (2 * Math.PI * i) / (n - half);
    X.push([factor * Math.cos(t) + randn2() * noise, factor * Math.sin(t) + randn2() * noise]);
    Y.push(1);
  }
  return normalize(shuffle(X, Y));
}

// ── Gaussian Blobs ────────────────────────────────────────────
// Two Gaussian clusters
function makeGaussian(n = 300, noise = 0.35) {
  const X = [], Y = [];
  const centres = [[-1, -1], [1, 1]];
  for (let c = 0; c < 2; c++) {
    const [cx, cy] = centres[c];
    for (let i = 0; i < Math.floor(n / 2); i++) {
      X.push([cx + randn2() * noise, cy + randn2() * noise]);
      Y.push(c);
    }
  }
  return normalize(shuffle(X, Y));
}

// ── XOR ───────────────────────────────────────────────────────
// Four quadrants, alternating labels
function makeXOR(n = 300, noise = 0.2) {
  const X = [], Y = [];
  for (let i = 0; i < n; i++) {
    const x1 = (Math.random() * 2 - 1);
    const x2 = (Math.random() * 2 - 1);
    const label = ((x1 > 0) !== (x2 > 0)) ? 1 : 0;
    X.push([x1 + randn2() * noise, x2 + randn2() * noise]);
    Y.push(label);
  }
  return normalize(shuffle(X, Y));
}

// ── Spiral ────────────────────────────────────────────────────
// Two interleaved spirals — hardest dataset
function makeSpiral(n = 300, noise = 0.1) {
  const X = [], Y = [];
  const half = Math.floor(n / 2);
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < half; i++) {
      const r = (i / half) * 1.0;
      const t = (i / half) * 4 * Math.PI + (c * Math.PI);
      X.push([r * Math.sin(t) + randn2() * noise, r * Math.cos(t) + randn2() * noise]);
      Y.push(c);
    }
  }
  return normalize(shuffle(X, Y));
}

// ── Helpers ───────────────────────────────────────────────────
function normalize(dataset) {
  const { X, Y } = dataset;
  // Compute per-feature mean and std
  const n   = X.length;
  const dim = X[0].length;
  const means = new Array(dim).fill(0);
  const stds  = new Array(dim).fill(0);

  for (const x of X) for (let d = 0; d < dim; d++) means[d] += x[d];
  for (let d = 0; d < dim; d++) means[d] /= n;
  for (const x of X) for (let d = 0; d < dim; d++) stds[d] += (x[d] - means[d]) ** 2;
  for (let d = 0; d < dim; d++) stds[d] = Math.sqrt(stds[d] / n) || 1;

  const Xn = X.map(x => x.map((v, d) => (v - means[d]) / stds[d]));
  return { X: Xn, Y, means, stds, raw: X };
}

function shuffle(X, Y) {
  const idx = Array.from({ length: X.length }, (_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return { X: idx.map(i => X[i]), Y: idx.map(i => Y[i]) };
}

function makeDataset(type, n = 300) {
  switch (type) {
    case 'moons':    return makeMoons(n);
    case 'circles':  return makeCircles(n);
    case 'gaussian': return makeGaussian(n);
    case 'xor':      return makeXOR(n);
    case 'spiral':   return makeSpiral(n);
    default:         return makeMoons(n);
  }
}

// Train/test split (80/20)
function splitDataset(dataset, ratio = 0.8) {
  const n     = dataset.X.length;
  const split = Math.floor(n * ratio);
  return {
    train: { X: dataset.X.slice(0, split), Y: dataset.Y.slice(0, split) },
    test:  { X: dataset.X.slice(split),    Y: dataset.Y.slice(split)    },
    raw:   dataset.raw,
    means: dataset.means,
    stds:  dataset.stds,
  };
}
