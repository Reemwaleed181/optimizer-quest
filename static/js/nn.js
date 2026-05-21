// ============================================================
//  nn.js — Real Neural Network with forward + backprop
// ============================================================

// ── Activations ─────────────────────────────────────────────
const Activations = {
  relu:    x => Math.max(0, x),
  reluD:   x => (x > 0 ? 1 : 0),
  sigmoid: x => 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, x)))),
  sigmoidD:x => { const s = 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, x)))); return s * (1 - s); },
  tanh:    x => Math.tanh(x),
  tanhD:   x => 1 - Math.tanh(x) ** 2,
};

// ── Weight init (He normal) ──────────────────────────────────
function randn() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function heInit(fanIn) {
  return randn() * Math.sqrt(2 / fanIn);
}

// ── Dense Layer ──────────────────────────────────────────────
class DenseLayer {
  constructor(inputSize, outputSize, activation = 'relu') {
    this.inputSize  = inputSize;
    this.outputSize = outputSize;
    this.activation = activation;

    // Weight matrix [outputSize x inputSize]
    this.W = Array.from({ length: outputSize }, () =>
      Array.from({ length: inputSize }, () => heInit(inputSize))
    );
    // Bias vector
    this.b = new Array(outputSize).fill(0);

    // Cached for backprop
    this.input  = null;
    this.z      = null;
    this.output = null;

    // Gradients
    this.dW = Array.from({ length: outputSize }, () => new Array(inputSize).fill(0));
    this.db = new Array(outputSize).fill(0);
  }

  forward(x) {
    this.input = x;
    const act  = Activations[this.activation];
    const actD = Activations[this.activation + 'D'];

    this.z = this.W.map((row, i) =>
      row.reduce((sum, w, j) => sum + w * x[j], 0) + this.b[i]
    );
    this.output = this.z.map(act);
    return this.output;
  }

  backward(dOut) {
    const actD = Activations[this.activation + 'D'];

    // dZ = dOut * activation'(z)
    const dZ = dOut.map((d, i) => d * actD(this.z[i]));

    // dW[i][j] = dZ[i] * input[j]
    this.dW = this.W.map((row, i) =>
      row.map((_, j) => dZ[i] * this.input[j])
    );
    this.db = [...dZ];

    // dInput[j] = sum_i(W[i][j] * dZ[i])
    const dInput = new Array(this.inputSize).fill(0);
    for (let i = 0; i < this.outputSize; i++) {
      for (let j = 0; j < this.inputSize; j++) {
        dInput[j] += this.W[i][j] * dZ[i];
      }
    }
    return dInput;
  }

  // Flatten all params for optimizer
  getParams() {
    const params = [];
    for (const row of this.W) for (const w of row) params.push(w);
    for (const b of this.b) params.push(b);
    return params;
  }

  getGrads() {
    const grads = [];
    for (const row of this.dW) for (const g of row) grads.push(g);
    for (const g of this.db) grads.push(g);
    return grads;
  }

  setParams(params, offset = 0) {
    for (let i = 0; i < this.outputSize; i++) {
      for (let j = 0; j < this.inputSize; j++) {
        this.W[i][j] = params[offset++];
      }
    }
    for (let i = 0; i < this.outputSize; i++) {
      this.b[i] = params[offset++];
    }
    return offset;
  }

  numParams() {
    return this.outputSize * this.inputSize + this.outputSize;
  }
}

// ── Neural Network ────────────────────────────────────────────
class NeuralNetwork {
  /**
   * @param {string} arch — 'small' | 'medium' | 'large'
   */
  constructor(arch = 'medium') {
    this.arch = arch;
    this.layers = this._buildLayers(arch);
  }

  _buildLayers(arch) {
    const configs = {
      small:  [[2, 4, 'relu'], [4, 1, 'sigmoid']],
      medium: [[2, 8, 'relu'], [8, 4, 'relu'], [4, 1, 'sigmoid']],
      large:  [[2, 16, 'relu'], [16, 8, 'relu'], [8, 4, 'relu'], [4, 1, 'sigmoid']],
    };
    return (configs[arch] || configs.medium).map(
      ([inp, out, act]) => new DenseLayer(inp, out, act)
    );
  }

  forward(x) {
    let out = x;
    for (const layer of this.layers) out = layer.forward(out);
    return out[0]; // scalar output
  }

  // Returns loss scalar
  backward(x, y, lossFn = 'bce') {
    const pred = this.forward(x);
    const eps  = 1e-8;
    let loss, dPred;

    if (lossFn === 'bce') {
      loss  = -(y * Math.log(pred + eps) + (1 - y) * Math.log(1 - pred + eps));
      dPred = -(y / (pred + eps) - (1 - y) / (1 - pred + eps));
    } else { // mse
      loss  = 0.5 * (pred - y) ** 2;
      dPred = pred - y;
    }

    // Backprop through layers in reverse
    let grad = [dPred];
    for (let i = this.layers.length - 1; i >= 0; i--) {
      grad = this.layers[i].backward(grad);
    }

    return loss;
  }

  // Get flat param + grad arrays
  getParams() {
    return this.layers.flatMap(l => l.getParams());
  }

  getGrads() {
    return this.layers.flatMap(l => l.getGrads());
  }

  setParams(flatParams) {
    let offset = 0;
    for (const layer of this.layers) {
      offset = layer.setParams(flatParams, offset);
    }
  }

  predict(x) {
    return this.forward(x) > 0.5 ? 1 : 0;
  }

  accuracy(X, Y) {
    let correct = 0;
    for (let i = 0; i < X.length; i++) {
      if (this.predict(X[i]) === Y[i]) correct++;
    }
    return correct / X.length;
  }

  avgLoss(X, Y, lossFn = 'bce') {
    const eps = 1e-8;
    let total = 0;
    for (let i = 0; i < X.length; i++) {
      const p = this.forward(X[i]);
      if (lossFn === 'bce') {
        total += -(Y[i] * Math.log(p + eps) + (1 - Y[i]) * Math.log(1 - p + eps));
      } else {
        total += 0.5 * (p - Y[i]) ** 2;
      }
    }
    return total / X.length;
  }

  // For boundary canvas: raw output [0,1]
  rawOutput(x) {
    return this.forward(x);
  }
}
