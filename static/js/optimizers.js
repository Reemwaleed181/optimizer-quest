// ============================================================
//  optimizers.js — SGD, Momentum, RMSprop, Adam, AdamW
//  All implement the real update rules exactly.
// ============================================================

class SGD {
  constructor(lr = 0.01) {
    this.lr = lr;
    this.name = 'SGD';
  }
  // w ← w − lr · ∇w
  step(params, grads) {
    for (let i = 0; i < params.length; i++) {
      params[i] -= this.lr * grads[i];
    }
  }
  reset() {}
}

// ─────────────────────────────────────────────────────────────
class MomentumSGD {
  constructor(lr = 0.01, mu = 0.9) {
    this.lr  = lr;
    this.mu  = mu;           // momentum coefficient β
    this.v   = null;         // velocity vector
    this.name = 'Momentum';
  }
  // v ← μv − lr·∇w
  // w ← w + v
  step(params, grads) {
    if (!this.v) this.v = new Float64Array(params.length);
    for (let i = 0; i < params.length; i++) {
      this.v[i] = this.mu * this.v[i] - this.lr * grads[i];
      params[i] += this.v[i];
    }
  }
  reset() { this.v = null; }
}

// ─────────────────────────────────────────────────────────────
class RMSprop {
  constructor(lr = 0.01, rho = 0.9, eps = 1e-8) {
    this.lr   = lr;
    this.rho  = rho;         // decay rate
    this.eps  = eps;
    this.cache = null;       // E[g²]
    this.name = 'RMSprop';
  }
  // cache ← ρ·cache + (1−ρ)·g²
  // w ← w − lr · g / √(cache + ε)
  step(params, grads) {
    if (!this.cache) this.cache = new Float64Array(params.length);
    for (let i = 0; i < params.length; i++) {
      this.cache[i] = this.rho * this.cache[i] + (1 - this.rho) * grads[i] * grads[i];
      params[i] -= this.lr * grads[i] / (Math.sqrt(this.cache[i]) + this.eps);
    }
  }
  reset() { this.cache = null; }
}

// ─────────────────────────────────────────────────────────────
class Adam {
  constructor(lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.lr    = lr;
    this.beta1 = beta1;      // 1st moment decay
    this.beta2 = beta2;      // 2nd moment decay
    this.eps   = eps;
    this.m     = null;       // 1st moment (mean)
    this.v     = null;       // 2nd moment (variance)
    this.t     = 0;          // time step
    this.name  = 'Adam';
  }
  // m ← β1·m + (1−β1)·g
  // v ← β2·v + (1−β2)·g²
  // m̂ = m / (1−β1^t),  v̂ = v / (1−β2^t)
  // w ← w − lr · m̂ / (√v̂ + ε)
  step(params, grads) {
    this.t++;
    if (!this.m) { this.m = new Float64Array(params.length); this.v = new Float64Array(params.length); }
    const bc1 = 1 - Math.pow(this.beta1, this.t);
    const bc2 = 1 - Math.pow(this.beta2, this.t);
    for (let i = 0; i < params.length; i++) {
      this.m[i] = this.beta1 * this.m[i] + (1 - this.beta1) * grads[i];
      this.v[i] = this.beta2 * this.v[i] + (1 - this.beta2) * grads[i] * grads[i];
      const mHat = this.m[i] / bc1;
      const vHat = this.v[i] / bc2;
      params[i] -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
    }
  }
  reset() { this.m = null; this.v = null; this.t = 0; }
}

// ─────────────────────────────────────────────────────────────
class AdamW {
  /**
   * AdamW decouples L2 weight decay from the gradient:
   * instead of adding λw to the gradient (which Adam scales),
   * it applies decay BEFORE the Adam step: w ← w(1 − lr·λ)
   */
  constructor(lr = 0.001, weightDecay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.lr          = lr;
    this.weightDecay = weightDecay;  // λ
    this.beta1       = beta1;
    this.beta2       = beta2;
    this.eps         = eps;
    this.m           = null;
    this.v           = null;
    this.t           = 0;
    this.name        = 'AdamW';
  }
  // w ← w · (1 − lr·λ)          ← decoupled weight decay
  // m ← β1·m + (1−β1)·g
  // v ← β2·v + (1−β2)·g²
  // w ← w − lr · m̂ / (√v̂ + ε)
  step(params, grads) {
    this.t++;
    if (!this.m) { this.m = new Float64Array(params.length); this.v = new Float64Array(params.length); }
    const bc1 = 1 - Math.pow(this.beta1, this.t);
    const bc2 = 1 - Math.pow(this.beta2, this.t);
    const decay = 1 - this.lr * this.weightDecay;
    for (let i = 0; i < params.length; i++) {
      params[i] *= decay;   // weight decay (decoupled)
      this.m[i] = this.beta1 * this.m[i] + (1 - this.beta1) * grads[i];
      this.v[i] = this.beta2 * this.v[i] + (1 - this.beta2) * grads[i] * grads[i];
      const mHat = this.m[i] / bc1;
      const vHat = this.v[i] / bc2;
      params[i] -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
    }
  }
  reset() { this.m = null; this.v = null; this.t = 0; }
}

// ─────────────────────────────────────────────────────────────
// Factory
function createOptimizer(name, lr) {
  switch (name) {
    case 'SGD':      return new SGD(lr);
    case 'Momentum': return new MomentumSGD(lr, 0.9);
    case 'RMSprop':  return new RMSprop(lr, 0.9);
    case 'Adam':     return new Adam(lr);
    case 'AdamW':    return new AdamW(lr, 0.01);
    default:         return new Adam(lr);
  }
}

// Static display info per optimizer
const OPT_INFO = {
  SGD: {
    color:   '#60a5fa',
    speed:   'Slow',
    formula: 'w ← w − α · ∇w',
    tips:    'Classic gradient descent. Simple and interpretable but sensitive to learning rate. No adaptive components.',
    bullets: ['No adaptive learning rates', 'Noisy with mini-batches', 'Can escape saddle points', 'Best baseline to compare'],
    radar:   [30, 65, 55, 80, 35],
    description: 'SGD updates parameters by moving in the opposite direction of the gradient. It processes data in small batches to add noise, which can help escape local minima.',
    hyperparameters: [
      { name: 'lr', default: '0.01', range: '0.0001–0.1', desc: 'Step size per update' },
      { name: 'momentum', default: '0', range: '0–0.99', desc: 'Velocity accumulation' }
    ],
    useCases: ['Image classification baselines', 'Simple models', 'When interpretability matters', 'Fine-tuning with small LR']
  },
  Momentum: {
    color:   '#f472b6',
    speed:   'Medium',
    formula: 'v ← β·v − α·∇w\nw ← w + v\n(β = 0.9)',
    tips:    'Accumulates velocity across steps. Dampens oscillations and accelerates in consistent directions.',
    bullets: ['Faster than plain SGD', 'Reduces gradient oscillation', 'Good for ill-conditioned problems', 'β=0.9 works well in practice'],
    radar:   [50, 75, 65, 72, 55],
    description: 'Momentum SGD adds a velocity term that accumulates past gradients, creating an "inertia" effect. Helps accelerate convergence and dampens oscillations.',
    hyperparameters: [
      { name: 'lr', default: '0.01', range: '0.0001–0.1', desc: 'Step size' },
      { name: 'momentum', default: '0.9', range: '0.5–0.999', desc: 'Velocity coefficient' }
    ],
    useCases: ['Deep CNNs', 'Object detection', 'Surgical phase recognition', 'Non-convex optimization']
  },
  RMSprop: {
    color:   '#34d399',
    speed:   'Fast',
    formula: 'E[g²] ← ρ·E[g²] + (1−ρ)·g²\nw ← w − α·g / √(E[g²]+ε)\n(ρ=0.9, ε=1e-8)',
    tips:    'Normalises gradients by a running average of squared gradients. Each parameter gets its own effective LR.',
    bullets: ['Per-parameter adaptive LR', 'Great for non-stationary losses', 'Widely used for RNNs', 'No bias correction'],
    radar:   [68, 82, 78, 70, 72],
    description: 'RMSprop adapts the learning rate for each parameter by dividing by the square root of an exponentially decaying average of squared gradients.',
    hyperparameters: [
      { name: 'lr', default: '0.001', range: '0.0001–0.01', desc: 'Step size' },
      { name: 'rho', default: '0.9', range: '0.8–0.99', desc: 'Decay rate for moving average' },
      { name: 'eps', default: '1e-8', range: '1e-8–1e-5', desc: 'Numerical stability' }
    ],
    useCases: ['RNNs and LSTMs', 'Reinforcement learning', 'Non-stationary problems', 'Speech recognition']
  },
  Adam: {
    color:   '#fbbf24',
    speed:   'Very Fast',
    formula: 'm ← β₁m + (1−β₁)g\nv ← β₂v + (1−β₂)g²\nw ← w − α·m̂/( √v̂ + ε)\n(β₁=0.9, β₂=0.999)',
    tips:    'Combines momentum and RMSprop with bias correction. Robust across many tasks; the standard deep-learning default.',
    bullets: ['Bias-corrected moments', 'Works well out of the box', 'Fast warm-up convergence', 'May generalise worse than AdamW'],
    radar:   [88, 92, 80, 75, 90],
    description: 'Adam (Adaptive Moment Estimation) combines the benefits of Momentum and RMSprop. Uses bias-corrected estimates of first and second moments.',
    hyperparameters: [
      { name: 'lr', default: '0.001', range: '0.0001–0.01', desc: 'Step size' },
      { name: 'beta1', default: '0.9', range: '0.9–0.999', desc: 'Decay for 1st moment' },
      { name: 'beta2', default: '0.999', range: '0.99–0.9999', desc: 'Decay for 2nd moment' },
      { name: 'eps', default: '1e-8', range: '1e-8–1e-5', desc: 'Numerical stability' }
    ],
    useCases: ['CNN image classification', 'GANs training', 'NLP models', 'General deep learning']
  },
  AdamW: {
    color:   '#a78bfa',
    speed:   'Very Fast',
    formula: 'w ← w·(1 − α·λ)     ← decay\nm ← β₁m + (1−β₁)g\nv ← β₂v + (1−β₂)g²\nw ← w − α·m̂/( √v̂ + ε)\n(λ=0.01 decoupled)',
    tips:    'Fixes Adam\'s L2 regularisation bug. Weight decay is applied directly to weights, not blended into the gradient.',
    bullets: ['Decoupled weight decay λ=0.01', 'Better generalisation than Adam', 'Default for transformers/LLMs', 'Stronger regularisation effect'],
    radar:   [85, 94, 88, 95, 88],
    description: 'AdamW (Adam with Weight Decay Fix) decouples weight decay from the gradient-based update. Provides proper L2 regularization that does not interfere with adaptive learning rates.',
    hyperparameters: [
      { name: 'lr', default: '0.001', range: '0.0001–0.01', desc: 'Step size' },
      { name: 'weight_decay', default: '0.01', range: '0.001–0.1', desc: 'Decay coefficient' },
      { name: 'beta1', default: '0.9', range: '0.9–0.999', desc: 'Decay for 1st moment' },
      { name: 'beta2', default: '0.999', range: '0.99–0.9999', desc: 'Decay for 2nd moment' }
    ],
    useCases: ['Transformers & LLMs', 'BERT/GPT training', 'Fine-tuning large models', 'When regularization matters']
  },
};

const OPT_LINKS = {
  SGD: 'https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py',
  Momentum: 'https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py',
  RMSprop: 'https://github.com/pytorch/pytorch/blob/main/torch/optim/rmsprop.py',
  Adam: 'https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py',
  AdamW: 'https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py'
};

const LOSS_CODE = {
  crossentropy: {
    label: 'Cross-Entropy',
    help: 'Best default for 10-class classification with raw logits.',
    link: 'https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/loss.py'
  },
  labelsmoothing: {
    label: 'Label-Smoothed Cross-Entropy',
    help: 'Regularized cross-entropy; useful when a model becomes overconfident.',
    link: 'https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/loss.py'
  },
  mse: {
    label: 'MSE on Class Probabilities',
    help: 'Educational comparison loss; usually less ideal than cross-entropy for classification.',
    link: 'https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/loss.py'
  }
};
