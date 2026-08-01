// NeuroForge Neural Engine — Optimizers
// Real optimizers with state tracking for visualization

export class SGD {
  constructor(lr = 0.01, momentum = 0) {
    this.lr = lr;
    this.momentum = momentum;
    this.velocities = new Map();
    this.step = 0;
    this.name = 'SGD';
  }

  update(layer) {
    this.step++;
    const id = layer;

    if (this.momentum > 0) {
      if (!this.velocities.has(id)) {
        this.velocities.set(id, {
          w: new Float64Array(layer.weightsGrad._size),
          b: new Float64Array(layer.biasesGrad._size)
        });
      }
      const vel = this.velocities.get(id);

      // Update velocities
      for (let i = 0; i < layer.weightsGrad._size; i++) {
        vel.w[i] = this.momentum * vel.w[i] - this.lr * layer.weightsGrad.data[i];
        layer.weights.data[i] += vel.w[i];
      }
      for (let i = 0; i < layer.biasesGrad._size; i++) {
        vel.b[i] = this.momentum * vel.b[i] - this.lr * layer.biasesGrad.data[i];
        layer.biases.data[i] += vel.b[i];
      }
    } else {
      for (let i = 0; i < layer.weightsGrad._size; i++) {
        layer.weights.data[i] -= this.lr * layer.weightsGrad.data[i];
      }
      for (let i = 0; i < layer.biasesGrad._size; i++) {
        layer.biases.data[i] -= this.lr * layer.biasesGrad.data[i];
      }
    }
  }

  reset() {
    this.velocities = new Map();
    this.step = 0;
  }
}

export class Adam {
  constructor(lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
    this.states = new Map();
    this.step = 0;
    this.name = 'Adam';
  }

  update(layer) {
    this.step++;
    const id = layer;

    if (!this.states.has(id)) {
      this.states.set(id, {
        mW: new Float64Array(layer.weightsGrad._size),
        vW: new Float64Array(layer.weightsGrad._size),
        mB: new Float64Array(layer.biasesGrad._size),
        vB: new Float64Array(layer.biasesGrad._size),
      });
    }

    const state = this.states.get(id);
    const bc1 = 1 - Math.pow(this.beta1, this.step);
    const bc2 = 1 - Math.pow(this.beta2, this.step);

    // Weights
    for (let i = 0; i < layer.weightsGrad._size; i++) {
      const g = layer.weightsGrad.data[i];
      state.mW[i] = this.beta1 * state.mW[i] + (1 - this.beta1) * g;
      state.vW[i] = this.beta2 * state.vW[i] + (1 - this.beta2) * g * g;
      const mHat = state.mW[i] / bc1;
      const vHat = state.vW[i] / bc2;
      layer.weights.data[i] -= this.lr * mHat / (Math.sqrt(vHat) + this.epsilon);
    }

    // Biases
    for (let i = 0; i < layer.biasesGrad._size; i++) {
      const g = layer.biasesGrad.data[i];
      state.mB[i] = this.beta1 * state.mB[i] + (1 - this.beta1) * g;
      state.vB[i] = this.beta2 * state.vB[i] + (1 - this.beta2) * g * g;
      const mHat = state.mB[i] / bc1;
      const vHat = state.vB[i] / bc2;
      layer.biases.data[i] -= this.lr * mHat / (Math.sqrt(vHat) + this.epsilon);
    }
  }

  reset() {
    this.states = new Map();
    this.step = 0;
  }
}

export class RMSProp {
  constructor(lr = 0.001, decay = 0.9, epsilon = 1e-8) {
    this.lr = lr;
    this.decay = decay;
    this.epsilon = epsilon;
    this.states = new Map();
    this.step = 0;
    this.name = 'RMSProp';
  }

  update(layer) {
    this.step++;
    const id = layer;

    if (!this.states.has(id)) {
      this.states.set(id, {
        vW: new Float64Array(layer.weightsGrad._size),
        vB: new Float64Array(layer.biasesGrad._size),
      });
    }

    const state = this.states.get(id);

    for (let i = 0; i < layer.weightsGrad._size; i++) {
      const g = layer.weightsGrad.data[i];
      state.vW[i] = this.decay * state.vW[i] + (1 - this.decay) * g * g;
      layer.weights.data[i] -= this.lr * g / (Math.sqrt(state.vW[i]) + this.epsilon);
    }

    for (let i = 0; i < layer.biasesGrad._size; i++) {
      const g = layer.biasesGrad.data[i];
      state.vB[i] = this.decay * state.vB[i] + (1 - this.decay) * g * g;
      layer.biases.data[i] -= this.lr * g / (Math.sqrt(state.vB[i]) + this.epsilon);
    }
  }

  reset() {
    this.states = new Map();
    this.step = 0;
  }
}

export function createOptimizer(name, config = {}) {
  switch (name) {
    case 'sgd':
      return new SGD(config.lr || 0.01, config.momentum || 0);
    case 'sgd_momentum':
      return new SGD(config.lr || 0.01, config.momentum || 0.9);
    case 'adam':
      return new Adam(config.lr || 0.001, config.beta1 || 0.9, config.beta2 || 0.999);
    case 'rmsprop':
      return new RMSProp(config.lr || 0.001, config.decay || 0.9);
    default:
      return new Adam(config.lr || 0.001);
  }
}
