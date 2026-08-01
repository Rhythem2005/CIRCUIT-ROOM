// Circuit Room Neural Engine — Layer System
// Real neural network layers with forward/backward pass

import { Tensor, activations } from './tensor.js';

export class DenseLayer {
  constructor(inputSize, outputSize, activationName = 'relu') {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.activationName = activationName;
    this.activation = activations[activationName] || activations.relu;

    // He initialization for ReLU variants, Xavier for others
    const useHe = ['relu', 'leakyRelu', 'gelu', 'swish'].includes(activationName);
    this.weights = useHe
      ? Tensor.he(inputSize, outputSize)
      : Tensor.xavier(inputSize, outputSize);
    this.biases = Tensor.zeros([outputSize]);

    // Cache for backward pass
    this.input = null;
    this.preActivation = null;
    this.output = null;

    // Gradients
    this.weightsGrad = null;
    this.biasesGrad = null;

    // History for visualization
    this.weightHistory = [];
    this.gradHistory = [];
    this.activationHistory = [];
  }

  forward(input) {
    this.input = input;
    // input: (batchSize, inputSize)
    // weights: (inputSize, outputSize)
    this.preActivation = input.matmul(this.weights).add(this.biases);
    this.output = this.activation.forward(this.preActivation);
    return this.output;
  }

  backward(gradOutput) {
    // gradOutput: (batchSize, outputSize)
    const batchSize = gradOutput.shape[0];

    // Gradient through activation
    let gradPreAct;
    if (this.activationName === 'softmax') {
      // For softmax + cross-entropy, gradient is passed through directly
      gradPreAct = gradOutput;
    } else {
      const actGrad = this.activation.backward(this.preActivation);
      gradPreAct = gradOutput.mul(actGrad);
    }

    // Weight gradient: input^T x gradPreAct
    this.weightsGrad = this.input.transpose().matmul(gradPreAct);
    
    // Bias gradient: sum over batch
    this.biasesGrad = gradPreAct.sum(0);

    // Record for visualization
    const wGradMag = Math.sqrt(
      this.weightsGrad.data.reduce((s, v) => s + v * v, 0) / this.weightsGrad._size
    );
    this.gradHistory.push(wGradMag);
    if (this.gradHistory.length > 200) this.gradHistory.shift();

    // Input gradient for previous layer
    const gradInput = gradPreAct.matmul(this.weights.transpose());
    return gradInput;
  }

  getWeightStats() {
    const w = this.weights.data;
    let min = Infinity, max = -Infinity, sum = 0, sumSq = 0;
    for (let i = 0; i < w.length; i++) {
      min = Math.min(min, w[i]);
      max = Math.max(max, w[i]);
      sum += w[i];
      sumSq += w[i] * w[i];
    }
    const mean = sum / w.length;
    const std = Math.sqrt(sumSq / w.length - mean * mean);
    return { min, max, mean, std };
  }

  getWeightDistribution(bins = 30) {
    const w = this.weights.data;
    const stats = this.getWeightStats();
    const range = stats.max - stats.min || 1;
    const histogram = new Array(bins).fill(0);
    for (let i = 0; i < w.length; i++) {
      const bin = Math.min(bins - 1, Math.floor((w[i] - stats.min) / range * bins));
      histogram[bin]++;
    }
    return {
      bins: histogram.map((count, i) => ({
        x: stats.min + (i + 0.5) * range / bins,
        y: count / w.length
      })),
      stats
    };
  }

  recordSnapshot() {
    this.weightHistory.push({
      weights: this.weights.clone(),
      stats: this.getWeightStats()
    });
    if (this.weightHistory.length > 100) this.weightHistory.shift();

    if (this.output) {
      const acts = this.output.data;
      let mean = 0, activeCount = 0;
      for (let i = 0; i < acts.length; i++) {
        mean += acts[i];
        if (Math.abs(acts[i]) > 1e-6) activeCount++;
      }
      this.activationHistory.push({
        mean: mean / acts.length,
        activeRatio: activeCount / acts.length
      });
      if (this.activationHistory.length > 200) this.activationHistory.shift();
    }
  }

  paramCount() {
    return this.inputSize * this.outputSize + this.outputSize;
  }

  toConfig() {
    return {
      type: 'dense',
      inputSize: this.inputSize,
      outputSize: this.outputSize,
      activation: this.activationName,
      params: this.paramCount()
    };
  }
}

export class DropoutLayer {
  constructor(rate = 0.5) {
    this.rate = rate;
    this.mask = null;
    this.training = true;
  }

  forward(input) {
    if (!this.training || this.rate === 0) return input;
    this.mask = input.map(() => Math.random() > this.rate ? 1 / (1 - this.rate) : 0);
    return input.mul(this.mask);
  }

  backward(gradOutput) {
    if (!this.training || this.rate === 0) return gradOutput;
    return gradOutput.mul(this.mask);
  }

  paramCount() { return 0; }
  recordSnapshot() {}
  toConfig() { return { type: 'dropout', rate: this.rate }; }
}

export class BatchNormLayer {
  constructor(size, momentum = 0.1) {
    this.size = size;
    this.momentum = momentum;
    this.gamma = Tensor.ones([size]);
    this.beta = Tensor.zeros([size]);
    this.runningMean = Tensor.zeros([size]);
    this.runningVar = Tensor.ones([size]);
    this.training = true;

    // Cache
    this.input = null;
    this.normalized = null;
    this.mean = null;
    this.variance = null;
    this.gammaGrad = null;
    this.betaGrad = null;
  }

  forward(input) {
    this.input = input;
    const [batchSize, features] = input.shape;

    if (this.training) {
      this.mean = input.mean(0);
      const diff = input.sub(new Tensor(
        new Float64Array(batchSize * features).fill(0).map((_, i) => this.mean.data[i % features]),
        [batchSize, features]
      ));
      this.variance = diff.pow(2).mean(0);

      // Update running stats
      for (let i = 0; i < this.size; i++) {
        this.runningMean.data[i] = (1 - this.momentum) * this.runningMean.data[i] + this.momentum * this.mean.data[i];
        this.runningVar.data[i] = (1 - this.momentum) * this.runningVar.data[i] + this.momentum * this.variance.data[i];
      }
    } else {
      this.mean = this.runningMean;
      this.variance = this.runningVar;
    }

    const eps = 1e-5;
    const result = new Float64Array(batchSize * features);
    const normalized = new Float64Array(batchSize * features);
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < features; j++) {
        const idx = i * features + j;
        normalized[idx] = (input.data[idx] - this.mean.data[j]) / Math.sqrt(this.variance.data[j] + eps);
        result[idx] = this.gamma.data[j] * normalized[idx] + this.beta.data[j];
      }
    }
    this.normalized = new Tensor(normalized, [batchSize, features]);
    return new Tensor(result, [batchSize, features]);
  }

  backward(gradOutput) {
    const [batchSize, features] = gradOutput.shape;
    const eps = 1e-5;

    this.gammaGrad = Tensor.zeros([this.size]);
    this.betaGrad = Tensor.zeros([this.size]);
    for (let j = 0; j < features; j++) {
      for (let i = 0; i < batchSize; i++) {
        const idx = i * features + j;
        this.gammaGrad.data[j] += gradOutput.data[idx] * this.normalized.data[idx];
        this.betaGrad.data[j] += gradOutput.data[idx];
      }
    }

    const gradInput = new Float64Array(batchSize * features);
    for (let j = 0; j < features; j++) {
      const invStd = 1 / Math.sqrt(this.variance.data[j] + eps);
      for (let i = 0; i < batchSize; i++) {
        const idx = i * features + j;
        gradInput[idx] = this.gamma.data[j] * invStd * (
          gradOutput.data[idx] -
          this.betaGrad.data[j] / batchSize -
          this.normalized.data[idx] * this.gammaGrad.data[j] / batchSize
        );
      }
    }
    return new Tensor(gradInput, [batchSize, features]);
  }

  paramCount() { return 2 * this.size; }
  recordSnapshot() {}
  toConfig() { return { type: 'batchnorm', size: this.size }; }
}
