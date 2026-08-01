// NeuroForge Neural Engine — Network Container
// Sequential model with full forward/backward pass and history

import { Tensor, activations, losses } from './tensor.js';
import { DenseLayer } from './layers.js';

export class Network {
  constructor() {
    this.layers = [];
    this.lossFunction = losses.mse;
    this.trained = false;

    // Training history
    this.history = {
      loss: [],
      accuracy: [],
      epochs: 0,
      learningRates: [],
    };

    // Per-layer activation snapshots for representation explorer
    this.layerActivations = [];
    this.representationHistory = [];
  }

  addLayer(layer) {
    this.layers.push(layer);
    return this;
  }

  static fromConfig(config) {
    const net = new Network();
    const { layerSizes, activationFn, lossName } = config;

    for (let i = 0; i < layerSizes.length - 1; i++) {
      const isLast = i === layerSizes.length - 2;
      const actName = isLast
        ? (layerSizes[layerSizes.length - 1] > 1 ? 'softmax' : 'sigmoid')
        : (activationFn || 'relu');
      net.addLayer(new DenseLayer(layerSizes[i], layerSizes[i + 1], actName));
    }

    if (lossName && losses[lossName]) {
      net.lossFunction = losses[lossName];
    } else {
      net.lossFunction = layerSizes[layerSizes.length - 1] > 1
        ? losses.crossEntropy
        : losses.binaryCrossEntropy;
    }

    return net;
  }

  forward(input) {
    let x = input;
    this.layerActivations = [{ name: 'Input', data: x.clone() }];

    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i];
      if (layer.forward) {
        x = layer.forward(x);
        this.layerActivations.push({
          name: `Layer ${i + 1}${layer.activation ? ` (${layer.activation.name})` : ''}`,
          data: x.clone(),
          preActivation: layer.preActivation ? layer.preActivation.clone() : null,
          config: layer.toConfig ? layer.toConfig() : null
        });
      }
    }
    return x;
  }

  backward(lossGrad) {
    let grad = lossGrad;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      if (this.layers[i].backward) {
        grad = this.layers[i].backward(grad);
      }
    }
    return grad;
  }

  computeLoss(predicted, target) {
    return this.lossFunction.forward(predicted, target);
  }

  computeLossGrad(predicted, target) {
    return this.lossFunction.backward(predicted, target);
  }

  trainStep(input, target, optimizer) {
    // Forward
    const output = this.forward(input);

    // Loss
    const loss = this.computeLoss(output, target);
    const lossGrad = this.computeLossGrad(output, target);

    // Backward
    this.backward(lossGrad);

    // Update weights
    for (const layer of this.layers) {
      if (layer.weights && layer.weightsGrad) {
        optimizer.update(layer);
        layer.recordSnapshot();
      }
    }

    // Compute accuracy
    let accuracy = 0;
    if (output.shape.length === 2 && output.shape[1] > 1) {
      // Multi-class: compare argmax
      const predClasses = output.argmax(1);
      const trueClasses = target.argmax(1);
      let correct = 0;
      for (let i = 0; i < predClasses._size; i++) {
        if (predClasses.data[i] === trueClasses.data[i]) correct++;
      }
      accuracy = correct / predClasses._size;
    } else {
      // Binary
      let correct = 0;
      for (let i = 0; i < output._size; i++) {
        const pred = output.data[i] > 0.5 ? 1 : 0;
        if (pred === target.data[i]) correct++;
      }
      accuracy = correct / output._size;
    }

    return { loss: loss.data[0], accuracy, output };
  }

  train(data, labels, optimizer, epochs = 1, batchSize = 32, callbacks = {}) {
    const n = data.shape[0];

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;
      let epochAcc = 0;
      let batches = 0;

      // Shuffle indices
      const indices = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      for (let i = 0; i < n; i += batchSize) {
        const end = Math.min(i + batchSize, n);
        const batchIndices = indices.slice(i, end);

        // Create batch
        const batchSize_ = batchIndices.length;
        const batchData = new Float64Array(batchSize_ * data.shape[1]);
        const batchLabels = new Float64Array(batchSize_ * labels.shape[1]);

        for (let b = 0; b < batchSize_; b++) {
          const idx = batchIndices[b];
          for (let j = 0; j < data.shape[1]; j++) {
            batchData[b * data.shape[1] + j] = data.data[idx * data.shape[1] + j];
          }
          for (let j = 0; j < labels.shape[1]; j++) {
            batchLabels[b * labels.shape[1] + j] = labels.data[idx * labels.shape[1] + j];
          }
        }

        const batchX = new Tensor(batchData, [batchSize_, data.shape[1]]);
        const batchY = new Tensor(batchLabels, [batchSize_, labels.shape[1]]);

        const result = this.trainStep(batchX, batchY, optimizer);
        epochLoss += result.loss;
        epochAcc += result.accuracy;
        batches++;
      }

      epochLoss /= batches;
      epochAcc /= batches;

      this.history.loss.push(epochLoss);
      this.history.accuracy.push(epochAcc);
      this.history.epochs++;
      this.history.learningRates.push(optimizer.lr);

      if (callbacks.onEpoch) {
        callbacks.onEpoch({
          epoch: this.history.epochs,
          loss: epochLoss,
          accuracy: epochAcc,
        });
      }
    }

    this.trained = true;
    return this.history;
  }

  predict(input) {
    // Set layers to eval mode
    for (const layer of this.layers) {
      if ('training' in layer) layer.training = false;
    }
    const output = this.forward(input);
    // Reset to training mode
    for (const layer of this.layers) {
      if ('training' in layer) layer.training = true;
    }
    return output;
  }

  // Get decision boundary for 2D inputs
  getDecisionBoundary(xRange, yRange, resolution = 50) {
    const points = [];
    const dx = (xRange[1] - xRange[0]) / resolution;
    const dy = (yRange[1] - yRange[0]) / resolution;

    const inputData = new Float64Array(resolution * resolution * 2);
    let idx = 0;
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        inputData[idx++] = xRange[0] + j * dx;
        inputData[idx++] = yRange[0] + i * dy;
      }
    }

    const input = new Tensor(inputData, [resolution * resolution, 2]);

    // Set eval mode
    for (const layer of this.layers) {
      if ('training' in layer) layer.training = false;
    }

    const output = this.forward(input);

    // Reset training mode
    for (const layer of this.layers) {
      if ('training' in layer) layer.training = true;
    }

    const grid = [];
    for (let i = 0; i < resolution; i++) {
      const row = [];
      for (let j = 0; j < resolution; j++) {
        const flatIdx = i * resolution + j;
        if (output.shape[1] > 1) {
          // Multi-class: get class probabilities
          const probs = [];
          for (let c = 0; c < output.shape[1]; c++) {
            probs.push(output.data[flatIdx * output.shape[1] + c]);
          }
          row.push(probs);
        } else {
          row.push([output.data[flatIdx], 1 - output.data[flatIdx]]);
        }
      }
      grid.push(row);
    }

    return { grid, xRange, yRange, resolution };
  }

  // Snapshot representations for all layers
  snapshotRepresentations(data) {
    this.predict(data);
    return this.layerActivations.map(la => ({
      name: la.name,
      data: la.data.toArray(),
      shape: la.data.shape
    }));
  }

  getGradientMagnitudes() {
    return this.layers
      .filter(l => l.weightsGrad)
      .map((l, i) => {
        const g = l.weightsGrad.data;
        let sum = 0;
        for (let j = 0; j < g.length; j++) sum += g[j] * g[j];
        return {
          layer: i,
          name: `Layer ${i + 1}`,
          magnitude: Math.sqrt(sum / g.length),
          maxGrad: Math.max(...Array.from(g).map(Math.abs))
        };
      });
  }

  totalParams() {
    return this.layers.reduce((s, l) => s + (l.paramCount ? l.paramCount() : 0), 0);
  }

  getConfig() {
    return {
      layers: this.layers.map(l => l.toConfig ? l.toConfig() : {}),
      totalParams: this.totalParams(),
      loss: this.lossFunction.name
    };
  }

  reset() {
    // Reinitialize all weights
    for (const layer of this.layers) {
      if (layer instanceof DenseLayer) {
        const useHe = ['relu', 'leakyRelu', 'gelu', 'swish'].includes(layer.activationName);
        layer.weights = useHe
          ? Tensor.he(layer.inputSize, layer.outputSize)
          : Tensor.xavier(layer.inputSize, layer.outputSize);
        layer.biases = Tensor.zeros([layer.outputSize]);
        layer.weightHistory = [];
        layer.gradHistory = [];
        layer.activationHistory = [];
      }
    }
    this.history = { loss: [], accuracy: [], epochs: 0, learningRates: [] };
    this.trained = false;
    this.representationHistory = [];
  }
}
