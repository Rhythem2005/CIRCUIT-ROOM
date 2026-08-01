// Circuit Room — CNN Engine for Digit Recognition
// High-efficiency Convolutional Neural Network Engine

import { MNIST_WEIGHTS } from './mnist_weights.js';

// Multi-channel conv2d (Optimized for speed)
export function conv2dMultiChannel(inputs, kernels, bias, stride = 1) {
  // inputs: [numInputChannels][H][W]
  // kernels: [numFilters][numInputChannels][kH][kW]
  const numFilters = kernels.length;
  const numChannels = inputs.length;
  const H = inputs[0].length;
  const W = inputs[0][0].length;
  const kH = kernels[0][0].length;
  const kW = kernels[0][0][0].length;
  const outH = Math.floor((H - kH) / stride) + 1;
  const outW = Math.floor((W - kW) / stride) + 1;

  const outputs = Array(numFilters);
  for (let fi = 0; fi < numFilters; fi++) {
    const output = Array(outH);
    const b = bias ? bias[fi] : 0;
    const kernelFi = kernels[fi];

    for (let i = 0; i < outH; i++) {
      const outRow = new Float32Array(outW);
      const inRowBase = i * stride;

      for (let j = 0; j < outW; j++) {
        let sum = b;
        const inColBase = j * stride;

        for (let ci = 0; ci < numChannels; ci++) {
          const inputCi = inputs[ci];
          const kCi = kernelFi[ci];

          for (let kh = 0; kh < kH; kh++) {
            const inRow = inputCi[inRowBase + kh];
            const kRow = kCi[kh];
            for (let kw = 0; kw < kW; kw++) {
              sum += inRow[inColBase + kw] * kRow[kw];
            }
          }
        }
        outRow[j] = sum;
      }
      output[i] = outRow;
    }
    outputs[fi] = output;
  }
  return outputs;
}

// ReLU activation on feature maps
export function reluMaps(maps) {
  return maps.map(m =>
    m.map(row => {
      const out = new Float32Array(row.length);
      for (let i = 0; i < row.length; i++) {
        out[i] = row[i] > 0 ? row[i] : 0;
      }
      return out;
    })
  );
}

// Max pooling 2x2 (Optimized)
export function maxPool2d(maps, poolSize = 2) {
  return maps.map(m => {
    const H = m.length;
    const W = m[0].length;
    const outH = Math.floor(H / poolSize);
    const outW = Math.floor(W / poolSize);
    const output = Array(outH);
    
    for (let i = 0; i < outH; i++) {
      const outRow = new Float32Array(outW);
      const rowBase = i * poolSize;
      
      for (let j = 0; j < outW; j++) {
        let maxVal = -Infinity;
        const colBase = j * poolSize;
        
        for (let pi = 0; pi < poolSize; pi++) {
          const mRow = m[rowBase + pi];
          for (let pj = 0; pj < poolSize; pj++) {
            const val = mRow[colBase + pj];
            if (val > maxVal) maxVal = val;
          }
        }
        outRow[j] = maxVal;
      }
      output[i] = outRow;
    }
    return output;
  });
}

// Flatten feature maps to 1D
export function flattenMaps(maps) {
  const flat = [];
  for (const m of maps) {
    for (const row of m) {
      for (let i = 0; i < row.length; i++) flat.push(row[i]);
    }
  }
  return flat;
}

// Dense layer forward pass (Optimized & Fixed shape)
export function denseForward(input, weights, bias) {
  // input: [inputSize], weights: [outputSize][inputSize] (PyTorch format), bias: [outputSize]
  const outputSize = weights.length;
  const inputSize = input.length;
  const output = new Float32Array(outputSize);
  for (let j = 0; j < outputSize; j++) {
    let sum = bias[j];
    const wRow = weights[j];
    for (let i = 0; i < inputSize; i++) {
      sum += input[i] * wRow[i];
    }
    output[j] = sum;
  }
  return output;
}

// Softmax
export function softmax(input) {
  let max = -Infinity;
  for (let i = 0; i < input.length; i++) {
    if (input[i] > max) max = input[i];
  }
  const exp = new Float32Array(input.length);
  let sum = 0;
  for (let i = 0; i < input.length; i++) {
    const val = Math.exp(input[i] - max);
    exp[i] = val;
    sum += val;
  }
  const res = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    res[i] = exp[i] / sum;
  }
  return res;
}

// Pre-built CNN model architecture using exported PyTorch weights
export function createDigitCNN() {
  return MNIST_WEIGHTS;
}

// Full forward pass with intermediate results for visualization
export function forwardPassCNN(inputImage, model) {
  // inputImage: 28x28 normalized [0, 1] array of arrays
  const stages = [];

  // Stage 0: Input
  stages.push({
    name: 'Input',
    type: 'input',
    maps: [inputImage],
    mapSize: [28, 28],
    description: 'Raw 28×28 grayscale input'
  });

  // Stage 1: Conv1 (1→8 channels, 3×3)
  const conv1Out = conv2dMultiChannel([inputImage], model.conv1.kernels, model.conv1.bias);
  stages.push({
    name: 'Conv Layer 1',
    type: 'conv',
    maps: conv1Out,
    mapSize: [26, 26],
    numFilters: 8,
    kernelSize: '3×3',
    description: 'Eight 3×3 filters detecting edges and basic shapes'
  });

  // Stage 2: ReLU
  const relu1Out = reluMaps(conv1Out);
  stages.push({
    name: 'ReLU',
    type: 'activation',
    maps: relu1Out,
    mapSize: [26, 26],
    description: 'Zeroing negative activations — keeping only "excited" neurons'
  });

  // Stage 3: MaxPool 2×2
  const pool1Out = maxPool2d(relu1Out, 2);
  stages.push({
    name: 'Max Pool 2×2',
    type: 'pool',
    maps: pool1Out,
    mapSize: [13, 13],
    description: 'Downsampling — keeping strongest activation in each 2×2 region'
  });

  // Stage 4: Conv2 (8→16 channels, 3×3)
  const conv2Out = conv2dMultiChannel(pool1Out, model.conv2.kernels, model.conv2.bias);
  stages.push({
    name: 'Conv Layer 2',
    type: 'conv',
    maps: conv2Out,
    mapSize: [11, 11],
    numFilters: 16,
    kernelSize: '3×3',
    description: 'Sixteen filters detecting higher-level features and patterns'
  });

  // Stage 5: ReLU
  const relu2Out = reluMaps(conv2Out);
  stages.push({
    name: 'ReLU',
    type: 'activation',
    maps: relu2Out,
    mapSize: [11, 11],
    description: 'Non-linear activation — removing negative responses'
  });

  // Stage 6: MaxPool 2×2
  const pool2Out = maxPool2d(relu2Out, 2);
  stages.push({
    name: 'Max Pool 2×2',
    type: 'pool',
    maps: pool2Out,
    mapSize: [5, 5],
    description: 'Further downsampling to 5×5 spatial resolution'
  });

  // Stage 7: Flatten
  const flat = flattenMaps(pool2Out);
  stages.push({
    name: 'Flatten',
    type: 'flatten',
    values: flat,
    description: `Reshaping ${pool2Out.length}×5×5 feature maps into a ${flat.length}-D vector`
  });

  // Stage 8: Dense 1
  const dense1Out = denseForward(flat, model.dense1.weights, model.dense1.bias);
  const dense1Relu = dense1Out.map(v => v > 0 ? v : 0);
  stages.push({
    name: 'Dense Layer (32)',
    type: 'dense',
    values: Array.from(dense1Relu),
    description: 'Fully connected layer combining all spatial features'
  });

  // Stage 9: Dense 2 + Softmax
  const dense2Out = denseForward(dense1Relu, model.dense2.weights, model.dense2.bias);
  const probs = softmax(Array.from(dense2Out));

  stages.push({
    name: 'Output (Softmax)',
    type: 'output',
    values: Array.from(probs),
    description: 'Final class probabilities for digits 0–9'
  });

  return stages;
}
