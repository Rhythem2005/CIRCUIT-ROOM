// Circuit Room Neural Engine — Tensor System
// Real tensor operations with basic autograd support

export class Tensor {
  constructor(data, shape = null, requiresGrad = false) {
    if (Array.isArray(data)) {
      this.data = this._flatten(data);
      this.shape = shape || this._inferShape(data);
    } else if (data instanceof Float64Array || data instanceof Float32Array) {
      this.data = new Float64Array(data);
      this.shape = shape || [data.length];
    } else {
      this.data = new Float64Array([data]);
      this.shape = [1];
    }
    this.requiresGrad = requiresGrad;
    this.grad = null;
    this._size = this.shape.reduce((a, b) => a * b, 1);
  }

  _flatten(arr) {
    const flat = [];
    const recurse = (a) => {
      if (Array.isArray(a)) a.forEach(recurse);
      else flat.push(a);
    };
    recurse(arr);
    return new Float64Array(flat);
  }

  _inferShape(arr) {
    const shape = [];
    let current = arr;
    while (Array.isArray(current)) {
      shape.push(current.length);
      current = current[0];
    }
    return shape;
  }

  static zeros(shape) {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float64Array(size), shape);
  }

  static ones(shape) {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size).fill(1);
    return new Tensor(data, shape);
  }

  static rand(shape) {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) data[i] = Math.random();
    return new Tensor(data, shape);
  }

  static randn(shape, mean = 0, std = 1) {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      // Box-Muller transform
      let u1 = Math.random(), u2 = Math.random();
      while (u1 === 0) u1 = Math.random();
      data[i] = mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    return new Tensor(data, shape);
  }

  // Xavier/Glorot initialization
  static xavier(fanIn, fanOut) {
    const std = Math.sqrt(2.0 / (fanIn + fanOut));
    return Tensor.randn([fanIn, fanOut], 0, std);
  }

  // He initialization
  static he(fanIn, fanOut) {
    const std = Math.sqrt(2.0 / fanIn);
    return Tensor.randn([fanIn, fanOut], 0, std);
  }

  get(indices) {
    let idx = 0;
    let stride = this._size;
    for (let i = 0; i < indices.length; i++) {
      stride /= this.shape[i];
      idx += indices[i] * stride;
    }
    return this.data[idx];
  }

  set(indices, value) {
    let idx = 0;
    let stride = this._size;
    for (let i = 0; i < indices.length; i++) {
      stride /= this.shape[i];
      idx += indices[i] * stride;
    }
    this.data[idx] = value;
  }

  reshape(newShape) {
    const size = newShape.reduce((a, b) => a * b, 1);
    if (size !== this._size) throw new Error(`Cannot reshape ${this.shape} to ${newShape}`);
    return new Tensor(new Float64Array(this.data), newShape);
  }

  transpose() {
    if (this.shape.length !== 2) throw new Error('Transpose only for 2D tensors');
    const [rows, cols] = this.shape;
    const result = new Float64Array(this._size);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result[j * rows + i] = this.data[i * cols + j];
      }
    }
    return new Tensor(result, [cols, rows]);
  }

  // Matrix multiplication: (M, K) x (K, N) -> (M, N)
  matmul(other) {
    if (this.shape.length !== 2 || other.shape.length !== 2) {
      throw new Error('matmul requires 2D tensors');
    }
    const [M, K1] = this.shape;
    const [K2, N] = other.shape;
    if (K1 !== K2) throw new Error(`matmul shape mismatch: ${this.shape} x ${other.shape}`);
    const result = new Float64Array(M * N);
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K1; k++) {
          sum += this.data[i * K1 + k] * other.data[k * N + j];
        }
        result[i * N + j] = sum;
      }
    }
    return new Tensor(result, [M, N]);
  }

  add(other) {
    if (other instanceof Tensor) {
      // Broadcasting: if other is 1D and this is 2D, broadcast along rows
      if (this.shape.length === 2 && other.shape.length === 1 && other.shape[0] === this.shape[1]) {
        const result = new Float64Array(this._size);
        const [rows, cols] = this.shape;
        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) {
            result[i * cols + j] = this.data[i * cols + j] + other.data[j];
          }
        }
        return new Tensor(result, [...this.shape]);
      }
      const result = new Float64Array(this._size);
      for (let i = 0; i < this._size; i++) result[i] = this.data[i] + other.data[i];
      return new Tensor(result, [...this.shape]);
    }
    const result = new Float64Array(this._size);
    for (let i = 0; i < this._size; i++) result[i] = this.data[i] + other;
    return new Tensor(result, [...this.shape]);
  }

  sub(other) {
    if (other instanceof Tensor) {
      const result = new Float64Array(this._size);
      for (let i = 0; i < this._size; i++) result[i] = this.data[i] - other.data[i];
      return new Tensor(result, [...this.shape]);
    }
    const result = new Float64Array(this._size);
    for (let i = 0; i < this._size; i++) result[i] = this.data[i] - other;
    return new Tensor(result, [...this.shape]);
  }

  mul(other) {
    if (other instanceof Tensor) {
      // Element-wise with broadcasting
      if (this.shape.length === 2 && other.shape.length === 2 &&
          this.shape[0] === other.shape[0] && this.shape[1] === other.shape[1]) {
        const result = new Float64Array(this._size);
        for (let i = 0; i < this._size; i++) result[i] = this.data[i] * other.data[i];
        return new Tensor(result, [...this.shape]);
      }
      const result = new Float64Array(this._size);
      for (let i = 0; i < this._size; i++) result[i] = this.data[i] * (other.data[i % other._size]);
      return new Tensor(result, [...this.shape]);
    }
    const result = new Float64Array(this._size);
    for (let i = 0; i < this._size; i++) result[i] = this.data[i] * other;
    return new Tensor(result, [...this.shape]);
  }

  div(scalar) {
    const result = new Float64Array(this._size);
    for (let i = 0; i < this._size; i++) result[i] = this.data[i] / scalar;
    return new Tensor(result, [...this.shape]);
  }

  neg() {
    return this.mul(-1);
  }

  pow(exp) {
    const result = new Float64Array(this._size);
    for (let i = 0; i < this._size; i++) result[i] = Math.pow(this.data[i], exp);
    return new Tensor(result, [...this.shape]);
  }

  sum(axis = null) {
    if (axis === null) {
      let s = 0;
      for (let i = 0; i < this._size; i++) s += this.data[i];
      return new Tensor([s], [1]);
    }
    if (axis === 0 && this.shape.length === 2) {
      const [rows, cols] = this.shape;
      const result = new Float64Array(cols);
      for (let j = 0; j < cols; j++) {
        for (let i = 0; i < rows; i++) {
          result[j] += this.data[i * cols + j];
        }
      }
      return new Tensor(result, [cols]);
    }
    if (axis === 1 && this.shape.length === 2) {
      const [rows, cols] = this.shape;
      const result = new Float64Array(rows);
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          result[i] += this.data[i * cols + j];
        }
      }
      return new Tensor(result, [rows]);
    }
    return this;
  }

  mean(axis = null) {
    if (axis === null) {
      return this.sum().div(this._size);
    }
    const s = this.sum(axis);
    return s.div(this.shape[axis]);
  }

  max(axis = null) {
    if (axis === null) {
      let m = -Infinity;
      for (let i = 0; i < this._size; i++) if (this.data[i] > m) m = this.data[i];
      return m;
    }
    if (axis === 1 && this.shape.length === 2) {
      const [rows, cols] = this.shape;
      const result = new Float64Array(rows);
      for (let i = 0; i < rows; i++) {
        let m = -Infinity;
        for (let j = 0; j < cols; j++) {
          if (this.data[i * cols + j] > m) m = this.data[i * cols + j];
        }
        result[i] = m;
      }
      return new Tensor(result, [rows]);
    }
    return this;
  }

  argmax(axis = 1) {
    if (this.shape.length === 2 && axis === 1) {
      const [rows, cols] = this.shape;
      const result = new Float64Array(rows);
      for (let i = 0; i < rows; i++) {
        let maxVal = -Infinity, maxIdx = 0;
        for (let j = 0; j < cols; j++) {
          if (this.data[i * cols + j] > maxVal) {
            maxVal = this.data[i * cols + j];
            maxIdx = j;
          }
        }
        result[i] = maxIdx;
      }
      return new Tensor(result, [rows]);
    }
    // 1D argmax
    let maxVal = -Infinity, maxIdx = 0;
    for (let i = 0; i < this._size; i++) {
      if (this.data[i] > maxVal) { maxVal = this.data[i]; maxIdx = i; }
    }
    return maxIdx;
  }

  // Apply function element-wise
  map(fn) {
    const result = new Float64Array(this._size);
    for (let i = 0; i < this._size; i++) result[i] = fn(this.data[i], i);
    return new Tensor(result, [...this.shape]);
  }

  // Get row from 2D tensor
  row(i) {
    if (this.shape.length !== 2) throw new Error('row() only for 2D');
    const cols = this.shape[1];
    return new Tensor(new Float64Array(this.data.buffer, i * cols * 8, cols), [1, cols]);
  }

  // Slice rows
  slice(start, end) {
    if (this.shape.length === 2) {
      const cols = this.shape[1];
      const rows = end - start;
      const result = new Float64Array(rows * cols);
      for (let i = 0; i < rows * cols; i++) {
        result[i] = this.data[start * cols + i];
      }
      return new Tensor(result, [rows, cols]);
    }
    return new Tensor(new Float64Array(this.data.slice(start, end)), [end - start]);
  }

  clone() {
    const t = new Tensor(new Float64Array(this.data), [...this.shape]);
    t.requiresGrad = this.requiresGrad;
    return t;
  }

  toArray() {
    if (this.shape.length === 1) return Array.from(this.data);
    if (this.shape.length === 2) {
      const [rows, cols] = this.shape;
      const result = [];
      for (let i = 0; i < rows; i++) {
        result.push(Array.from(this.data.slice(i * cols, (i + 1) * cols)));
      }
      return result;
    }
    return Array.from(this.data);
  }

  toString() {
    return `Tensor(shape=[${this.shape}], data=[${Array.from(this.data.slice(0, 6)).map(v => v.toFixed(4)).join(', ')}${this._size > 6 ? '...' : ''}])`;
  }
}

// Activation functions as standalone operations
export const activations = {
  relu: {
    forward: (x) => x.map(v => Math.max(0, v)),
    backward: (x) => x.map(v => v > 0 ? 1 : 0),
    name: 'ReLU'
  },
  sigmoid: {
    forward: (x) => x.map(v => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, v))))),
    backward: (x) => {
      const s = activations.sigmoid.forward(x);
      return s.map((v, i) => v * (1 - v));
    },
    name: 'Sigmoid'
  },
  tanh: {
    forward: (x) => x.map(v => Math.tanh(v)),
    backward: (x) => x.map(v => 1 - Math.tanh(v) ** 2),
    name: 'Tanh'
  },
  gelu: {
    forward: (x) => x.map(v => {
      const cdf = 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v * v * v)));
      return v * cdf;
    }),
    backward: (x) => x.map(v => {
      const k = Math.sqrt(2 / Math.PI);
      const inner = k * (v + 0.044715 * v * v * v);
      const tanhInner = Math.tanh(inner);
      const cdf = 0.5 * (1 + tanhInner);
      const pdf = 0.5 * (1 - tanhInner * tanhInner) * k * (1 + 3 * 0.044715 * v * v);
      return cdf + v * pdf;
    }),
    name: 'GELU'
  },
  leakyRelu: {
    forward: (x) => x.map(v => v > 0 ? v : 0.01 * v),
    backward: (x) => x.map(v => v > 0 ? 1 : 0.01),
    name: 'Leaky ReLU'
  },
  swish: {
    forward: (x) => x.map(v => v / (1 + Math.exp(-v))),
    backward: (x) => x.map(v => {
      const sig = 1 / (1 + Math.exp(-v));
      return sig + v * sig * (1 - sig);
    }),
    name: 'Swish'
  },
  softmax: {
    forward: (x) => {
      if (x.shape.length === 2) {
        const [rows, cols] = x.shape;
        const result = new Float64Array(x._size);
        for (let i = 0; i < rows; i++) {
          let maxVal = -Infinity;
          for (let j = 0; j < cols; j++) maxVal = Math.max(maxVal, x.data[i * cols + j]);
          let sum = 0;
          for (let j = 0; j < cols; j++) {
            result[i * cols + j] = Math.exp(x.data[i * cols + j] - maxVal);
            sum += result[i * cols + j];
          }
          for (let j = 0; j < cols; j++) result[i * cols + j] /= sum;
        }
        return new Tensor(result, [...x.shape]);
      }
      const maxVal = x.max();
      const exp = x.map(v => Math.exp(v - maxVal));
      let sum = 0;
      for (let i = 0; i < exp._size; i++) sum += exp.data[i];
      return exp.div(sum);
    },
    name: 'Softmax'
  },
  linear: {
    forward: (x) => x.clone(),
    backward: (x) => Tensor.ones(x.shape),
    name: 'Linear'
  }
};

// Loss functions
export const losses = {
  mse: {
    forward: (predicted, target) => {
      const diff = predicted.sub(target);
      return diff.pow(2).mean();
    },
    backward: (predicted, target) => {
      const n = predicted._size;
      return predicted.sub(target).mul(2 / n);
    },
    name: 'MSE'
  },
  crossEntropy: {
    forward: (predicted, target) => {
      // predicted: softmax output (N, C), target: one-hot or class indices
      const eps = 1e-7;
      const logPred = predicted.map(v => Math.log(Math.max(v, eps)));
      const prod = logPred.mul(target);
      return prod.sum().mul(-1).div(predicted.shape[0]);
    },
    backward: (predicted, target) => {
      const eps = 1e-7;
      const n = predicted.shape[0];
      return predicted.sub(target).div(n);
    },
    name: 'Cross-Entropy'
  },
  binaryCrossEntropy: {
    forward: (predicted, target) => {
      const eps = 1e-7;
      let sum = 0;
      for (let i = 0; i < predicted._size; i++) {
        const p = Math.max(eps, Math.min(1 - eps, predicted.data[i]));
        const t = target.data[i];
        sum += -(t * Math.log(p) + (1 - t) * Math.log(1 - p));
      }
      return new Tensor([sum / predicted._size], [1]);
    },
    backward: (predicted, target) => {
      const eps = 1e-7;
      const n = predicted._size;
      return predicted.map((v, i) => {
        const p = Math.max(eps, Math.min(1 - eps, v));
        const t = target.data[i];
        return (-t / p + (1 - t) / (1 - p)) / n;
      });
    },
    name: 'Binary Cross-Entropy'
  }
};
