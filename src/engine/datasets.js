// Circuit Room Neural Engine — Datasets
// Procedurally generated 2D classification datasets

import { Tensor } from './tensor.js';

export function generateSpiral(n = 200, classes = 3, noise = 0.15) {
  const points = [];
  const labels = [];
  const perClass = Math.floor(n / classes);

  for (let c = 0; c < classes; c++) {
    for (let i = 0; i < perClass; i++) {
      const r = (i / perClass) * 1.0;
      const t = (c * 4 + i * 4 / perClass) + (Math.random() * noise);
      const x = r * Math.sin(t * Math.PI);
      const y = r * Math.cos(t * Math.PI);
      points.push([x, y]);
      const label = new Array(classes).fill(0);
      label[c] = 1;
      labels.push(label);
    }
  }

  return {
    data: new Tensor(points.flat(), [points.length, 2]),
    labels: new Tensor(labels.flat(), [labels.length, classes]),
    points,
    rawLabels: labels.map(l => l.indexOf(1)),
    classes,
    name: 'Spiral',
    description: 'Interleaved spiral arms — requires nonlinear boundaries'
  };
}

export function generateCircles(n = 200, noise = 0.05) {
  const points = [];
  const labels = [];
  const half = Math.floor(n / 2);

  for (let i = 0; i < half; i++) {
    const angle = (i / half) * 2 * Math.PI;
    const r = 0.35 + (Math.random() - 0.5) * noise;
    points.push([r * Math.cos(angle), r * Math.sin(angle)]);
    labels.push([1, 0]);
  }

  for (let i = 0; i < half; i++) {
    const angle = (i / half) * 2 * Math.PI;
    const r = 0.8 + (Math.random() - 0.5) * noise;
    points.push([r * Math.cos(angle), r * Math.sin(angle)]);
    labels.push([0, 1]);
  }

  return {
    data: new Tensor(points.flat(), [points.length, 2]),
    labels: new Tensor(labels.flat(), [labels.length, 2]),
    points,
    rawLabels: labels.map(l => l.indexOf(1)),
    classes: 2,
    name: 'Circles',
    description: 'Concentric circles — demonstrates radial decision boundaries'
  };
}

export function generateMoons(n = 200, noise = 0.1) {
  const points = [];
  const labels = [];
  const half = Math.floor(n / 2);

  for (let i = 0; i < half; i++) {
    const angle = (i / half) * Math.PI;
    const x = Math.cos(angle) + (Math.random() - 0.5) * noise;
    const y = Math.sin(angle) + (Math.random() - 0.5) * noise;
    points.push([x, y]);
    labels.push([1, 0]);
  }

  for (let i = 0; i < half; i++) {
    const angle = (i / half) * Math.PI;
    const x = 1 - Math.cos(angle) + (Math.random() - 0.5) * noise;
    const y = 1 - Math.sin(angle) - 0.5 + (Math.random() - 0.5) * noise;
    points.push([x, y]);
    labels.push([0, 1]);
  }

  return {
    data: new Tensor(points.flat(), [points.length, 2]),
    labels: new Tensor(labels.flat(), [labels.length, 2]),
    points,
    rawLabels: labels.map(l => l.indexOf(1)),
    classes: 2,
    name: 'Moons',
    description: 'Two interleaving half circles — classic nonlinear classification'
  };
}

export function generateXOR(n = 200, noise = 0.15) {
  const points = [];
  const labels = [];

  for (let i = 0; i < n; i++) {
    const x = (Math.random() - 0.5) * 2;
    const y = (Math.random() - 0.5) * 2;
    const cls = (x * y > 0) ? 0 : 1;
    points.push([x + (Math.random() - 0.5) * noise, y + (Math.random() - 0.5) * noise]);
    labels.push(cls === 0 ? [1, 0] : [0, 1]);
  }

  return {
    data: new Tensor(points.flat(), [points.length, 2]),
    labels: new Tensor(labels.flat(), [labels.length, 2]),
    points,
    rawLabels: labels.map(l => l.indexOf(1)),
    classes: 2,
    name: 'XOR',
    description: 'XOR pattern — impossible for linear classifiers'
  };
}

export function generateGaussian(n = 200, clusters = 4, spread = 0.15) {
  const points = [];
  const labels = [];
  const perCluster = Math.floor(n / clusters);

  const centers = [];
  for (let c = 0; c < clusters; c++) {
    const angle = (c / clusters) * 2 * Math.PI;
    centers.push([0.5 * Math.cos(angle), 0.5 * Math.sin(angle)]);
  }

  for (let c = 0; c < clusters; c++) {
    for (let i = 0; i < perCluster; i++) {
      const x = centers[c][0] + (Math.random() - 0.5) * spread * 2;
      const y = centers[c][1] + (Math.random() - 0.5) * spread * 2;
      points.push([x, y]);
      const label = new Array(clusters).fill(0);
      label[c] = 1;
      labels.push(label);
    }
  }

  return {
    data: new Tensor(points.flat(), [points.length, 2]),
    labels: new Tensor(labels.flat(), [labels.length, clusters]),
    points,
    rawLabels: labels.map(l => l.indexOf(1)),
    classes: clusters,
    name: 'Gaussian Clusters',
    description: 'Gaussian blobs — baseline for testing network capacity'
  };
}

// Linear dataset - should be solvable by a single neuron
export function generateLinear(n = 200, noise = 0.1) {
  const points = [];
  const labels = [];

  for (let i = 0; i < n; i++) {
    const x = (Math.random() - 0.5) * 2;
    const y = (Math.random() - 0.5) * 2;
    const cls = (x + y > 0) ? 0 : 1;
    points.push([x + (Math.random() - 0.5) * noise, y + (Math.random() - 0.5) * noise]);
    labels.push(cls === 0 ? [1, 0] : [0, 1]);
  }

  return {
    data: new Tensor(points.flat(), [points.length, 2]),
    labels: new Tensor(labels.flat(), [labels.length, 2]),
    points,
    rawLabels: labels.map(l => l.indexOf(1)),
    classes: 2,
    name: 'Linear',
    description: 'Linearly separable — even a single neuron can solve this'
  };
}

export const DATASETS = {
  spiral: generateSpiral,
  circles: generateCircles,
  moons: generateMoons,
  xor: generateXOR,
  gaussian: generateGaussian,
  linear: generateLinear,
};

export const DATASET_LIST = [
  { id: 'spiral', name: 'Spiral', icon: '🌀', difficulty: 'Hard' },
  { id: 'circles', name: 'Circles', icon: '⭕', difficulty: 'Medium' },
  { id: 'moons', name: 'Moons', icon: '🌙', difficulty: 'Medium' },
  { id: 'xor', name: 'XOR', icon: '✖️', difficulty: 'Medium' },
  { id: 'gaussian', name: 'Gaussian', icon: '🔵', difficulty: 'Easy' },
  { id: 'linear', name: 'Linear', icon: '📏', difficulty: 'Easy' },
];

export const CLASS_COLORS = [
  '#8b5cf6', // Purple
  '#f59e0b', // Amber
  '#10b981', // Emerald
  '#f43f5e', // Rose
  '#06b6d4', // Cyan
  '#3b82f6', // Blue
  '#ef4444', // Red
  '#22c55e', // Green
];

export const CLASS_COLORS_LIGHT = [
  'rgba(139, 92, 246, 0.15)',
  'rgba(245, 158, 11, 0.15)',
  'rgba(16, 185, 129, 0.15)',
  'rgba(244, 63, 94, 0.15)',
  'rgba(6, 182, 212, 0.15)',
  'rgba(59, 130, 246, 0.15)',
  'rgba(239, 68, 68, 0.15)',
  'rgba(34, 197, 94, 0.15)',
];
