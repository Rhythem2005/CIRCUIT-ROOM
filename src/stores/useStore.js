// Circuit Room — Global State Store (Zustand)

import { create } from 'zustand';
import { Network } from '../engine/network.js';
import { DenseLayer } from '../engine/layers.js';
import { createOptimizer } from '../engine/optimizers.js';
import { DATASETS } from '../engine/datasets.js';

const useStore = create((set, get) => ({
  // Navigation
  activeModule: 'home',
  setActiveModule: (module) => set({ activeModule: module }),

  // Network Configuration
  layerSizes: [2, 8, 8, 2],
  activationFn: 'relu',
  network: null,
  networkVersion: 0,

  // Dataset
  datasetId: 'spiral',
  dataset: null,
  datasetSize: 300,

  // Training State
  isTraining: false,
  trainingSpeed: 5,
  epoch: 0,
  batchSize: 32,
  optimizerName: 'adam',
  learningRate: 0.01,
  optimizer: null,
  trainingHistory: { loss: [], accuracy: [] },
  decisionBoundary: null,
  representationSnapshots: [],

  // Inspector
  selectedLayer: null,
  selectedNeuron: null,
  inspectorOpen: false,

  // Initialize network
  initNetwork: () => {
    const state = get();
    const config = {
      layerSizes: state.layerSizes,
      activationFn: state.activationFn,
    };
    const network = Network.fromConfig(config);
    const optimizer = createOptimizer(state.optimizerName, { lr: state.learningRate });
    const dataset = DATASETS[state.datasetId](state.datasetSize, undefined);

    set({
      network,
      optimizer,
      dataset,
      trainingHistory: { loss: [], accuracy: [] },
      epoch: 0,
      isTraining: false,
      decisionBoundary: null,
      representationSnapshots: [],
      networkVersion: state.networkVersion + 1,
    });
  },

  // Update layer sizes
  setLayerSizes: (sizes) => {
    set({ layerSizes: sizes });
    get().initNetwork();
  },

  addLayer: (index, size = 8) => {
    const sizes = [...get().layerSizes];
    sizes.splice(index, 0, size);
    set({ layerSizes: sizes });
    get().initNetwork();
  },

  removeLayer: (index) => {
    const sizes = [...get().layerSizes];
    if (sizes.length <= 2) return;
    sizes.splice(index, 1);
    set({ layerSizes: sizes });
    get().initNetwork();
  },

  updateLayerSize: (index, size) => {
    const sizes = [...get().layerSizes];
    sizes[index] = Math.max(1, Math.min(64, size));
    set({ layerSizes: sizes });
    get().initNetwork();
  },

  // Update hyperparameters
  setActivation: (fn) => {
    set({ activationFn: fn });
    get().initNetwork();
  },

  setDataset: (id) => {
    set({ datasetId: id });
    get().initNetwork();
  },

  setLearningRate: (lr) => {
    set({ learningRate: lr });
    const opt = get().optimizer;
    if (opt) opt.lr = lr;
  },

  setOptimizerName: (name) => {
    set({ optimizerName: name });
    const state = get();
    const optimizer = createOptimizer(name, { lr: state.learningRate });
    set({ optimizer });
  },

  setBatchSize: (bs) => set({ batchSize: bs }),
  setTrainingSpeed: (speed) => set({ trainingSpeed: speed }),

  // Training controls
  startTraining: () => set({ isTraining: true }),
  stopTraining: () => set({ isTraining: false }),
  toggleTraining: () => set(s => ({ isTraining: !s.isTraining })),

  // Perform one training epoch
  trainOneEpoch: () => {
    const { network, dataset, optimizer, batchSize } = get();
    if (!network || !dataset || !optimizer) return null;

    const result = network.train(
      dataset.data, dataset.labels, optimizer, 1, batchSize
    );

    const loss = result.loss[result.loss.length - 1];
    const accuracy = result.accuracy[result.accuracy.length - 1];

    set(state => ({
      epoch: state.epoch + 1,
      trainingHistory: {
        loss: [...state.trainingHistory.loss, loss],
        accuracy: [...state.trainingHistory.accuracy, accuracy],
      },
      networkVersion: state.networkVersion + 1,
    }));

    return { loss, accuracy };
  },

  // Update decision boundary
  updateDecisionBoundary: () => {
    const { network, dataset } = get();
    if (!network || !dataset) return;

    const allPoints = dataset.points;
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const p of allPoints) {
      xMin = Math.min(xMin, p[0]);
      xMax = Math.max(xMax, p[0]);
      yMin = Math.min(yMin, p[1]);
      yMax = Math.max(yMax, p[1]);
    }
    const pad = 0.3;
    const boundary = network.getDecisionBoundary(
      [xMin - pad, xMax + pad],
      [yMin - pad, yMax + pad],
      60
    );
    set({ decisionBoundary: boundary });
  },

  // Snapshot representations
  snapshotRepresentations: () => {
    const { network, dataset } = get();
    if (!network || !dataset) return;
    const reps = network.snapshotRepresentations(dataset.data);
    set(state => ({
      representationSnapshots: [...state.representationSnapshots, {
        epoch: state.epoch,
        layers: reps
      }]
    }));
  },

  // Reset everything
  resetTraining: () => {
    const state = get();
    if (state.network) state.network.reset();
    if (state.optimizer) state.optimizer.reset();
    set({
      epoch: 0,
      isTraining: false,
      trainingHistory: { loss: [], accuracy: [] },
      decisionBoundary: null,
      representationSnapshots: [],
      networkVersion: state.networkVersion + 1,
    });
  },

  // Inspector
  setSelectedLayer: (idx) => set({ selectedLayer: idx, inspectorOpen: true }),
  setSelectedNeuron: (layerIdx, neuronIdx) => set({
    selectedLayer: layerIdx,
    selectedNeuron: neuronIdx,
    inspectorOpen: true
  }),
  closeInspector: () => set({ inspectorOpen: false, selectedLayer: null, selectedNeuron: null }),
}));

export default useStore;
