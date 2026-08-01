// Circuit Room — Training Arena (Redesigned)

import { useEffect, useRef, useCallback, useState } from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, SkipForward, RotateCcw, Activity, Zap, Settings2, Upload, TrendingDown } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip } from 'recharts';
import useStore from '../../stores/useStore';
import { CLASS_PALETTE, gradientColor } from '../../components/ui/ColorSystem.js';

function DecisionBoundaryCanvas() {
  const canvasRef = useRef(null);
  const { decisionBoundary, dataset, networkVersion } = useStore();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    ctx.clearRect(0, 0, W, H);

    if (!dataset) return;

    const points = dataset.points;
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const p of points) {
      xMin = Math.min(xMin, p[0]); xMax = Math.max(xMax, p[0]);
      yMin = Math.min(yMin, p[1]); yMax = Math.max(yMax, p[1]);
    }
    const pad = 0.3;
    xMin -= pad; xMax += pad; yMin -= pad; yMax += pad;

    // Decision boundary heatmap
    if (decisionBoundary) {
      const { grid, resolution } = decisionBoundary;
      const cellW = W / resolution, cellH = H / resolution;
      for (let i = 0; i < resolution; i++) {
        for (let j = 0; j < resolution; j++) {
          const probs = grid[i][j];
          let maxProb = 0, maxClass = 0;
          for (let c = 0; c < probs.length; c++) {
            if (probs[c] > maxProb) { maxProb = probs[c]; maxClass = c; }
          }
          const color = CLASS_PALETTE[maxClass % CLASS_PALETTE.length];
          const alpha = Math.round((0.06 + maxProb * 0.16) * 255).toString(16).padStart(2, '0');
          ctx.fillStyle = color + alpha;
          ctx.fillRect(j * cellW, i * cellH, cellW + 1, cellH + 1);
        }
      }
    }

    // Grid lines
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.03)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 8; i++) {
      ctx.beginPath(); ctx.moveTo(i * W / 8, 0); ctx.lineTo(i * W / 8, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * H / 8); ctx.lineTo(W, i * H / 8); ctx.stroke();
    }

    const toX = (x) => ((x - xMin) / (xMax - xMin)) * W;
    const toY = (y) => H - ((y - yMin) / (yMax - yMin)) * H;

    // Data points
    for (let i = 0; i < points.length; i++) {
      const [x, y] = points[i];
      const cls = dataset.rawLabels[i];
      const cx = toX(x), cy = toY(y);

      ctx.beginPath();
      ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = CLASS_PALETTE[cls % CLASS_PALETTE.length];
      ctx.fill();
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }
  }, [decisionBoundary, dataset, networkVersion]);

  return <canvas ref={canvasRef} className="w-full h-full" />;
}

function GradientFlowBar({ network }) {
  if (!network) return null;
  const gradients = network.getGradientMagnitudes();
  const maxMag = Math.max(...gradients.map(g => g.magnitude), 0.001);

  return (
    <div className="space-y-1">
      {gradients.map((g, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="text-mono text-[10px] text-text-ghost w-8">L{i + 1}</span>
          <div className="flex-1 h-[6px] rounded-full overflow-hidden" style={{ background: 'rgba(6, 7, 14, 0.6)' }}>
            <motion.div
              className="h-full rounded-full"
              animate={{ width: `${Math.min(100, (g.magnitude / maxMag) * 100)}%` }}
              transition={{ duration: 0.3 }}
              style={{ backgroundColor: gradientColor(g.magnitude, maxMag) }}
            />
          </div>
          <span className="text-mono text-[9px] text-text-ghost w-14 text-right">{g.magnitude.toExponential(1)}</span>
        </div>
      ))}
    </div>
  );
}

// Custom data input modal
function CustomDataPanel({ onClose }) {
  const [rawData, setRawData] = useState('0.5, 0.3, 0\n-0.2, 0.8, 1\n0.7, -0.1, 0\n-0.5, -0.6, 1');
  const { initNetwork } = useStore();

  const handleApply = () => {
    // Parse CSV: x, y, class
    try {
      const lines = rawData.trim().split('\n').filter(l => l.trim());
      const points = [];
      const labels = [];
      let maxClass = 0;

      for (const line of lines) {
        const parts = line.split(',').map(s => parseFloat(s.trim()));
        if (parts.length >= 3 && parts.every(isFinite)) {
          points.push([parts[0], parts[1]]);
          labels.push(Math.round(parts[2]));
          maxClass = Math.max(maxClass, Math.round(parts[2]));
        }
      }

      if (points.length < 4) { alert('Need at least 4 data points'); return; }

      const classes = maxClass + 1;
      const { Tensor } = require('../../engine/tensor.js');
      const flatPoints = points.flat();
      const oneHotLabels = labels.flatMap(l => {
        const oh = new Array(classes).fill(0);
        oh[l] = 1;
        return oh;
      });

      const customDataset = {
        data: new Tensor(new Float64Array(flatPoints), [points.length, 2]),
        labels: new Tensor(new Float64Array(oneHotLabels), [labels.length, classes]),
        points,
        rawLabels: labels,
        classes,
        name: 'Custom',
        description: 'User-provided dataset'
      };

      useStore.setState({ dataset: customDataset, datasetId: 'custom' });
      initNetwork();
      onClose();
    } catch (e) {
      alert('Error parsing data: ' + e.message);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-4 rounded-xl sf-raised space-y-3"
    >
      <div className="flex items-center justify-between">
        <div className="text-label flex items-center gap-1.5">
          <Upload className="w-3 h-3" /> Custom Data
        </div>
        <button onClick={onClose} className="text-text-ghost hover:text-text-secondary text-xs">✕</button>
      </div>
      <p className="text-[10px] text-text-tertiary">Enter data as CSV: x, y, class (0 or 1)</p>
      <textarea
        value={rawData}
        onChange={(e) => setRawData(e.target.value)}
        className="control-input h-28 text-mono text-[10px] resize-none"
        placeholder="x, y, class"
      />
      <button onClick={handleApply} className="btn-primary w-full text-xs py-2">Apply Custom Data</button>
    </motion.div>
  );
}

export default function TrainingArena() {
  const {
    network, dataset, optimizer, epoch, isTraining, trainingSpeed,
    trainingHistory, decisionBoundary, layerSizes, activationFn,
    datasetId, optimizerName, learningRate, batchSize, networkVersion,
    initNetwork, trainOneEpoch, updateDecisionBoundary,
    toggleTraining, resetTraining,
    setLearningRate, setOptimizerName, setBatchSize, setTrainingSpeed,
    setDataset, snapshotRepresentations
  } = useStore();

  const animRef = useRef(null);
  const [showCustomData, setShowCustomData] = useState(false);

  useEffect(() => { if (!network) initNetwork(); }, []);

  // Training loop
  useEffect(() => {
    if (!isTraining || !network) return;
    let fc = 0;
    const run = () => {
      for (let i = 0; i < Math.max(1, Math.floor(trainingSpeed / 2)); i++) trainOneEpoch();
      fc++;
      if (fc % 3 === 0) updateDecisionBoundary();
      if (fc % 20 === 0) snapshotRepresentations();
      animRef.current = requestAnimationFrame(run);
    };
    animRef.current = requestAnimationFrame(run);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [isTraining, trainingSpeed, network]);

  useEffect(() => {
    if (network && dataset) updateDecisionBoundary();
  }, [networkVersion]);

  const lossData = trainingHistory.loss.map((l, i) => ({
    epoch: i + 1, loss: l, accuracy: trainingHistory.accuracy[i]
  }));
  const lastLoss = trainingHistory.loss.length > 0 ? trainingHistory.loss[trainingHistory.loss.length - 1] : null;
  const lastAcc = trainingHistory.accuracy.length > 0 ? trainingHistory.accuracy[trainingHistory.accuracy.length - 1] : null;

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="module-header">
        <div className="flex items-center gap-3">
          <div className="module-icon" style={{ background: 'linear-gradient(135deg, #fbbf24, #f59e0b)' }}>
            <Activity className="w-4 h-4 text-white" />
          </div>
          <div>
            <h2 className="text-headline text-base text-text-primary">Training Arena</h2>
            <p className="text-[11px] text-text-tertiary mt-0.5">Live training with decision boundaries, gradients, and metrics</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => { trainOneEpoch(); updateDecisionBoundary(); }} disabled={isTraining}
            className="btn-ghost flex items-center gap-1 text-xs disabled:opacity-30">
            <SkipForward className="w-3.5 h-3.5" /> Step
          </button>
          <motion.button onClick={toggleTraining}
            className={`flex items-center gap-1.5 text-xs font-semibold px-4 py-2 rounded-[10px] ${
              isTraining ? 'bg-nf-rose/80 text-white' : 'btn-primary'
            }`}
            whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}>
            {isTraining ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            {isTraining ? 'Pause' : 'Train'}
          </motion.button>
          <button onClick={resetTraining} className="btn-ghost flex items-center gap-1 text-xs">
            <RotateCcw className="w-3.5 h-3.5" /> Reset
          </button>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Main Area */}
        <div className="flex-1 flex flex-col p-4 gap-3 overflow-auto">
          {/* Stats */}
          <div className="grid grid-cols-4 gap-2">
            {[
              { label: 'Epoch', value: epoch, color: 'text-text-primary' },
              { label: 'Loss', value: lastLoss !== null ? lastLoss.toFixed(4) : '—', color: 'text-nf-amber' },
              { label: 'Accuracy', value: lastAcc !== null ? (lastAcc * 100).toFixed(1) + '%' : '—', color: 'text-nf-emerald' },
              { label: 'Status', value: isTraining ? 'Training' : epoch > 0 ? 'Paused' : 'Ready', color: isTraining ? 'text-nf-emerald' : 'text-text-tertiary' },
            ].map((s, i) => (
              <div key={i} className="stat-card text-center">
                <div className={`text-base font-bold text-mono ${s.color}`}>{s.value}</div>
                <div className="text-[9px] text-text-ghost uppercase tracking-wider mt-0.5">{s.label}</div>
              </div>
            ))}
          </div>

          {/* Decision Boundary + Charts */}
          <div className="flex-1 grid grid-cols-2 gap-3 min-h-0">
            <div className="viz-container p-3 flex flex-col">
              <div className="text-xs font-semibold text-text-secondary mb-2 flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-nf-violet" /> Decision Boundary
              </div>
              <div className="flex-1 min-h-0 rounded-lg overflow-hidden">
                <DecisionBoundaryCanvas />
              </div>
            </div>

            <div className="flex flex-col gap-3">
              <div className="flex-1 viz-container p-3">
                <div className="text-xs font-semibold text-text-secondary mb-2 flex items-center gap-1.5">
                  <TrendingDown className="w-3 h-3 text-nf-amber" /> Loss
                </div>
                <div className="h-[calc(100%-24px)]">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={lossData.slice(-200)}>
                      <defs>
                        <linearGradient id="lossG" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.2} />
                          <stop offset="100%" stopColor="#f59e0b" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,92,246,0.04)" />
                      <XAxis dataKey="epoch" tick={{ fontSize: 9, fill: '#3a3d5c' }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fontSize: 9, fill: '#3a3d5c' }} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{ background: 'rgba(12,13,26,0.95)', border: '1px solid rgba(139,92,246,0.1)', borderRadius: 10, fontSize: 11, color: '#9295b3' }} />
                      <Area type="monotone" dataKey="loss" stroke="#f59e0b" fill="url(#lossG)" strokeWidth={1.5} dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="flex-1 viz-container p-3">
                <div className="text-xs font-semibold text-text-secondary mb-2 flex items-center gap-1.5">
                  <Zap className="w-3 h-3 text-nf-emerald" /> Accuracy
                </div>
                <div className="h-[calc(100%-24px)]">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={lossData.slice(-200)}>
                      <defs>
                        <linearGradient id="accG" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#10b981" stopOpacity={0.2} />
                          <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,92,246,0.04)" />
                      <XAxis dataKey="epoch" tick={{ fontSize: 9, fill: '#3a3d5c' }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fontSize: 9, fill: '#3a3d5c' }} axisLine={false} tickLine={false} domain={[0, 1]} />
                      <Tooltip contentStyle={{ background: 'rgba(12,13,26,0.95)', border: '1px solid rgba(139,92,246,0.1)', borderRadius: 10, fontSize: 11, color: '#9295b3' }}
                        formatter={(v) => [(v * 100).toFixed(1) + '%', 'Accuracy']} />
                      <Area type="monotone" dataKey="accuracy" stroke="#10b981" fill="url(#accG)" strokeWidth={1.5} dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>

          {/* Gradient Flow */}
          <div className="viz-container p-3">
            <div className="text-xs font-semibold text-text-secondary mb-2 flex items-center gap-1.5">
              <Activity className="w-3 h-3 text-nf-violet" /> Gradient Flow
              <span className="text-[10px] text-text-ghost font-normal">— magnitude per layer</span>
            </div>
            <GradientFlowBar network={network} />
          </div>
        </div>

        {/* Config Panel */}
        <div className="w-52 border-l border-white/[0.04] overflow-y-auto" style={{ background: 'var(--color-deep)' }}>
          <div className="panel-section">
            <div className="text-label flex items-center gap-1">
              <Settings2 className="w-3 h-3" /> Hyperparameters
            </div>
          </div>

          <div className="panel-section">
            <label className="text-[10px] text-text-tertiary">Optimizer</label>
            <select value={optimizerName} onChange={(e) => setOptimizerName(e.target.value)}
              className="control-select mt-1 w-full text-xs">
              <option value="adam">Adam</option>
              <option value="sgd">SGD</option>
              <option value="sgd_momentum">SGD + Momentum</option>
              <option value="rmsprop">RMSProp</option>
            </select>
          </div>

          <div className="panel-section">
            <label className="text-[10px] text-text-tertiary flex justify-between">
              <span>Learning Rate</span>
              <span className="text-mono text-nf-violet">{learningRate.toFixed(4)}</span>
            </label>
            <input type="range" min="-4" max="-0.5" step="0.1"
              value={Math.log10(learningRate)}
              onChange={(e) => setLearningRate(Math.pow(10, parseFloat(e.target.value)))}
              className="mt-1.5 w-full" />
          </div>

          <div className="panel-section">
            <label className="text-[10px] text-text-tertiary flex justify-between">
              <span>Batch Size</span>
              <span className="text-mono text-nf-violet">{batchSize}</span>
            </label>
            <input type="range" min="4" max="128" step="4" value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value))}
              className="mt-1.5 w-full" />
          </div>

          <div className="panel-section">
            <label className="text-[10px] text-text-tertiary flex justify-between">
              <span>Speed</span>
              <span className="text-mono text-nf-violet">{trainingSpeed}x</span>
            </label>
            <input type="range" min="1" max="20" step="1" value={trainingSpeed}
              onChange={(e) => setTrainingSpeed(parseInt(e.target.value))}
              className="mt-1.5 w-full" />
          </div>

          <div className="panel-section">
            <label className="text-[10px] text-text-tertiary">Dataset</label>
            <select value={datasetId} onChange={(e) => setDataset(e.target.value)}
              className="control-select mt-1 w-full text-xs">
              <option value="spiral">🌀 Spiral</option>
              <option value="circles">⭕ Circles</option>
              <option value="moons">🌙 Moons</option>
              <option value="xor">✖️ XOR</option>
              <option value="gaussian">🔵 Gaussian</option>
              <option value="linear">📏 Linear</option>
            </select>
            <button onClick={() => setShowCustomData(!showCustomData)}
              className="mt-2 w-full text-[10px] text-nf-violet hover:text-nf-cyan flex items-center justify-center gap-1 transition-colors">
              <Upload className="w-3 h-3" /> Use Custom Data
            </button>
          </div>

          {showCustomData && (
            <div className="panel-section">
              <CustomDataPanel onClose={() => setShowCustomData(false)} />
            </div>
          )}

          <div className="panel-section">
            <label className="text-[10px] text-text-tertiary">Activation</label>
            <select value={activationFn}
              onChange={(e) => { useStore.getState().setActivation(e.target.value); }}
              className="control-select mt-1 w-full text-xs">
              <option value="relu">ReLU</option>
              <option value="sigmoid">Sigmoid</option>
              <option value="tanh">Tanh</option>
              <option value="gelu">GELU</option>
              <option value="swish">Swish</option>
              <option value="leakyRelu">Leaky ReLU</option>
            </select>
          </div>

          {/* Weight stats */}
          {network && network.layers.filter(l => l.weights).length > 0 && (
            <div className="panel-section">
              <label className="text-label">Weight Stats</label>
              {network.layers.filter(l => l.weights).map((layer, i) => {
                const s = layer.getWeightStats();
                return (
                  <div key={i} className="mt-2 p-2 rounded-lg sf-deep">
                    <div className="text-mono text-[9px] text-nf-violet mb-0.5">Layer {i + 1}</div>
                    <div className="grid grid-cols-2 gap-x-2 text-[9px]">
                      <span className="text-text-ghost">μ</span>
                      <span className="text-mono text-text-secondary">{s.mean.toFixed(4)}</span>
                      <span className="text-text-ghost">σ</span>
                      <span className="text-mono text-text-secondary">{s.std.toFixed(4)}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
