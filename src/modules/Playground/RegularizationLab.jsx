// NeuroForge — Regularization Lab
// Visualize overfitting, dropout, L1/L2 regularization effects

import { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Shield, Play, Pause, RotateCcw } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { Network } from '../../engine/network.js';
import { DenseLayer } from '../../engine/layers.js';
import { Adam } from '../../engine/optimizers.js';
import { DATASETS, CLASS_COLORS } from '../../engine/datasets.js';
import { Tensor } from '../../engine/tensor.js';

function DecisionBoundaryMini({ network, dataset, width = 250, height = 250, title }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !dataset) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const points = dataset.points;
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const p of points) {
      xMin = Math.min(xMin, p[0]); xMax = Math.max(xMax, p[0]);
      yMin = Math.min(yMin, p[1]); yMax = Math.max(yMax, p[1]);
    }
    const pad = 0.4;
    xMin -= pad; xMax += pad; yMin -= pad; yMax += pad;

    // Draw decision boundary
    if (network) {
      const res = 40;
      const boundary = network.getDecisionBoundary([xMin, xMax], [yMin, yMax], res);
      const cellW = width / res;
      const cellH = height / res;

      for (let i = 0; i < res; i++) {
        for (let j = 0; j < res; j++) {
          const probs = boundary.grid[i][j];
          let maxProb = 0, maxClass = 0;
          for (let c = 0; c < probs.length; c++) {
            if (probs[c] > maxProb) { maxProb = probs[c]; maxClass = c; }
          }
          const color = CLASS_COLORS[maxClass % CLASS_COLORS.length];
          const alpha = 0.08 + maxProb * 0.15;
          ctx.fillStyle = color + Math.round(alpha * 255).toString(16).padStart(2, '0');
          ctx.fillRect(j * cellW, i * cellH, cellW + 1, cellH + 1);
        }
      }
    }

    // Draw points
    const toCanvasX = (x) => ((x - xMin) / (xMax - xMin)) * width;
    const toCanvasY = (y) => height - ((y - yMin) / (yMax - yMin)) * height;

    for (let i = 0; i < points.length; i++) {
      const [x, y] = points[i];
      const cls = dataset.rawLabels[i];
      ctx.beginPath();
      ctx.arc(toCanvasX(x), toCanvasY(y), 3, 0, Math.PI * 2);
      ctx.fillStyle = CLASS_COLORS[cls % CLASS_COLORS.length];
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    // Title
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.font = '11px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(title, width / 2, 16);

  }, [network, dataset, width, height, title]);

  return <canvas ref={canvasRef} className="rounded-xl border border-white/5" />;
}

export default function RegularizationLab() {
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [history, setHistory] = useState({ noReg: [], withReg: [] });
  const networkRef = useRef(null);
  const regNetworkRef = useRef(null);
  const datasetRef = useRef(null);
  const animRef = useRef(null);
  const [version, setVersion] = useState(0);

  const [regType, setRegType] = useState('dropout');
  const [complexity, setComplexity] = useState('high');

  const initialize = useCallback(() => {
    const ds = DATASETS.moons(150, 0.15);
    datasetRef.current = ds;

    // Overfit network (high capacity)
    const sizes = complexity === 'high' ? [2, 32, 32, 16, 2] : [2, 8, 8, 2];
    const net1 = Network.fromConfig({ layerSizes: sizes, activationFn: 'relu' });
    const net2 = Network.fromConfig({ layerSizes: sizes, activationFn: 'relu' });

    // Copy initial weights so both start the same
    for (let i = 0; i < net1.layers.length; i++) {
      if (net1.layers[i].weights) {
        net2.layers[i].weights = net1.layers[i].weights.clone();
        net2.layers[i].biases = net1.layers[i].biases.clone();
      }
    }

    networkRef.current = { net: net1, opt: new Adam(0.005) };
    regNetworkRef.current = { net: net2, opt: new Adam(0.005) };

    setHistory({ noReg: [], withReg: [] });
    setEpoch(0);
    setVersion(v => v + 1);
  }, [complexity]);

  useEffect(() => { initialize(); }, [initialize]);

  const trainStep = useCallback(() => {
    const ds = datasetRef.current;
    if (!ds || !networkRef.current || !regNetworkRef.current) return;

    // Train without regularization
    const res1 = networkRef.current.net.train(ds.data, ds.labels, networkRef.current.opt, 1, 16);

    // Train with regularization (simulate by adding noise / L2)
    const res2 = regNetworkRef.current.net.train(ds.data, ds.labels, regNetworkRef.current.opt, 1, 16);

    // Apply L2 regularization manually
    if (regType === 'l2') {
      const lambda = 0.01;
      for (const layer of regNetworkRef.current.net.layers) {
        if (layer.weights) {
          for (let i = 0; i < layer.weights._size; i++) {
            layer.weights.data[i] *= (1 - lambda);
          }
        }
      }
    }

    // Simulate dropout effect by zeroing random weights occasionally
    if (regType === 'dropout') {
      for (const layer of regNetworkRef.current.net.layers) {
        if (layer.weights) {
          for (let i = 0; i < layer.weights._size; i++) {
            if (Math.random() < 0.05) {
              layer.weights.data[i] *= 0.8;
            }
          }
        }
      }
    }

    // L1 regularization
    if (regType === 'l1') {
      const lambda = 0.005;
      for (const layer of regNetworkRef.current.net.layers) {
        if (layer.weights) {
          for (let i = 0; i < layer.weights._size; i++) {
            layer.weights.data[i] -= lambda * Math.sign(layer.weights.data[i]);
          }
        }
      }
    }

    setHistory(prev => ({
      noReg: [...prev.noReg, {
        epoch: prev.noReg.length + 1,
        loss: res1.loss[res1.loss.length - 1],
        acc: res1.accuracy[res1.accuracy.length - 1],
      }],
      withReg: [...prev.withReg, {
        epoch: prev.withReg.length + 1,
        loss: res2.loss[res2.loss.length - 1],
        acc: res2.accuracy[res2.accuracy.length - 1],
      }],
    }));

    setEpoch(e => e + 1);
    setVersion(v => v + 1);
  }, [regType]);

  useEffect(() => {
    if (!isTraining) return;
    const tick = () => {
      for (let i = 0; i < 3; i++) trainStep();
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [isTraining, trainStep]);

  // Merge history for charts
  const chartData = history.noReg.map((nr, i) => ({
    epoch: nr.epoch,
    'No Regularization': nr.loss,
    [`With ${regType.toUpperCase()}`]: history.withReg[i]?.loss || 0,
  }));

  return (
    <div className="h-full overflow-auto p-5 space-y-4">
      {/* Controls */}
      <div className="flex items-center gap-4 flex-wrap">
        <div>
          <label className="text-[10px] text-gray-500">Regularization</label>
          <select
            value={regType}
            onChange={(e) => { setRegType(e.target.value); initialize(); }}
            className="ml-2 bg-white/5 border border-white/10 rounded px-2 py-1 text-xs text-white"
          >
            <option value="dropout" className="bg-forge-900">Dropout</option>
            <option value="l2" className="bg-forge-900">L2 (Weight Decay)</option>
            <option value="l1" className="bg-forge-900">L1 (Sparsity)</option>
          </select>
        </div>
        <div>
          <label className="text-[10px] text-gray-500">Model Complexity</label>
          <select
            value={complexity}
            onChange={(e) => { setComplexity(e.target.value); initialize(); }}
            className="ml-2 bg-white/5 border border-white/10 rounded px-2 py-1 text-xs text-white"
          >
            <option value="high" className="bg-forge-900">High (overfit-prone)</option>
            <option value="low" className="bg-forge-900">Low (appropriate)</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsTraining(!isTraining)}
            className={`px-3 py-1.5 rounded-lg text-white text-xs font-semibold flex items-center gap-1 ${isTraining ? 'bg-red-500/80' : 'bg-gradient-to-r from-emerald-500 to-green-600'
              }`}
          >
            {isTraining ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
            {isTraining ? 'Pause' : 'Train Both'}
          </button>
          <button
            onClick={initialize}
            className="px-3 py-1.5 rounded-lg border border-white/10 text-gray-400 text-xs flex items-center gap-1 hover:bg-white/5"
          >
            <RotateCcw className="w-3 h-3" /> Reset
          </button>
          <span className="text-xs text-gray-500 mono-value">Epoch: {epoch}</span>
        </div>
      </div>

      {/* Side-by-side decision boundaries */}
      <div className="grid grid-cols-2 gap-4">
        <div className="rounded-2xl glass-light p-3">
          <DecisionBoundaryMini
            network={networkRef.current?.net}
            dataset={datasetRef.current}
            width={350}
            height={280}
            title="No Regularization (may overfit)"
          />
        </div>
        <div className="rounded-2xl glass-light p-3">
          <DecisionBoundaryMini
            network={regNetworkRef.current?.net}
            dataset={datasetRef.current}
            width={350}
            height={280}
            title={`With ${regType.toUpperCase()} Regularization`}
          />
        </div>
      </div>

      {/* Loss comparison chart */}
      <div className="rounded-2xl glass-light p-3" style={{ height: 200 }}>
        <div className="text-xs font-semibold text-gray-300 mb-2">Training Loss Comparison</div>
        <ResponsiveContainer width="100%" height="85%">
          <AreaChart data={chartData.slice(-150)}>
            <defs>
              <linearGradient id="noRegGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ef4444" stopOpacity={0.2} />
                <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="regGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" stopOpacity={0.2} />
                <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
            <XAxis dataKey="epoch" tick={{ fontSize: 9, fill: '#6b7280' }} axisLine={false} />
            <YAxis tick={{ fontSize: 9, fill: '#6b7280' }} axisLine={false} />
            <Tooltip contentStyle={{ background: 'rgba(15,17,34,0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 11 }} />
            <Area type="monotone" dataKey="No Regularization" stroke="#ef4444" fill="url(#noRegGrad)" strokeWidth={2} dot={false} />
            <Area type="monotone" dataKey={`With ${regType.toUpperCase()}`} stroke="#10b981" fill="url(#regGrad)" strokeWidth={2} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Educational */}
      <div className="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/10">
        <h4 className="text-xs font-bold text-emerald-300 mb-1">🛡️ Why Regularize?</h4>
        <p className="text-[10px] text-gray-400 leading-relaxed">
          Without regularization, complex models <strong className="text-red-300">memorize training data</strong> instead of learning general patterns.
          The unregularized boundary may look very jagged — it fits noise. Regularization
          encourages <strong className="text-emerald-300">smoother, simpler boundaries</strong> that generalize to unseen data.
          <strong className="text-emerald-300"> Dropout</strong> randomly disables neurons, forcing the network to be redundant.
          <strong className="text-emerald-300"> L2</strong> penalizes large weights. <strong className="text-emerald-300">L1</strong> promotes sparsity.
        </p>
      </div>
    </div>
  );
}
