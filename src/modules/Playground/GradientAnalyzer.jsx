// NeuroForge — Gradient Flow Analyzer
// Visualize vanishing/exploding gradients and solutions

import { useRef, useEffect, useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { TrendingDown, Play, RotateCcw, AlertTriangle, CheckCircle } from 'lucide-react';

function simulateGradientFlow(numLayers, activation, useSkipConnections = false, useBatchNorm = false) {
  const layers = [];
  let gradMagnitude = 1.0;

  for (let i = numLayers - 1; i >= 0; i--) {
    // Simulate activation derivative effect on gradients
    let activationFactor;
    switch (activation) {
      case 'sigmoid':
        activationFactor = 0.25; // max derivative of sigmoid is 0.25
        break;
      case 'tanh':
        activationFactor = 0.65; // typical derivative
        break;
      case 'relu':
        activationFactor = 0.5; // 50% of neurons active on average
        break;
      case 'leaky_relu':
        activationFactor = 0.55;
        break;
      case 'gelu':
        activationFactor = 0.6;
        break;
      default:
        activationFactor = 0.5;
    }

    // Weight scale factor (random initialization effects)
    const weightFactor = 0.95 + Math.random() * 0.1;

    // BatchNorm stabilizes
    if (useBatchNorm) {
      activationFactor = Math.max(0.8, activationFactor * 1.3);
    }

    gradMagnitude *= activationFactor * weightFactor;

    // Skip connections add direct gradient path
    if (useSkipConnections && i % 2 === 0) {
      gradMagnitude = Math.max(gradMagnitude, 0.7);
    }

    layers.unshift({
      layer: i + 1,
      gradient: gradMagnitude,
      activation: activationFactor,
    });
  }

  return layers;
}

function GradientFlowChart({ layers, height = 300 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !layers.length) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width;
    const H = height;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const padding = { left: 50, right: 20, top: 20, bottom: 40 };
    const plotW = W - padding.left - padding.right;
    const plotH = H - padding.top - padding.bottom;

    // Background
    ctx.fillStyle = 'rgba(15, 17, 34, 0.5)';
    ctx.fillRect(0, 0, W, H);

    // Find y range (log scale)
    const maxGrad = Math.max(...layers.map(l => l.gradient), 1);
    const minGrad = Math.min(...layers.map(l => l.gradient), 0.0001);

    const logMax = Math.log10(Math.max(maxGrad, 10));
    const logMin = Math.log10(Math.max(minGrad, 1e-15));

    const toX = (i) => padding.left + (i / (layers.length - 1)) * plotW;
    const toY = (grad) => {
      const logVal = Math.log10(Math.max(grad, 1e-15));
      const t = (logVal - logMin) / (logMax - logMin || 1);
      return padding.top + (1 - t) * plotH;
    };

    // Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 0.5;
    for (let p = Math.ceil(logMin); p <= Math.floor(logMax); p++) {
      const y = toY(Math.pow(10, p));
      ctx.beginPath(); ctx.moveTo(padding.left, y); ctx.lineTo(W - padding.right, y); ctx.stroke();
      ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.font = '9px JetBrains Mono';
      ctx.textAlign = 'right';
      ctx.fillText(`10^${p}`, padding.left - 5, y + 3);
    }

    // Healthy gradient zone
    const healthyTop = toY(2);
    const healthyBottom = toY(0.1);
    ctx.fillStyle = 'rgba(16, 185, 129, 0.05)';
    ctx.fillRect(padding.left, healthyTop, plotW, healthyBottom - healthyTop);
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.2)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(padding.left, healthyTop); ctx.lineTo(W - padding.right, healthyTop); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(padding.left, healthyBottom); ctx.lineTo(W - padding.right, healthyBottom); ctx.stroke();
    ctx.setLineDash([]);

    // Draw gradient bars
    const barWidth = Math.max(4, plotW / layers.length * 0.6);
    for (let i = 0; i < layers.length; i++) {
      const x = toX(i) - barWidth / 2;
      const y = toY(layers[i].gradient);
      const barH = toY(Math.pow(10, logMin)) - y;

      // Color based on health
      let color;
      const grad = layers[i].gradient;
      if (grad < 0.001) color = '#ef4444'; // Vanishing — red
      else if (grad > 5) color = '#f59e0b'; // Exploding — amber
      else color = '#8b5cf6'; // Healthy — purple

      // Gradient fill
      const gradient = ctx.createLinearGradient(x, y, x, y + barH);
      gradient.addColorStop(0, color);
      gradient.addColorStop(1, color + '33');
      ctx.fillStyle = gradient;
      ctx.fillRect(x, y, barWidth, barH);

      // Glow
      ctx.fillStyle = color + '33';
      ctx.fillRect(x - 2, y, barWidth + 4, 3);
    }

    // Draw line connecting tops
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.6)';
    ctx.lineWidth = 2;
    for (let i = 0; i < layers.length; i++) {
      const x = toX(i);
      const y = toY(layers[i].gradient);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // X-axis labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.font = '9px Inter';
    ctx.textAlign = 'center';
    const step = Math.max(1, Math.floor(layers.length / 15));
    for (let i = 0; i < layers.length; i += step) {
      ctx.fillText(`L${layers[i].layer}`, toX(i), H - padding.bottom + 15);
    }

    // Labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.font = '10px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Layer (backward: output → input)', W / 2, H - 5);

    ctx.save();
    ctx.translate(12, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Gradient Magnitude (log)', 0, 0);
    ctx.restore();

  }, [layers, height]);

  return <canvas ref={canvasRef} className="w-full rounded-xl border border-white/5" style={{ height }} />;
}

export default function GradientAnalyzer() {
  const [numLayers, setNumLayers] = useState(20);
  const [activation, setActivation] = useState('sigmoid');
  const [useSkip, setUseSkip] = useState(false);
  const [useBN, setUseBN] = useState(false);
  const [version, setVersion] = useState(0);

  const layers = useMemo(
    () => simulateGradientFlow(numLayers, activation, useSkip, useBN),
    [numLayers, activation, useSkip, useBN, version]
  );

  const lastGrad = layers[0]?.gradient || 0;
  const status = lastGrad < 0.001 ? 'vanishing' : lastGrad > 5 ? 'exploding' : 'healthy';

  return (
    <div className="h-full overflow-auto p-5 space-y-4">
      {/* Controls */}
      <div className="flex items-center gap-6 flex-wrap">
        <div>
          <label className="text-[10px] text-gray-500 flex justify-between w-32">
            <span>Depth</span>
            <span className="mono-value text-purple-400">{numLayers} layers</span>
          </label>
          <input
            type="range" min="3" max="50" value={numLayers}
            onChange={(e) => setNumLayers(parseInt(e.target.value))}
            className="w-32 accent-purple-500"
          />
        </div>
        <div>
          <label className="text-[10px] text-gray-500">Activation</label>
          <select
            value={activation}
            onChange={(e) => setActivation(e.target.value)}
            className="ml-2 bg-white/5 border border-white/10 rounded px-2 py-1 text-xs text-white"
          >
            <option value="sigmoid" className="bg-forge-900">Sigmoid ⚠️</option>
            <option value="tanh" className="bg-forge-900">Tanh ⚠️</option>
            <option value="relu" className="bg-forge-900">ReLU ✓</option>
            <option value="leaky_relu" className="bg-forge-900">Leaky ReLU ✓</option>
            <option value="gelu" className="bg-forge-900">GELU ✓</option>
          </select>
        </div>
        <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
          <input type="checkbox" checked={useSkip} onChange={(e) => setUseSkip(e.target.checked)} className="accent-emerald-500" />
          Skip Connections (ResNet)
        </label>
        <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
          <input type="checkbox" checked={useBN} onChange={(e) => setUseBN(e.target.checked)} className="accent-emerald-500" />
          Batch Normalization
        </label>
        <button
          onClick={() => setVersion(v => v + 1)}
          className="px-3 py-1.5 rounded-lg border border-white/10 text-gray-400 text-xs flex items-center gap-1 hover:bg-white/5"
        >
          <RotateCcw className="w-3 h-3" /> Re-simulate
        </button>
      </div>

      {/* Status */}
      <motion.div
        key={status}
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className={`p-3 rounded-xl flex items-center gap-3 ${status === 'vanishing'
            ? 'bg-red-500/10 border border-red-500/20'
            : status === 'exploding'
              ? 'bg-amber-500/10 border border-amber-500/20'
              : 'bg-emerald-500/10 border border-emerald-500/20'
          }`}
      >
        {status === 'healthy' ? (
          <CheckCircle className="w-5 h-5 text-emerald-400" />
        ) : (
          <AlertTriangle className="w-5 h-5 text-red-400" />
        )}
        <div>
          <div className={`text-xs font-bold ${status === 'vanishing' ? 'text-red-300' : status === 'exploding' ? 'text-amber-300' : 'text-emerald-300'
            }`}>
            {status === 'vanishing' ? '⚠️ Vanishing Gradients Detected!' :
              status === 'exploding' ? '⚠️ Exploding Gradients Detected!' :
                '✅ Gradients are Healthy'}
          </div>
          <div className="text-[10px] text-gray-400 mt-0.5">
            First layer gradient: <span className="mono-value">{lastGrad.toExponential(2)}</span>
            {status === 'vanishing' && ' — Early layers barely learn. Try ReLU, skip connections, or batch norm.'}
            {status === 'exploding' && ' — Gradients are too large. Try gradient clipping or lower learning rate.'}
            {status === 'healthy' && ' — Gradients flow well across all layers.'}
          </div>
        </div>
      </motion.div>

      {/* Chart */}
      <GradientFlowChart layers={layers} height={320} />

      {/* Explanation cards */}
      <div className="grid grid-cols-3 gap-3">
        <div className="p-3 rounded-xl bg-red-500/5 border border-red-500/10">
          <h4 className="text-xs font-bold text-red-300 mb-1">Vanishing Gradients</h4>
          <p className="text-[10px] text-gray-400 leading-relaxed">
            When using sigmoid/tanh, gradients get multiplied by small values (&lt;1) at each layer.
            After many layers, gradients shrink to near zero — early layers stop learning.
          </p>
        </div>
        <div className="p-3 rounded-xl bg-amber-500/5 border border-amber-500/10">
          <h4 className="text-xs font-bold text-amber-300 mb-1">Exploding Gradients</h4>
          <p className="text-[10px] text-gray-400 leading-relaxed">
            If weight initializations are too large, gradients can multiply to enormous values,
            causing unstable training with NaN losses. Solved by gradient clipping.
          </p>
        </div>
        <div className="p-3 rounded-xl bg-emerald-500/5 border border-emerald-500/10">
          <h4 className="text-xs font-bold text-emerald-300 mb-1">Solutions</h4>
          <p className="text-[10px] text-gray-400 leading-relaxed">
            <strong className="text-emerald-300">ReLU</strong> activations, <strong className="text-emerald-300">skip connections</strong> (ResNet),
            <strong className="text-emerald-300"> batch normalization</strong>, and proper <strong className="text-emerald-300">weight initialization</strong> (He/Xavier).
          </p>
        </div>
      </div>
    </div>
  );
}
