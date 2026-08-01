// Circuit Room — Activation Function Explorer
// Interactive visualization of activation functions and their properties

import { useRef, useEffect, useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Info } from 'lucide-react';

const activationFunctions = [
  {
    name: 'ReLU',
    fn: (x) => Math.max(0, x),
    derivative: (x) => x > 0 ? 1 : 0,
    color: '#8b5cf6',
    formula: 'f(x) = max(0, x)',
    pros: ['No saturation for positive values', 'Computationally efficient', 'Sparse activation'],
    cons: ['Dead neurons (output=0 for x<0)', 'Not zero-centered'],
    description: 'The most popular activation. Simple but effective.'
  },
  {
    name: 'Sigmoid',
    fn: (x) => 1 / (1 + Math.exp(-x)),
    derivative: (x) => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); },
    color: '#f59e0b',
    formula: 'f(x) = 1 / (1 + e^{-x})',
    pros: ['Smooth gradient', 'Output between 0 and 1', 'Good for probabilities'],
    cons: ['Vanishing gradients', 'Not zero-centered', 'Saturates at extremes'],
    description: 'Classic activation, now mainly used in output layers for binary classification.'
  },
  {
    name: 'Tanh',
    fn: (x) => Math.tanh(x),
    derivative: (x) => 1 - Math.tanh(x) ** 2,
    color: '#10b981',
    formula: 'f(x) = tanh(x)',
    pros: ['Zero-centered', 'Stronger gradients than sigmoid'],
    cons: ['Still has vanishing gradient problem', 'Saturates at extremes'],
    description: 'Zero-centered sigmoid. Better than sigmoid for hidden layers.'
  },
  {
    name: 'GELU',
    fn: (x) => x * 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))),
    derivative: (x) => {
      const k = Math.sqrt(2 / Math.PI);
      const inner = k * (x + 0.044715 * x * x * x);
      const t = Math.tanh(inner);
      const cdf = 0.5 * (1 + t);
      const pdf = 0.5 * (1 - t * t) * k * (1 + 3 * 0.044715 * x * x);
      return cdf + x * pdf;
    },
    color: '#06b6d4',
    formula: 'f(x) = x · Φ(x)',
    pros: ['Smooth', 'Used in BERT/GPT', 'Approximates ReLU probabilistically'],
    cons: ['More expensive to compute'],
    description: 'The activation behind modern transformers. Smooth version of ReLU.'
  },
  {
    name: 'Leaky ReLU',
    fn: (x) => x > 0 ? x : 0.01 * x,
    derivative: (x) => x > 0 ? 1 : 0.01,
    color: '#f43f5e',
    formula: 'f(x) = max(0.01x, x)',
    pros: ['No dead neurons', 'Fast computation'],
    cons: ['Leak factor is arbitrary'],
    description: 'Fixes the dead neuron problem by allowing small negative values.'
  },
  {
    name: 'Swish',
    fn: (x) => x / (1 + Math.exp(-x)),
    derivative: (x) => {
      const s = 1 / (1 + Math.exp(-x));
      return s + x * s * (1 - s);
    },
    color: '#3b82f6',
    formula: 'f(x) = x · σ(x)',
    pros: ['Smooth', 'Self-gated', 'Often outperforms ReLU'],
    cons: ['More expensive than ReLU'],
    description: 'Google\'s self-gated activation. Smooth and non-monotonic.'
  },
  {
    name: 'Softplus',
    fn: (x) => Math.log(1 + Math.exp(x)),
    derivative: (x) => 1 / (1 + Math.exp(-x)),
    color: '#a855f7',
    formula: 'f(x) = ln(1 + e^x)',
    pros: ['Smooth approximation of ReLU', 'Always differentiable'],
    cons: ['Slower than ReLU', 'Not widely used'],
    description: 'A smooth version of ReLU. Derivative is the sigmoid function.'
  },
];

function ActivationGraph({ activations, showDerivatives, inputValue }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const xRange = [-5, 5];
    const yRange = [-2, 3];

    const toCanvasX = (x) => ((x - xRange[0]) / (xRange[1] - xRange[0])) * W;
    const toCanvasY = (y) => H - ((y - yRange[0]) / (yRange[1] - yRange[0])) * H;

    // Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 0.5;
    for (let x = xRange[0]; x <= xRange[1]; x++) {
      const cx = toCanvasX(x);
      ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
    }
    for (let y = yRange[0]; y <= yRange[1]; y++) {
      const cy = toCanvasY(y);
      ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    const zeroX = toCanvasX(0);
    const zeroY = toCanvasY(0);
    ctx.beginPath(); ctx.moveTo(zeroX, 0); ctx.lineTo(zeroX, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, zeroY); ctx.lineTo(W, zeroY); ctx.stroke();

    // Axis labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.font = '9px JetBrains Mono';
    ctx.textAlign = 'center';
    for (let x = xRange[0]; x <= xRange[1]; x++) {
      if (x === 0) continue;
      ctx.fillText(x.toString(), toCanvasX(x), zeroY + 14);
    }
    ctx.textAlign = 'right';
    for (let y = yRange[0]; y <= yRange[1]; y++) {
      if (y === 0) continue;
      ctx.fillText(y.toString(), zeroX - 6, toCanvasY(y) + 3);
    }

    // Draw functions
    const step = 0.02;
    for (const act of activations) {
      // Main function
      ctx.beginPath();
      ctx.strokeStyle = act.color;
      ctx.lineWidth = 2.5;
      let started = false;
      for (let x = xRange[0]; x <= xRange[1]; x += step) {
        const y = act.fn(x);
        const cx = toCanvasX(x);
        const cy = toCanvasY(y);
        if (cy >= -10 && cy <= H + 10) {
          if (!started) { ctx.moveTo(cx, cy); started = true; }
          else ctx.lineTo(cx, cy);
        }
      }
      ctx.stroke();

      // Derivative (dashed)
      if (showDerivatives) {
        ctx.beginPath();
        ctx.strokeStyle = act.color + '66';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        started = false;
        for (let x = xRange[0]; x <= xRange[1]; x += step) {
          const y = act.derivative(x);
          const cx = toCanvasX(x);
          const cy = toCanvasY(y);
          if (cy >= -10 && cy <= H + 10) {
            if (!started) { ctx.moveTo(cx, cy); started = true; }
            else ctx.lineTo(cx, cy);
          }
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // Input value indicator
    if (inputValue !== null) {
      const ix = toCanvasX(inputValue);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(ix, 0); ctx.lineTo(ix, H); ctx.stroke();
      ctx.setLineDash([]);

      // Show values on the line
      for (const act of activations) {
        const y = act.fn(inputValue);
        const cx = toCanvasX(inputValue);
        const cy = toCanvasY(y);

        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = act.color;
        ctx.fill();

        ctx.fillStyle = 'white';
        ctx.font = '9px JetBrains Mono';
        ctx.textAlign = 'left';
        ctx.fillText(`${act.name}: ${y.toFixed(3)}`, cx + 8, cy - 4);
      }
    }

  }, [activations, showDerivatives, inputValue]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full rounded-xl border border-white/5"
      style={{ height: 350 }}
    />
  );
}

export default function ActivationExplorer() {
  const [selectedFunctions, setSelectedFunctions] = useState(['ReLU', 'Sigmoid', 'GELU']);
  const [showDerivatives, setShowDerivatives] = useState(false);
  const [inputValue, setInputValue] = useState(null);
  const [hoverX, setHoverX] = useState(0);

  const activeActivations = activationFunctions.filter(a => selectedFunctions.includes(a.name));

  return (
    <div className="h-full overflow-auto p-5 space-y-5">
      {/* Function selector */}
      <div>
        <label className="text-[10px] uppercase tracking-wider text-gray-500 font-semibold">Select Functions</label>
        <div className="mt-2 flex gap-2 flex-wrap">
          {activationFunctions.map((act) => (
            <button
              key={act.name}
              onClick={() => {
                setSelectedFunctions(prev =>
                  prev.includes(act.name)
                    ? prev.filter(n => n !== act.name)
                    : [...prev, act.name]
                );
              }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${selectedFunctions.includes(act.name)
                  ? 'bg-white/10 text-white border border-white/10'
                  : 'text-gray-500 hover:text-white hover:bg-white/5 border border-transparent'
                }`}
            >
              <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: act.color }} />
              {act.name}
            </button>
          ))}
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-6">
        <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
          <input
            type="checkbox"
            checked={showDerivatives}
            onChange={(e) => setShowDerivatives(e.target.checked)}
            className="accent-purple-500"
          />
          Show Derivatives (dashed)
        </label>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Test Input:</span>
          <input
            type="range"
            min="-4"
            max="4"
            step="0.1"
            value={hoverX}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              setHoverX(v);
              setInputValue(v);
            }}
            className="w-40 accent-purple-500"
          />
          <span className="mono-value text-xs text-purple-400 w-10">{hoverX.toFixed(1)}</span>
        </div>
      </div>

      {/* Graph */}
      <ActivationGraph
        activations={activeActivations}
        showDerivatives={showDerivatives}
        inputValue={inputValue}
      />

      {/* Function details cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {activeActivations.map((act) => (
          <motion.div
            key={act.name}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-4 rounded-xl glass-light"
          >
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: act.color }} />
              <span className="text-sm font-bold text-white">{act.name}</span>
            </div>
            <div className="text-[11px] mono-value text-gray-300 mb-2 p-2 bg-black/20 rounded-lg">
              {act.formula}
            </div>
            <p className="text-[10px] text-gray-400 mb-2">{act.description}</p>

            {inputValue !== null && (
              <div className="p-2 rounded-lg bg-white/[0.03] mb-2">
                <div className="text-[10px] text-gray-500">f({inputValue.toFixed(1)}) =</div>
                <div className="text-sm font-bold mono-value" style={{ color: act.color }}>
                  {act.fn(inputValue).toFixed(4)}
                </div>
                {showDerivatives && (
                  <>
                    <div className="text-[10px] text-gray-500 mt-1">f'({inputValue.toFixed(1)}) =</div>
                    <div className="text-xs mono-value text-gray-300">
                      {act.derivative(inputValue).toFixed(4)}
                    </div>
                  </>
                )}
              </div>
            )}

            <div className="space-y-1">
              <div className="text-[9px] text-emerald-400 uppercase font-semibold">Pros</div>
              {act.pros.map((p, i) => (
                <div key={i} className="text-[10px] text-gray-400 flex items-start gap-1">
                  <span className="text-emerald-500 mt-0.5">+</span> {p}
                </div>
              ))}
              <div className="text-[9px] text-red-400 uppercase font-semibold mt-1">Cons</div>
              {act.cons.map((c, i) => (
                <div key={i} className="text-[10px] text-gray-400 flex items-start gap-1">
                  <span className="text-red-500 mt-0.5">−</span> {c}
                </div>
              ))}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
