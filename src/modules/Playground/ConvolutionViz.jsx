// Circuit Room — Convolution Visualizer
// Step-by-step kernel sliding animation over images

import { useRef, useEffect, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, SkipForward, RotateCcw } from 'lucide-react';

// Sample image data (8x8 grayscale patterns)
const sampleImages = {
  'Vertical Edge': Array.from({ length: 8 }, (_, i) =>
    Array.from({ length: 8 }, (_, j) => j < 4 ? 200 : 50)
  ),
  'Horizontal Edge': Array.from({ length: 8 }, (_, i) =>
    Array.from({ length: 8 }, (_, j) => i < 4 ? 200 : 50)
  ),
  'Diagonal': Array.from({ length: 8 }, (_, i) =>
    Array.from({ length: 8 }, (_, j) => i === j || i === j + 1 || i === j - 1 ? 255 : 30)
  ),
  'Checkerboard': Array.from({ length: 8 }, (_, i) =>
    Array.from({ length: 8 }, (_, j) => (i + j) % 2 === 0 ? 220 : 30)
  ),
  'Cross': Array.from({ length: 8 }, (_, i) =>
    Array.from({ length: 8 }, (_, j) => (i === 3 || i === 4 || j === 3 || j === 4) ? 240 : 20)
  ),
  'Gradient': Array.from({ length: 8 }, (_, i) =>
    Array.from({ length: 8 }, (_, j) => Math.round((i + j) / 14 * 255))
  ),
};

const kernels = {
  'Edge Detect (H)': { data: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]], description: 'Detects horizontal edges' },
  'Edge Detect (V)': { data: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], description: 'Detects vertical edges' },
  'Sharpen': { data: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], description: 'Enhances edges and details' },
  'Blur': { data: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]], description: 'Averages neighboring pixels' },
  'Emboss': { data: [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], description: 'Creates 3D emboss effect' },
  'Sobel X': { data: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], description: 'Gradient in X direction' },
  'Laplacian': { data: [[0, 1, 0], [1, -4, 1], [0, 1, 0]], description: 'Second derivative edge detection' },
  'Identity': { data: [[0, 0, 0], [0, 1, 0], [0, 0, 0]], description: 'Output equals input (no change)' },
};

function applyConvolution(image, kernel) {
  const kSize = kernel.length;
  const pad = Math.floor(kSize / 2);
  const h = image.length;
  const w = image[0].length;
  const output = [];

  for (let i = 0; i < h - kSize + 1; i++) {
    const row = [];
    for (let j = 0; j < w - kSize + 1; j++) {
      let sum = 0;
      for (let ki = 0; ki < kSize; ki++) {
        for (let kj = 0; kj < kSize; kj++) {
          sum += image[i + ki][j + kj] * kernel[ki][kj];
        }
      }
      row.push(sum);
    }
    output.push(row);
  }
  return output;
}

function PixelGrid({ data, cellSize = 40, highlight = null, label = '', kernelOverlay = null }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || !data.length) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rows = data.length;
    const cols = data[0].length;
    const W = cols * cellSize + 2;
    const H = rows * cellSize + 20;

    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    // Normalize values for display
    let minVal = Infinity, maxVal = -Infinity;
    for (const row of data) for (const v of row) {
      minVal = Math.min(minVal, v);
      maxVal = Math.max(maxVal, v);
    }
    const range = maxVal - minVal || 1;

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const val = data[i][j];
        const normalized = (val - minVal) / range;
        const brightness = Math.round(normalized * 255);

        const x = j * cellSize;
        const y = i * cellSize;

        ctx.fillStyle = `rgb(${brightness}, ${brightness}, ${brightness})`;
        ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);

        // Value text
        if (cellSize >= 28) {
          ctx.fillStyle = brightness > 128 ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.7)';
          ctx.font = `${Math.max(8, cellSize * 0.22)}px JetBrains Mono`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(
            Number.isInteger(val) ? val.toString() : val.toFixed(1),
            x + cellSize / 2,
            y + cellSize / 2
          );
        }

        // Grid lines
        ctx.strokeStyle = 'rgba(139, 92, 246, 0.1)';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x, y, cellSize, cellSize);
      }
    }

    // Kernel overlay highlight
    if (highlight) {
      const { row, col, kSize } = highlight;
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2;
      ctx.strokeRect(col * cellSize, row * cellSize, kSize * cellSize, kSize * cellSize);

      // Show kernel values overlaid
      if (kernelOverlay) {
        for (let ki = 0; ki < kSize; ki++) {
          for (let kj = 0; kj < kSize; kj++) {
            const x = (col + kj) * cellSize;
            const y = (row + ki) * cellSize;
            ctx.fillStyle = 'rgba(245, 158, 11, 0.3)';
            ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);

            // Kernel value
            ctx.fillStyle = '#f59e0b';
            ctx.font = `${Math.max(7, cellSize * 0.18)}px JetBrains Mono`;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'top';
            ctx.fillText(`×${kernelOverlay[ki][kj].toFixed(1)}`, x + cellSize - 3, y + 2);
          }
        }
      }
    }

    // Label
    if (label) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.font = '10px Inter';
      ctx.textAlign = 'center';
      ctx.fillText(label, W / 2, rows * cellSize + 14);
    }

  }, [data, cellSize, highlight, label, kernelOverlay]);

  return <canvas ref={canvasRef} />;
}

export default function ConvolutionViz() {
  const [selectedImage, setSelectedImage] = useState('Vertical Edge');
  const [selectedKernel, setSelectedKernel] = useState('Edge Detect (V)');
  const [currentPos, setCurrentPos] = useState({ row: 0, col: 0 });
  const [isAnimating, setIsAnimating] = useState(false);
  const [customKernel, setCustomKernel] = useState(null);
  const animRef = useRef(null);

  const image = sampleImages[selectedImage];
  const kernel = kernels[selectedKernel];
  const output = applyConvolution(image, kernel.data);
  const kSize = kernel.data.length;

  const totalSteps = (image.length - kSize + 1) * (image[0].length - kSize + 1);

  const animate = useCallback(() => {
    setIsAnimating(true);
    let step = 0;
    const cols = image[0].length - kSize + 1;

    const tick = () => {
      const row = Math.floor(step / cols);
      const col = step % cols;
      setCurrentPos({ row, col });
      step++;

      if (step < totalSteps) {
        animRef.current = setTimeout(tick, 300);
      } else {
        setIsAnimating(false);
      }
    };
    tick();
  }, [image, kSize, totalSteps]);

  const stopAnimation = () => {
    if (animRef.current) clearTimeout(animRef.current);
    setIsAnimating(false);
  };

  const stepForward = () => {
    const cols = image[0].length - kSize + 1;
    const rows = image.length - kSize + 1;
    let nextCol = currentPos.col + 1;
    let nextRow = currentPos.row;
    if (nextCol >= cols) { nextCol = 0; nextRow++; }
    if (nextRow >= rows) { nextRow = 0; nextCol = 0; }
    setCurrentPos({ row: nextRow, col: nextCol });
  };

  // Compute current convolution value
  let currentValue = 0;
  const multiplications = [];
  for (let ki = 0; ki < kSize; ki++) {
    for (let kj = 0; kj < kSize; kj++) {
      const imgVal = image[currentPos.row + ki]?.[currentPos.col + kj] || 0;
      const kerVal = kernel.data[ki][kj];
      const product = imgVal * kerVal;
      multiplications.push({ imgVal, kerVal, product });
      currentValue += product;
    }
  }

  return (
    <div className="h-full overflow-auto p-5 space-y-4">
      {/* Controls */}
      <div className="flex items-center gap-4 flex-wrap">
        <div>
          <label className="text-[10px] text-gray-500">Input Image</label>
          <select
            value={selectedImage}
            onChange={(e) => { setSelectedImage(e.target.value); setCurrentPos({ row: 0, col: 0 }); }}
            className="ml-2 bg-white/5 border border-white/10 rounded px-2 py-1 text-xs text-white"
          >
            {Object.keys(sampleImages).map(k => (
              <option key={k} value={k} className="bg-forge-900">{k}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-[10px] text-gray-500">Kernel</label>
          <select
            value={selectedKernel}
            onChange={(e) => { setSelectedKernel(e.target.value); setCurrentPos({ row: 0, col: 0 }); }}
            className="ml-2 bg-white/5 border border-white/10 rounded px-2 py-1 text-xs text-white"
          >
            {Object.keys(kernels).map(k => (
              <option key={k} value={k} className="bg-forge-900">{k}</option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          {!isAnimating ? (
            <button onClick={animate} className="px-3 py-1.5 rounded-lg bg-gradient-to-r from-blue-600 to-cyan-600 text-white text-xs font-semibold flex items-center gap-1">
              <Play className="w-3 h-3" /> Animate
            </button>
          ) : (
            <button onClick={stopAnimation} className="px-3 py-1.5 rounded-lg bg-red-500/80 text-white text-xs font-semibold flex items-center gap-1">
              <Pause className="w-3 h-3" /> Stop
            </button>
          )}
          <button onClick={stepForward} className="px-3 py-1.5 rounded-lg border border-white/10 text-gray-300 text-xs flex items-center gap-1 hover:bg-white/5">
            <SkipForward className="w-3 h-3" /> Step
          </button>
          <button onClick={() => setCurrentPos({ row: 0, col: 0 })} className="px-3 py-1.5 rounded-lg border border-white/10 text-gray-400 text-xs flex items-center gap-1 hover:bg-white/5">
            <RotateCcw className="w-3 h-3" /> Reset
          </button>
        </div>
      </div>

      {/* Visualization */}
      <div className="flex items-start gap-6 flex-wrap">
        {/* Input */}
        <div>
          <div className="text-xs font-semibold text-gray-300 mb-2">Input Image</div>
          <PixelGrid
            data={image}
            cellSize={42}
            highlight={{ row: currentPos.row, col: currentPos.col, kSize }}
            kernelOverlay={kernel.data}
            label="Input (8×8)"
          />
        </div>

        {/* Kernel */}
        <div className="flex flex-col items-center">
          <div className="text-xs font-semibold text-gray-300 mb-2">Kernel (3×3)</div>
          <PixelGrid
            data={kernel.data}
            cellSize={50}
            label={selectedKernel}
          />
          <p className="text-[10px] text-gray-500 mt-2 max-w-[160px] text-center">{kernel.description}</p>
        </div>

        {/* Operation */}
        <div className="flex flex-col items-center">
          <div className="text-xs font-semibold text-amber-400 mb-2">Current Computation</div>
          <div className="p-3 rounded-xl glass-light text-xs mono-value space-y-1 min-w-[200px]">
            {multiplications.slice(0, 6).map((m, i) => (
              <div key={i} className="flex items-center gap-1 text-[10px]">
                <span className="text-gray-400">{m.imgVal}</span>
                <span className="text-gray-600">×</span>
                <span className="text-amber-400">{m.kerVal.toFixed(1)}</span>
                <span className="text-gray-600">=</span>
                <span className={m.product >= 0 ? 'text-emerald-400' : 'text-red-400'}>{m.product.toFixed(1)}</span>
              </div>
            ))}
            {multiplications.length > 6 && <div className="text-gray-600">...</div>}
            <div className="border-t border-white/10 pt-1 mt-1">
              <span className="text-gray-400">Sum = </span>
              <span className="text-white font-bold">{currentValue.toFixed(1)}</span>
            </div>
          </div>
        </div>

        {/* Output */}
        <div>
          <div className="text-xs font-semibold text-gray-300 mb-2">Output Feature Map</div>
          <PixelGrid
            data={output}
            cellSize={42}
            highlight={{ row: currentPos.row, col: currentPos.col, kSize: 1 }}
            label={`Output (${output.length}×${output[0]?.length || 0})`}
          />
        </div>
      </div>

      {/* Educational */}
      <div className="p-4 rounded-xl bg-blue-500/5 border border-blue-500/10">
        <h4 className="text-xs font-bold text-blue-300 mb-1">🔬 How Convolution Works</h4>
        <p className="text-[10px] text-gray-400 leading-relaxed">
          The kernel slides across the input image. At each position, we multiply every kernel value with the
          overlapping pixel, sum the products, and write the result to the output. Different kernels detect
          different features: edges, textures, patterns. In a CNN, these kernels are <strong className="text-blue-300">learned from data</strong>,
          not hand-crafted.
        </p>
      </div>
    </div>
  );
}
