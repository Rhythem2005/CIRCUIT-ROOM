import { useRef, useState, useEffect, useCallback, useMemo } from "react";
import { MNIST_WEIGHTS } from '../../engine/mnist_weights.js';

/* =========================================================================
   NeuroForge — CNN Digit Recognition (standalone educational build)
   Draw a digit, watch a real (tiny, hand-initialized) CNN run forward,
   and read exactly what each layer is doing and why, at each step.
   No external deps beyond React — canvas2d only, so this drops straight
   into any artifact/sandbox. Port the *pattern* back into your r3f version.
   ========================================================================= */

/* ---------- procedural generation removed — using real weights ---------- */
/* ---------- forward-pass primitives (same math as production engine) ---------- */
function conv2dMultiChannel(inputs, kernels, bias, stride = 1) {
  const numFilters = kernels.length;
  const numChannels = inputs.length;
  const H = inputs[0].length, W = inputs[0][0].length;
  const kH = kernels[0][0].length, kW = kernels[0][0][0].length;
  const outH = Math.floor((H - kH) / stride) + 1;
  const outW = Math.floor((W - kW) / stride) + 1;
  const outputs = Array(numFilters);
  for (let fi = 0; fi < numFilters; fi++) {
    const out = Array(outH);
    const b = bias ? bias[fi] : 0;
    const kFi = kernels[fi];
    for (let i = 0; i < outH; i++) {
      const row = new Float32Array(outW);
      for (let j = 0; j < outW; j++) {
        let sum = b;
        for (let ci = 0; ci < numChannels; ci++) {
          const inCi = inputs[ci];
          const kCi = kFi[ci];
          for (let kh = 0; kh < kH; kh++) {
            for (let kw = 0; kw < kW; kw++) sum += inCi[i + kh][j + kw] * kCi[kh][kw];
          }
        }
        row[j] = sum;
      }
      out[i] = row;
    }
    outputs[fi] = out;
  }
  return outputs;
}
function reluMaps(maps) {
  return maps.map((m) => m.map((row) => Float32Array.from(row, (v) => (v > 0 ? v : 0))));
}
function maxPool2d(maps, poolSize = 2) {
  return maps.map((m) => {
    const H = m.length, W = m[0].length;
    const outH = Math.floor(H / poolSize), outW = Math.floor(W / poolSize);
    const out = Array(outH);
    for (let i = 0; i < outH; i++) {
      const row = new Float32Array(outW);
      for (let j = 0; j < outW; j++) {
        let mx = -Infinity;
        for (let pi = 0; pi < poolSize; pi++) {
          for (let pj = 0; pj < poolSize; pj++) {
            const v = m[i * poolSize + pi][j * poolSize + pj];
            if (v > mx) mx = v;
          }
        }
        row[j] = mx;
      }
      out[i] = row;
    }
    return out;
  });
}
function flattenMaps(maps) {
  const flat = [];
  for (const m of maps) for (const row of m) for (let i = 0; i < row.length; i++) flat.push(row[i]);
  return flat;
}
function denseForward(input, weights, bias) {
  // weights expects [inSize][outSize] from Python export
  const inSize = input.length;
  const outSize = weights[0].length;
  const out = new Float32Array(outSize);
  for (let j = 0; j < outSize; j++) {
    let sum = bias[j];
    for (let i = 0; i < inSize; i++) sum += input[i] * weights[i][j];
    out[j] = sum;
  }
  return out;
}
function softmax(input) {
  let max = -Infinity;
  for (const v of input) if (v > max) max = v;
  const exp = Float32Array.from(input, (v) => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return Float32Array.from(exp, (v) => v / sum);
}

const STAGE_META = {
  input: {
    color: "#64748b",
    tag: "Input",
    headline: "A 28×28 grid of numbers",
    body:
      "Your drawing gets shrunk to a 28×28 grid, and every pixel is turned into a single number between 0 (black) and 1 (white). The network never sees a 'picture' the way you do — it only ever sees this grid of numbers, which is exactly the format MNIST was trained on.",
  },
  conv: {
    color: "#8b5cf6",
    tag: "Convolution",
    headline: "Sliding small pattern-detectors over the image",
    body:
      "Each filter is a tiny 3×3 grid of numbers (a kernel). It slides across every position in the image, multiplying its 9 numbers against the 9 pixels underneath and summing the result. Where the patch under the filter matches the filter's pattern, the sum spikes high; where it doesn't, the sum stays low or goes negative. Different filters are shaped to fire on different things — horizontal edges, vertical edges, corners, blobs.",
  },
  activation: {
    color: "#06b6d4",
    tag: "ReLU activation",
    headline: "Throwing away the 'not detected' evidence",
    body:
      "ReLU (Rectified Linear Unit) is the simplest possible rule: if a value is negative, set it to zero; otherwise leave it alone. A negative activation means 'this filter's pattern was actively absent here' — the network doesn't need that nuance, it only cares about 'how strongly present.' Zeroing negatives also breaks linearity, which is what lets stacked layers learn curved decision boundaries instead of just one big straight line.",
  },
  pool: {
    color: "#f59e0b",
    tag: "Max pooling",
    headline: "Keeping only the strongest signal in each neighborhood",
    body:
      "Max pooling looks at each 2×2 block of the feature map and keeps only the single largest value, discarding the other three. This shrinks the map to a quarter of its size and makes the network care less about the exact pixel position of a feature and more about whether it appeared somewhere in that neighborhood — a property called translation invariance.",
  },
  flatten: {
    color: "#ec4899",
    tag: "Flatten",
    headline: "Unrolling the 3D stack into one long list",
    body:
      "At this point the data is a stack of 16 small 5×5 feature maps — 400 numbers arranged spatially. Flatten just unrolls that 3D block into a single 400-number list so it can be fed into an ordinary fully-connected layer, which only understands flat vectors, not grids.",
  },
  dense: {
    color: "#10b981",
    tag: "Dense (fully connected)",
    headline: "Every feature votes on every neuron",
    body:
      "Each of the 32 neurons here looks at all 400 incoming numbers at once and computes its own weighted sum — effectively asking 'how much do I care about this exact combination of features?' This is where the network starts combining spatially-separate clues ('a loop near the top' + 'a straight line at the bottom') into higher-level concepts.",
  },
  output: {
    color: "#f97316",
    tag: "Output (softmax)",
    headline: "Turning 10 raw scores into a probability distribution",
    body:
      "The final layer produces one raw score per digit (0–9). Softmax exponentiates each score and divides by the total, which guarantees every result is between 0 and 1 and all ten add up to exactly 1 — a genuine probability distribution the network is 'voting' on.",
  },
};

function forwardPassCNN(inputImage, model) {
  const stages = [];
  stages.push({ name: "Input", type: "input", maps: [inputImage], mapSize: [28, 28] });

  const conv1 = conv2dMultiChannel([inputImage], model.conv1.kernels, model.conv1.bias);
  stages.push({ name: "Conv layer 1", type: "conv", maps: conv1, mapSize: [26, 26], numFilters: 8, kernelSize: "3×3" });

  const relu1 = reluMaps(conv1);
  stages.push({ name: "ReLU", type: "activation", maps: relu1, mapSize: [26, 26] });

  const pool1 = maxPool2d(relu1, 2);
  stages.push({ name: "Max pool 2×2", type: "pool", maps: pool1, mapSize: [13, 13] });

  const conv2 = conv2dMultiChannel(pool1, model.conv2.kernels, model.conv2.bias);
  stages.push({ name: "Conv layer 2", type: "conv", maps: conv2, mapSize: [11, 11], numFilters: 16, kernelSize: "3×3" });

  const relu2 = reluMaps(conv2);
  stages.push({ name: "ReLU", type: "activation", maps: relu2, mapSize: [11, 11] });

  const pool2 = maxPool2d(relu2, 2);
  stages.push({ name: "Max pool 2×2", type: "pool", maps: pool2, mapSize: [5, 5] });

  const flat = flattenMaps(pool2);
  stages.push({ name: "Flatten", type: "flatten", values: flat });

  const dense1Raw = denseForward(flat, model.dense1.weights, model.dense1.bias);
  const dense1 = Float32Array.from(dense1Raw, (v) => (v > 0 ? v : 0));
  stages.push({ name: "Dense layer (32)", type: "dense", values: Array.from(dense1) });

  const dense2 = denseForward(dense1, model.dense2.weights, model.dense2.bias);
  const probs = softmax(Array.from(dense2));
  stages.push({ name: "Output (softmax)", type: "output", values: Array.from(probs) });

  return stages;
}

/* ---------- color helpers ---------- */
function heatmapColor(t) {
  // t in [0,1] -> deep violet (low) through cyan/amber (high), dark-mode friendly
  const stops = [
    [8, 8, 20], [76, 29, 149], [124, 58, 237], [6, 182, 212], [245, 158, 11], [253, 224, 71],
  ];
  const n = stops.length - 1;
  const scaled = Math.max(0, Math.min(1, t)) * n;
  const i = Math.min(n - 1, Math.floor(scaled));
  const f = scaled - i;
  const a = stops[i], b = stops[i + 1];
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bl = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${r},${g},${bl})`;
}

/* ---------- drawing canvas ---------- */
function DrawingCanvas({ onImageChange, size = 260 }) {
  const canvasRef = useRef(null);
  const isDrawing = useRef(false);
  const lastPos = useRef(null);
  const [active, setActive] = useState(false);
  const [hasDrawn, setHasDrawn] = useState(false);

  const getImageData = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const ctx = canvas.getContext("2d");
    const imgDataRaw = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    
    // Find bounding box
    let minX = canvas.width, minY = canvas.height, maxX = -1, maxY = -1;
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        if (imgDataRaw[(y * canvas.width + x) * 4] > 10) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    
    if (maxX < minX || maxY < minY) return null; // Empty drawing
    
    const boxW = maxX - minX + 1;
    const boxH = maxY - minY + 1;
    const scale = 20.0 / Math.max(boxW, boxH);
    
    const scaledW = Math.max(1, Math.round(boxW * scale));
    const scaledH = Math.max(1, Math.round(boxH * scale));
    
    // Crop and scale
    const tmp = document.createElement("canvas");
    tmp.width = scaledW; tmp.height = scaledH;
    const tctx = tmp.getContext("2d");
    tctx.drawImage(canvas, minX, minY, boxW, boxH, 0, 0, scaledW, scaledH);
    
    const scaledData = tctx.getImageData(0, 0, scaledW, scaledH).data;
    
    // Center of mass
    let cx = 0, cy = 0, total = 0;
    for (let y = 0; y < scaledH; y++) {
      for (let x = 0; x < scaledW; x++) {
        const v = scaledData[(y * scaledW + x) * 4];
        cx += x * v; cy += y * v; total += v;
      }
    }
    if (total > 0) { cx /= total; cy /= total; }
    else { cx = scaledW / 2; cy = scaledH / 2; }
    
    // Place into 28x28 at center of mass
    const finalCanvas = document.createElement("canvas");
    finalCanvas.width = 28; finalCanvas.height = 28;
    const fctx = finalCanvas.getContext("2d");
    fctx.fillStyle = "#000"; fctx.fillRect(0, 0, 28, 28);
    const dx = Math.round(14 - cx);
    const dy = Math.round(14 - cy);
    fctx.drawImage(tmp, dx, dy);
    
    // Standard MNIST normalization
    const fData = fctx.getImageData(0, 0, 28, 28).data;
    const img = [];
    for (let i = 0; i < 28; i++) {
      const row = new Float32Array(28);
      for (let j = 0; j < 28; j++) {
        const p = fData[(i * 28 + j) * 4] / 255.0;
        row[j] = (p - 0.1307) / 0.3081;
      }
      img.push(row);
    }
    return img;
  }, []);

  const clear = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    lastPos.current = null;
    setHasDrawn(false);
    onImageChange(null);
  }, [onImageChange]);

  const pos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) * (canvasRef.current.width / rect.width),
      y: (clientY - rect.top) * (canvasRef.current.height / rect.height),
    };
  };

  const down = (e) => {
    e.preventDefault();
    isDrawing.current = true;
    setActive(true);
    lastPos.current = pos(e);
  };
  const move = (e) => {
    if (!isDrawing.current) return;
    e.preventDefault();
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const p = pos(e);
    if (lastPos.current) {
      ctx.beginPath();
      ctx.moveTo(lastPos.current.x, lastPos.current.y);
      ctx.lineTo(p.x, p.y);
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 16;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.stroke();
    }
    lastPos.current = p;
  };
  const up = () => {
    if (!isDrawing.current) return;
    isDrawing.current = false;
    lastPos.current = null;
    setActive(false);
    setHasDrawn(true);
    onImageChange(getImageData());
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = 280; canvas.height = 280;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, 280, 280);
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
      <div
        style={{
          position: "relative", width: size, height: size, borderRadius: 14, overflow: "hidden",
          border: "1px solid rgba(139,92,246,0.35)",
          boxShadow: active ? "0 0 0 3px rgba(139,92,246,0.25), 0 0 30px rgba(139,92,246,0.35)" : "0 0 0 1px rgba(255,255,255,0.03)",
          transition: "box-shadow 0.2s ease",
        }}
      >
        <canvas
          ref={canvasRef}
          style={{ width: size, height: size, cursor: "crosshair", touchAction: "none", display: "block" }}
          onMouseDown={down} onMouseMove={move} onMouseUp={up} onMouseLeave={up}
          onTouchStart={down} onTouchMove={move} onTouchEnd={up}
        />
        <div
          style={{
            position: "absolute", inset: 0, pointerEvents: "none",
            backgroundImage:
              "linear-gradient(rgba(139,92,246,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(139,92,246,0.06) 1px, transparent 1px)",
            backgroundSize: `${size / 28}px ${size / 28}px`,
          }}
        />
        {!hasDrawn && (
          <div style={{
            position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
            pointerEvents: "none", color: "rgba(255,255,255,0.25)", fontSize: 13, fontFamily: "system-ui, sans-serif",
          }}>
            draw a digit here
          </div>
        )}
      </div>
      <button
        onClick={clear}
        style={{
          background: "transparent", border: "1px solid rgba(255,255,255,0.12)", color: "rgba(255,255,255,0.65)",
          borderRadius: 8, padding: "6px 14px", fontSize: 12, cursor: "pointer", fontFamily: "system-ui, sans-serif",
        }}
        onMouseEnter={(e) => (e.currentTarget.style.borderColor = "rgba(255,255,255,0.3)")}
        onMouseLeave={(e) => (e.currentTarget.style.borderColor = "rgba(255,255,255,0.12)")}
      >
        Clear canvas
      </button>
    </div>
  );
}

/* ---------- one feature-map tile drawn on a small canvas ---------- */
function FeatureMapTile({ map, size = 64, label, highlight, onHover }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !map) return;
    const h = map.length, w = map[0].length;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d");
    let min = Infinity, max = -Infinity;
    for (const row of map) for (let i = 0; i < row.length; i++) { min = Math.min(min, row[i]); max = Math.max(max, row[i]); }
    const range = max - min || 1;
    const imgData = ctx.createImageData(w, h);
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        const v = map[i][j];
        const norm = (v - min) / range;
        const [r, g, b] = heatmapColor(norm).match(/\d+/g).map(Number);
        const idx = (i * w + j) * 4;
        imgData.data[idx] = r; imgData.data[idx + 1] = g; imgData.data[idx + 2] = b; imgData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }, [map]);

  return (
    <div
      onMouseEnter={() => onHover && onHover(label)}
      onMouseLeave={() => onHover && onHover(null)}
      style={{
        width: size, height: size, borderRadius: 6, overflow: "hidden",
        border: highlight ? "1.5px solid #f59e0b" : "1px solid rgba(255,255,255,0.08)",
        boxShadow: highlight ? "0 0 12px rgba(245,158,11,0.4)" : "none",
        cursor: "pointer", flexShrink: 0, transition: "border 0.15s ease, box-shadow 0.15s ease",
      }}
      title={label}
    >
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%", imageRendering: "pixelated", display: "block" }} />
    </div>
  );
}

/* ---------- grid of feature maps for conv/activation/pool stages ---------- */
function FeatureMapGrid({ maps, tileSize = 64 }) {
  const [hovered, setHovered] = useState(null);
  if (!maps || !maps.length) return null;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
      {maps.map((m, i) => (
        <FeatureMapTile
          key={i}
          map={m}
          size={tileSize}
          label={`filter ${i + 1}`}
          highlight={hovered === `filter ${i + 1}`}
          onHover={setHovered}
        />
      ))}
    </div>
  );
}

/* ---------- kernel mini-preview (shows the actual 3x3 numbers for conv1) ---------- */
function KernelPreview({ kernels }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
      {kernels.map((k, i) => {
        const grid = k[0]; // single input channel for conv1
        const flat = grid.flat();
        const max = Math.max(...flat.map(Math.abs)) || 1;
        return (
          <div key={i} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
            <div style={{
              display: "grid", gridTemplateColumns: "repeat(3, 16px)", gridTemplateRows: "repeat(3, 16px)",
              border: "1px solid rgba(255,255,255,0.1)", borderRadius: 4, overflow: "hidden",
            }}>
              {grid.flatMap((row, ri) => row.map((v, ci) => {
                const t = (v / max + 1) / 2;
                return (
                  <div key={`${ri}-${ci}`} style={{
                    width: 16, height: 16, background: heatmapColor(t),
                    display: "flex", alignItems: "center", justifyContent: "center",
                  }} />
                );
              }))}
            </div>
            <span style={{ fontSize: 9, color: "rgba(255,255,255,0.4)", fontFamily: "monospace" }}>f{i + 1}</span>
          </div>
        );
      })}
    </div>
  );
}

/* ---------- dense vector shown as a bar column ---------- */
function DenseVector({ values, max = 64 }) {
  const display = values.slice(0, max);
  const maxAbs = Math.max(...display.map((v) => Math.abs(v)), 0.001);
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 90, flexWrap: "nowrap", overflowX: "auto", paddingBottom: 2 }}>
      {display.map((v, i) => {
        const h = Math.max(2, (Math.abs(v) / maxAbs) * 84);
        return (
          <div
            key={i}
            title={`neuron ${i + 1}: ${v.toFixed(3)}`}
            style={{
              width: 6, height: h, borderRadius: 2, flexShrink: 0,
              background: heatmapColor(Math.abs(v) / maxAbs),
              opacity: 0.9,
            }}
          />
        );
      })}
      {values.length > max && (
        <span style={{ fontSize: 10, color: "rgba(255,255,255,0.35)", marginLeft: 6, alignSelf: "center" }}>
          +{values.length - max} more
        </span>
      )}
    </div>
  );
}

/* ---------- output probability bars ---------- */
function OutputProbs({ probs }) {
  const maxIdx = probs.indexOf(Math.max(...probs));
  return (
    <div style={{ width: "100%" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {probs.map((p, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{
              width: 14, textAlign: "center", fontFamily: "monospace", fontSize: 12,
              color: i === maxIdx ? "#f59e0b" : "rgba(255,255,255,0.45)",
              fontWeight: i === maxIdx ? 700 : 400,
            }}>{i}</span>
            <div style={{ flex: 1, height: 16, borderRadius: 5, background: "rgba(255,255,255,0.05)", overflow: "hidden" }}>
              <div style={{
                height: "100%", width: `${p * 100}%`, borderRadius: 5,
                background: i === maxIdx ? "linear-gradient(90deg,#f59e0b,#f97316)" : "rgba(139,92,246,0.4)",
                transition: "width 0.3s ease",
              }} />
            </div>
            <span style={{
              width: 44, textAlign: "right", fontFamily: "monospace", fontSize: 11,
              color: i === maxIdx ? "#f59e0b" : "rgba(255,255,255,0.45)",
              fontWeight: i === maxIdx ? 700 : 400,
            }}>{(p * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
      <div style={{ textAlign: "center", marginTop: 16 }}>
        <div style={{ fontSize: 10, color: "rgba(255,255,255,0.4)", textTransform: "uppercase", letterSpacing: 1 }}>
          Predicted digit
        </div>
        <div style={{ fontSize: 44, fontWeight: 800, color: "#f59e0b", textShadow: "0 0 24px rgba(245,158,11,0.35)" }}>
          {maxIdx}
        </div>
      </div>
    </div>
  );
}

/* ---------- pipeline strip at the top of the right panel ---------- */
function PipelineStrip({ stages, activeStage, setActiveStage }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4, overflowX: "auto", paddingBottom: 2 }}>
      {stages.map((stage, i) => {
        const meta = STAGE_META[stage.type];
        const isActive = activeStage === i;
        return (
          <div key={i} style={{ display: "flex", alignItems: "center", flexShrink: 0 }}>
            <button
              onClick={() => setActiveStage(i)}
              style={{
                padding: "6px 10px", borderRadius: 8, fontSize: 11, whiteSpace: "nowrap",
                border: "1px solid " + (isActive ? meta.color + "88" : "rgba(255,255,255,0.08)"),
                background: isActive ? meta.color + "22" : "transparent",
                color: isActive ? meta.color : "rgba(255,255,255,0.5)",
                cursor: "pointer", fontWeight: isActive ? 700 : 500,
                fontFamily: "system-ui, sans-serif", transition: "all 0.15s ease",
              }}
            >
              {stage.name}
            </button>
            {i < stages.length - 1 && (
              <span style={{ color: "rgba(255,255,255,0.2)", margin: "0 2px", fontSize: 12 }}>›</span>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ---------- "why this matters" math panel, expandable ---------- */
function MathDetail({ stage }) {
  const [open, setOpen] = useState(false);
  const meta = STAGE_META[stage.type];

  let formula = null;
  let extra = null;
  if (stage.type === "conv") {
    formula = "output[i,j] = Σ (input patch ⊙ kernel) + bias";
    extra = `This stage has ${stage.numFilters} filters, each ${stage.kernelSize}. Every filter is applied to the whole image independently, producing ${stage.numFilters} separate feature maps — one per filter — each showing where that specific pattern was found.`;
  } else if (stage.type === "activation") {
    formula = "output = max(0, input)";
    extra = "Applied element-wise. No parameters, no learning happens here — it's a fixed rule.";
  } else if (stage.type === "pool") {
    formula = "output[i,j] = max(2×2 block)";
    extra = "Also has no learnable parameters. It's a deterministic downsampling rule, which is part of why CNNs generalize well with relatively few weights.";
  } else if (stage.type === "dense") {
    formula = "output[j] = Σ (input[i] × weight[j,i]) + bias[j]";
    extra = "Unlike conv layers, every input connects to every output — hence 'fully connected.' This is where most of the network's parameters usually live.";
  } else if (stage.type === "output") {
    formula = "prob[i] = eˣⁱ / Σ eˣᵏ";
    extra = "Softmax amplifies the gap between the largest score and the rest, which is why the network's top guess usually ends up with a probability much higher than any runner-up, even when the raw scores were close.";
  } else if (stage.type === "flatten") {
    formula = "vector = concat(map₁, map₂, ..., map₁₆)";
    extra = "Purely a reshape — no computation. 16 maps of 5×5 become one vector of 400 numbers.";
  }

  if (!formula) return null;

  return (
    <div style={{ marginTop: 10 }}>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          display: "flex", alignItems: "center", gap: 6, background: "transparent", border: "none",
          color: "rgba(255,255,255,0.5)", fontSize: 11, cursor: "pointer", padding: 0, fontFamily: "system-ui, sans-serif",
        }}
      >
        <span style={{ transform: open ? "rotate(90deg)" : "none", transition: "transform 0.15s ease", display: "inline-block" }}>›</span>
        {open ? "Hide the math" : "Show the math"}
      </button>
      {open && (
        <div style={{
          marginTop: 8, padding: "10px 12px", borderRadius: 8, background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.06)",
        }}>
          <div style={{ fontFamily: "monospace", fontSize: 12.5, color: meta.color, marginBottom: 6 }}>{formula}</div>
          <div style={{ fontSize: 12, color: "rgba(255,255,255,0.55)", lineHeight: 1.5 }}>{extra}</div>
        </div>
      )}
    </div>
  );
}

/* ---------- top-level "how CNNs work" primer, collapsible ---------- */
function CNNPrimer() {
  const [open, setOpen] = useState(false);
  const steps = [
    { t: "Convolution", c: "#8b5cf6", d: "Small filters slide over the image hunting for local patterns — edges, corners, curves." },
    { t: "Activation", c: "#06b6d4", d: "ReLU keeps only positive 'this pattern is present' signals, discarding the rest." },
    { t: "Pooling", c: "#f59e0b", d: "Shrinks the map, keeping the strongest response in each neighborhood." },
    { t: "Repeat, deeper", c: "#ec4899", d: "A second conv+pool stage combines early edges into more complex shapes like loops and corners." },
    { t: "Dense layers", c: "#10b981", d: "Fully-connected layers combine all spatial features into a global judgment." },
    { t: "Softmax", c: "#f97316", d: "Converts 10 raw scores into a clean probability distribution over digits 0–9." },
  ];
  return (
    <div style={{
      borderRadius: 12, border: "1px solid rgba(139,92,246,0.2)", background: "rgba(139,92,246,0.04)",
      padding: "14px 16px", marginBottom: 16,
    }}>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          display: "flex", alignItems: "center", justifyContent: "space-between", width: "100%",
          background: "transparent", border: "none", cursor: "pointer", padding: 0,
        }}
      >
        <span style={{ fontSize: 14, fontWeight: 700, color: "#c4b5fd", fontFamily: "system-ui, sans-serif" }}>
          How does a CNN actually recognize a digit?
        </span>
        <span style={{ color: "rgba(255,255,255,0.4)", fontSize: 13, transform: open ? "rotate(180deg)" : "none", transition: "transform 0.2s" }}>▾</span>
      </button>
      {open && (
        <div style={{ marginTop: 12 }}>
          <p style={{ fontSize: 12.5, color: "rgba(255,255,255,0.65)", lineHeight: 1.6, margin: "0 0 12px" }}>
            A convolutional neural network never "sees" a digit the way you do. It builds understanding in layers,
            starting from raw pixels and ending at a decision — each layer only ever knows what the layer before it
            handed over. The trick is that early layers detect tiny, simple patterns, and later layers combine those
            simple patterns into increasingly abstract concepts, until the final layer can say "this combination of
            features has only ever meant the digit 7 in my training data."
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 10 }}>
            {steps.map((s, i) => (
              <div key={i} style={{
                borderLeft: `2px solid ${s.c}`, paddingLeft: 10, display: "flex", flexDirection: "column", gap: 2,
              }}>
                <span style={{ fontSize: 11.5, fontWeight: 700, color: s.c, fontFamily: "system-ui, sans-serif" }}>
                  {i + 1}. {s.t}
                </span>
                <span style={{ fontSize: 11, color: "rgba(255,255,255,0.5)", lineHeight: 1.45 }}>{s.d}</span>
              </div>
            ))}
          </div>
          <p style={{ fontSize: 11.5, color: "rgba(255,255,255,0.4)", lineHeight: 1.55, marginTop: 12, marginBottom: 0, fontStyle: "italic" }}>
            Note: this demo uses hand-crafted edge/corner filters for layer 1 (so what you see is a genuine edge
            detector, not noise) and small seeded-random weights for the rest — it wasn't trained on real MNIST data,
            so treat the prediction as illustrative of the mechanism, not a production-accurate classifier.
          </p>
        </div>
      )}
    </div>
  );
}

/* ---------- main component ---------- */
export default function CNNExplainer() {
  const model = useMemo(() => MNIST_WEIGHTS, []);
  const [image, setImage] = useState(null);
  const [stages, setStages] = useState(null);
  const [activeStage, setActiveStage] = useState(0);

  const handleImage = useCallback((img) => {
    setImage(img);
    if (!img) { setStages(null); setActiveStage(0); return; }
    let hasContent = false;
    for (let i = 0; i < 28 && !hasContent; i++)
      for (let j = 0; j < 28 && !hasContent; j++)
        if (img[i][j] > 0.1) hasContent = true;
    if (!hasContent) { setStages(null); return; }
    const result = forwardPassCNN(img, model);
    setStages(result);
    setActiveStage(0);
  }, [model]);

  const current = stages ? stages[activeStage] : null;
  const meta = current ? STAGE_META[current.type] : null;

  return (
    <div style={{
      fontFamily: "system-ui, -apple-system, sans-serif",
      background: "#06070e",
      minHeight: "100vh",
      color: "#fff",
      padding: "28px 20px",
    }}>
      <div style={{ maxWidth: 1180, margin: "0 auto" }}>
        {/* header */}
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
          <div style={{
            width: 40, height: 40, borderRadius: 10, flexShrink: 0,
            background: "linear-gradient(135deg,#ec4899,#8b5cf6)",
            display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18,
          }}>🧠</div>
          <div>
            <h1 style={{ fontSize: 18, fontWeight: 700, margin: 0 }}>CNN digit recognition</h1>
            <p style={{ fontSize: 12, color: "rgba(255,255,255,0.45)", margin: "2px 0 0" }}>
              Draw a digit and watch a convolutional neural network process it, layer by layer, in real time
            </p>
          </div>
        </div>

        <CNNPrimer />

        <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
          {/* left: draw + output */}
          <div style={{
            width: 300, flexShrink: 0, display: "flex", flexDirection: "column", gap: 20,
            borderRadius: 14, border: "1px solid rgba(255,255,255,0.06)", background: "rgba(255,255,255,0.015)",
            padding: 18,
          }}>
            <DrawingCanvas onImageChange={handleImage} size={260} />
            {stages && (
              <div>
                <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 1, color: "rgba(255,255,255,0.4)", marginBottom: 8 }}>
                  Prediction
                </div>
                <OutputProbs probs={stages[stages.length - 1].values} />
              </div>
            )}
            {!stages && (
              <div style={{ fontSize: 12, color: "rgba(255,255,255,0.35)", lineHeight: 1.6, textAlign: "center" }}>
                Draw any digit 0–9 on the canvas above. As soon as you lift the pen, the full forward pass runs and
                you can step through every layer on the right.
              </div>
            )}
          </div>

          {/* right: pipeline + stage detail */}
          <div style={{
            flex: 1, minWidth: 340, borderRadius: 14, border: "1px solid rgba(255,255,255,0.06)",
            background: "rgba(255,255,255,0.015)", padding: 18, display: "flex", flexDirection: "column", gap: 16,
          }}>
            {stages ? (
              <>
                <PipelineStrip stages={stages} activeStage={activeStage} setActiveStage={setActiveStage} />

                <div>
                  <div style={{
                    display: "inline-block", fontSize: 10, fontWeight: 700, letterSpacing: 0.5,
                    textTransform: "uppercase", color: meta.color, background: meta.color + "1a",
                    border: `1px solid ${meta.color}44`, borderRadius: 6, padding: "3px 8px", marginBottom: 8,
                  }}>
                    {meta.tag}
                  </div>
                  <h2 style={{ fontSize: 16, fontWeight: 700, margin: "0 0 6px" }}>{meta.headline}</h2>
                  <p style={{ fontSize: 12.5, color: "rgba(255,255,255,0.6)", lineHeight: 1.6, margin: 0 }}>{meta.body}</p>
                  <MathDetail stage={current} />
                </div>

                <div style={{ borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: 16 }}>
                  {current.type === "input" && (
                    <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
                      <FeatureMapTile map={current.maps[0]} size={140} label="input" />
                      <span style={{ fontSize: 11, color: "rgba(255,255,255,0.4)" }}>28 × 28 = 784 pixel values, each 0–1</span>
                    </div>
                  )}

                  {current.type === "conv" && current.name === "Conv layer 1" && (
                    <div>
                      <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginBottom: 8 }}>
                        The 8 filters (actual kernel weights, red = negative, amber/yellow = positive):
                      </div>
                      <div style={{ marginBottom: 14 }}><KernelPreview kernels={model.conv1.kernels} /></div>
                      <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginBottom: 8 }}>
                        Resulting feature maps — hover any tile to see which filter produced it:
                      </div>
                      <FeatureMapGrid maps={current.maps} tileSize={72} />
                    </div>
                  )}

                  {current.type !== "conv" || current.name !== "Conv layer 1" ? (
                    current.maps && (
                      <div>
                        <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginBottom: 8 }}>
                          {current.maps.length} feature map{current.maps.length > 1 ? "s" : ""}, {current.mapSize[0]}×{current.mapSize[1]} each:
                        </div>
                        <FeatureMapGrid maps={current.maps} tileSize={current.mapSize[0] > 20 ? 56 : 72} />
                      </div>
                    )
                  ) : null}

                  {current.values && current.type !== "output" && (
                    <div>
                      <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginBottom: 8 }}>
                        {current.values.length}-value vector — bar height/color = magnitude:
                      </div>
                      <DenseVector values={current.values} />
                    </div>
                  )}

                  {current.type === "output" && <OutputProbs probs={current.values} />}
                </div>
              </>
            ) : (
              <div style={{
                flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                minHeight: 320, color: "rgba(255,255,255,0.3)", gap: 8,
              }}>
                <div style={{ fontSize: 28 }}>✏️</div>
                <div style={{ fontSize: 13 }}>Draw a digit to see the network run</div>
              </div>
            )}
          </div>
        </div>

        {/* pipeline overview strip */}
        {stages && (
          <div style={{ marginTop: 20, display: "flex", alignItems: "center", gap: 3, flexWrap: "wrap" }}>
            {stages.map((s, i) => {
              const m = STAGE_META[s.type];
              const isActive = activeStage === i;
              return (
                <div
                  key={i}
                  onClick={() => setActiveStage(i)}
                  title={s.name}
                  style={{
                    width: 26, height: 18, borderRadius: 4, cursor: "pointer",
                    background: isActive ? m.color : m.color + "33",
                    boxShadow: isActive ? `0 0 10px ${m.color}88` : "none",
                    transition: "all 0.15s ease",
                  }}
                />
              );
            })}
            <span style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", marginLeft: 8 }}>full pipeline — click any block</span>
          </div>
        )}
      </div>
    </div>
  );
}