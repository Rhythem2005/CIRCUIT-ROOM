// NeuroForge — Representation Explorer (Redesigned)

import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Layers, Play, Clock, Eye } from 'lucide-react';
import useStore from '../../stores/useStore';
import { CLASS_PALETTE } from '../../components/ui/ColorSystem.js';

function LayerScatterPlot({ data, labels, title, width = 280, height = 240 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    // PCA to 2D
    let points2D;
    const dim = Array.isArray(data[0]) ? data[0].length : 1;

    if (dim === 1) {
      points2D = data.map((d, i) => [Array.isArray(d) ? d[0] : d, i / data.length]);
    } else if (dim === 2) {
      points2D = data.map(d => [d[0], d[1]]);
    } else {
      const n = data.length;
      const means = new Array(dim).fill(0);
      for (const row of data) for (let j = 0; j < dim; j++) means[j] += row[j] / n;
      const variances = new Array(dim).fill(0);
      for (const row of data) for (let j = 0; j < dim; j++) variances[j] += (row[j] - means[j]) ** 2;
      const sortedDims = variances.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v);
      const d1 = sortedDims[0].i;
      const d2 = sortedDims.length > 1 ? sortedDims[1].i : 0;
      points2D = data.map(row => [row[d1] - means[d1], row[d2] - means[d2]]);
    }

    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const [x, y] of points2D) {
      if (isFinite(x) && isFinite(y)) {
        xMin = Math.min(xMin, x); xMax = Math.max(xMax, x);
        yMin = Math.min(yMin, y); yMax = Math.max(yMax, y);
      }
    }
    const pad = 0.1;
    const xRange = (xMax - xMin) || 1;
    const yRange = (yMax - yMin) || 1;
    xMin -= xRange * pad; xMax += xRange * pad;
    yMin -= yRange * pad; yMax += yRange * pad;

    const toX = (x) => ((x - xMin) / (xMax - xMin)) * (width - 28) + 14;
    const toY = (y) => height - 22 - ((y - yMin) / (yMax - yMin)) * (height - 36);

    // Background
    ctx.fillStyle = 'rgba(6, 7, 14, 0.6)';
    ctx.fillRect(0, 0, width, height);

    // Grid
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.03)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const gx = 14 + i * (width - 28) / 4;
      ctx.beginPath(); ctx.moveTo(gx, 8); ctx.lineTo(gx, height - 22); ctx.stroke();
      const gy = 8 + i * (height - 36) / 4;
      ctx.beginPath(); ctx.moveTo(14, gy); ctx.lineTo(width - 14, gy); ctx.stroke();
    }

    // Points
    for (let i = 0; i < points2D.length; i++) {
      const [x, y] = points2D[i];
      if (!isFinite(x) || !isFinite(y)) continue;
      const cls = labels[i];
      const cx = toX(x);
      const cy = toY(y);
      ctx.beginPath();
      ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = CLASS_PALETTE[cls % CLASS_PALETTE.length] + 'cc';
      ctx.fill();
    }

    // Title
    ctx.fillStyle = 'rgba(146, 149, 179, 0.6)';
    ctx.font = '10px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(title, width / 2, height - 5);
  }, [data, labels, width, height, title]);

  return <canvas ref={canvasRef} className="rounded-lg" style={{ width, height, border: '1px solid rgba(139,92,246,0.06)' }} />;
}

export default function RepresentationExplorer() {
  const {
    network, dataset, epoch, initNetwork,
    representationSnapshots, snapshotRepresentations,
    trainOneEpoch, updateDecisionBoundary
  } = useStore();
  const [timelineIdx, setTimelineIdx] = useState(0);

  useEffect(() => { if (!network) initNetwork(); }, []);

  const captureSnapshot = () => { if (network && dataset) snapshotRepresentations(); };
  const trainAndCapture = () => {
    for (let i = 0; i < 10; i++) trainOneEpoch();
    captureSnapshot();
    updateDecisionBoundary();
  };

  const reps = (network && dataset) ? network.snapshotRepresentations(dataset.data) : [];

  return (
    <div className="h-full flex flex-col">
      <div className="module-header">
        <div className="flex items-center gap-3">
          <div className="module-icon" style={{ background: 'linear-gradient(135deg, #22d3ee, #60a5fa)' }}>
            <Layers className="w-4 h-4 text-white" />
          </div>
          <div>
            <h2 className="text-headline text-base text-text-primary">Representation Explorer</h2>
            <p className="text-[11px] text-text-tertiary mt-0.5">How each layer transforms data from tangled to separable</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={captureSnapshot} className="btn-ghost flex items-center gap-1.5 text-xs">
            <Eye className="w-3.5 h-3.5" /> Snapshot
          </button>
          <button onClick={trainAndCapture} className="btn-primary flex items-center gap-1.5 text-xs">
            <Play className="w-3.5 h-3.5" /> Train 10 + Snapshot
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-auto p-5 space-y-6">
        {/* Current representations */}
        <div>
          <div className="text-label mb-3 flex items-center gap-2">
            Current Layer Views
            <span className="text-[10px] text-text-ghost font-normal normal-case tracking-normal">(Epoch {epoch})</span>
          </div>
          <div className="flex gap-3 overflow-x-auto pb-3">
            {reps.map((rep, i) => (
              <motion.div key={i} initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.08 }} className="flex-shrink-0">
                <LayerScatterPlot data={rep.data} labels={dataset?.rawLabels || []} title={rep.name} />
              </motion.div>
            ))}
          </div>
          {reps.length > 1 && (
            <div className="flex items-center gap-1 mt-1 px-4">
              {reps.map((_, i) => (
                <div key={i} className="flex items-center gap-1 flex-1">
                  <div className={`h-[2px] flex-1 rounded-full ${i === 0 ? 'bg-nf-cyan/20' : i === reps.length - 1 ? 'bg-nf-amber/20' : 'bg-nf-violet/20'}`} />
                  {i < reps.length - 1 && <span className="text-[10px] text-text-ghost">→</span>}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Timeline */}
        {representationSnapshots.length > 0 && (
          <div>
            <div className="text-label mb-3 flex items-center gap-2">
              <Clock className="w-3 h-3 text-nf-violet" /> Training Timeline
              <span className="text-[10px] text-text-ghost font-normal normal-case tracking-normal">({representationSnapshots.length} snapshots)</span>
            </div>
            <input type="range" min="0" max={representationSnapshots.length - 1} value={timelineIdx}
              onChange={(e) => setTimelineIdx(parseInt(e.target.value))} className="w-full mb-2" />
            <div className="flex justify-between text-[9px] text-text-ghost mb-3">
              <span>Epoch {representationSnapshots[0]?.epoch || 0}</span>
              <span>Epoch {representationSnapshots[representationSnapshots.length - 1]?.epoch || 0}</span>
            </div>
            {representationSnapshots[timelineIdx] && (
              <div className="flex gap-3 overflow-x-auto pb-3">
                {representationSnapshots[timelineIdx].layers.map((rep, i) => (
                  <div key={i} className="flex-shrink-0">
                    <LayerScatterPlot data={rep.data} labels={dataset?.rawLabels || []}
                      title={`${rep.name} (E${representationSnapshots[timelineIdx].epoch})`} />
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Educational */}
        <div className="callout callout-cyan">
          <h4 className="text-xs font-bold text-nf-cyan mb-1">💡 What am I looking at?</h4>
          <p className="text-[11px] text-text-secondary leading-relaxed">
            Each plot shows how a layer "sees" the data. The input shows raw 2D coordinates. Hidden layers learn
            to <strong className="text-nf-cyan">untangle</strong> the data — transforming complex, interleaved patterns into linearly
            separable clusters. This is deep learning's core mechanism: <strong className="text-nf-cyan">hierarchical feature extraction</strong>.
          </p>
        </div>
      </div>
    </div>
  );
}
