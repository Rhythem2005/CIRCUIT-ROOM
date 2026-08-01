// Circuit Room — Attention Lab (Redesigned)

import { useRef, useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Thermometer, Eye } from 'lucide-react';
import { heatmapColor } from '../../components/ui/ColorSystem.js';

function computeAttention(tokens, dModel = 32, numHeads = 4, temperature = 1.0) {
  const seqLen = tokens.length;
  const headDim = Math.floor(dModel / numHeads);
  const embeddings = tokens.map((token, i) => {
    const embed = new Array(dModel).fill(0);
    for (let j = 0; j < dModel; j++) {
      const cc = token.charCodeAt(j % token.length) || 0;
      embed[j] = Math.sin(cc * 0.1 + j * 0.3 + i * 0.7) * 0.5;
    }
    return embed;
  });

  const heads = [];
  for (let h = 0; h < numHeads; h++) {
    const seed = h * 17 + 3;
    const Q = embeddings.map(e => e.slice(h*headDim, (h+1)*headDim).map((v,j) => v*Math.cos(seed+j*0.5)+0.1*Math.sin(seed*j)));
    const K = embeddings.map(e => e.slice(h*headDim, (h+1)*headDim).map((v,j) => v*Math.sin(seed+j*0.3)+0.1*Math.cos(seed*j)));
    const V = embeddings.map(e => e.slice(h*headDim, (h+1)*headDim).map((v,j) => v*0.8+0.2*Math.sin(j+seed)));
    const scale = Math.sqrt(headDim);
    const scores = []; 
    for (let i = 0; i < seqLen; i++) {
      const row = [];
      for (let j = 0; j < seqLen; j++) {
        let dot = 0; for (let k = 0; k < headDim; k++) dot += Q[i][k]*K[j][k];
        row.push(dot / (scale * temperature));
      }
      scores.push(row);
    }
    const attention = scores.map(row => {
      const mx = Math.max(...row); const exp = row.map(v => Math.exp(v-mx));
      const sum = exp.reduce((a,b) => a+b, 0); return exp.map(v => v/sum);
    });
    const output = Array.from({ length: seqLen }, (_, i) => {
      const out = new Array(headDim).fill(0);
      for (let j = 0; j < seqLen; j++) for (let k = 0; k < headDim; k++) out[k] += attention[i][j]*V[j][k];
      return out;
    });
    heads.push({ Q, K, V, scores, attention, output });
  }
  const combined = Array.from({ length: seqLen }, (_, i) => {
    const row = [];
    for (let j = 0; j < seqLen; j++) { let avg = 0; for (let h = 0; h < numHeads; h++) avg += heads[h].attention[i][j]; row.push(avg/numHeads); }
    return row;
  });
  return { heads, combinedAttention: combined, embeddings };
}

function AttentionHeatmap({ attention, tokens, selectedToken, onTokenSelect }) {
  const canvasRef = useRef(null);
  const seqLen = tokens.length;
  const cellSize = Math.min(40, 400 / seqLen);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !attention) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const pad = 60;
    const W = pad + seqLen*cellSize + 10;
    const H = pad + seqLen*cellSize + 10;
    canvas.width = W*dpr; canvas.height = H*dpr;
    canvas.style.width = W+'px'; canvas.style.height = H+'px';
    ctx.scale(dpr, dpr); ctx.clearRect(0, 0, W, H);

    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < seqLen; j++) {
        const val = attention[i][j];
        const x = pad + j*cellSize, y = pad + i*cellSize;
        ctx.fillStyle = heatmapColor(val);
        ctx.fillRect(x, y, cellSize-1, cellSize-1);
        if (cellSize > 25) {
          ctx.fillStyle = val > 0.3 ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.2)';
          ctx.font = `${Math.max(8, cellSize*0.25)}px JetBrains Mono`;
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(val.toFixed(2), x+cellSize/2, y+cellSize/2);
        }
      }
    }
    if (selectedToken !== null && selectedToken < seqLen) {
      ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2;
      ctx.strokeRect(pad, pad+selectedToken*cellSize, seqLen*cellSize, cellSize);
    }
    ctx.font = '10px Inter'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let i = 0; i < seqLen; i++) {
      const label = tokens[i].length > 6 ? tokens[i].slice(0,5)+'…' : tokens[i];
      ctx.fillStyle = i === selectedToken ? '#f59e0b' : 'rgba(146,149,179,0.5)';
      ctx.fillText(label, pad-5, pad+i*cellSize+cellSize/2);
    }
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    for (let j = 0; j < seqLen; j++) {
      ctx.save(); ctx.translate(pad+j*cellSize+cellSize/2, pad-5); ctx.rotate(-Math.PI/4);
      const label = tokens[j].length > 6 ? tokens[j].slice(0,5)+'…' : tokens[j];
      ctx.fillStyle = j === selectedToken ? '#f59e0b' : 'rgba(146,149,179,0.5)';
      ctx.fillText(label, 0, 0); ctx.restore();
    }
  }, [attention, tokens, selectedToken, cellSize]);

  const handleClick = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const y = e.clientY - rect.top - 60;
    const row = Math.floor(y / cellSize);
    if (row >= 0 && row < seqLen) onTokenSelect(row === selectedToken ? null : row);
  };
  return <canvas ref={canvasRef} onClick={handleClick} className="cursor-pointer" />;
}

function TokenArcs({ tokens, attention, selectedToken }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !attention || !tokens.length) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.offsetWidth; const H = 120;
    canvas.width = W*dpr; canvas.height = H*dpr; ctx.scale(dpr, dpr); ctx.clearRect(0, 0, W, H);
    const spacing = W / (tokens.length + 1); const baseY = H - 30;

    for (let i = 0; i < tokens.length; i++) {
      const x = spacing*(i+1); const boxW = Math.min(60, spacing-10);
      ctx.fillStyle = selectedToken === i ? 'rgba(245,158,11,0.15)' : 'rgba(139,92,246,0.06)';
      ctx.strokeStyle = selectedToken === i ? 'rgba(245,158,11,0.4)' : 'rgba(139,92,246,0.15)';
      ctx.lineWidth = 1; ctx.beginPath(); ctx.roundRect(x-boxW/2, baseY-10, boxW, 20, 4); ctx.fill(); ctx.stroke();
      ctx.fillStyle = selectedToken === i ? '#f59e0b' : 'rgba(232,232,240,0.7)';
      ctx.font = '10px Inter'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(tokens[i].length > 7 ? tokens[i].slice(0,6)+'…' : tokens[i], x, baseY);
    }

    if (selectedToken !== null && selectedToken < tokens.length) {
      const fromX = spacing*(selectedToken+1);
      for (let j = 0; j < tokens.length; j++) {
        const attn = attention[selectedToken][j]; if (attn < 0.05) continue;
        const toX = spacing*(j+1); const arcH = Math.abs(selectedToken-j)*15+20;
        ctx.beginPath(); ctx.moveTo(fromX, baseY-12);
        ctx.quadraticCurveTo((fromX+toX)/2, baseY-12-arcH, toX, baseY-12);
        ctx.strokeStyle = `rgba(139,92,246,${attn})`; ctx.lineWidth = 1+attn*4; ctx.stroke();
        if (attn > 0.1) {
          ctx.fillStyle = `rgba(139,92,246,${Math.min(1, attn+0.3)})`;
          ctx.font = '8px JetBrains Mono'; ctx.textAlign = 'center';
          ctx.fillText(attn.toFixed(2), (fromX+toX)/2, baseY-12-arcH/2);
        }
      }
    }
  }, [tokens, attention, selectedToken]);
  return <canvas ref={canvasRef} className="w-full" style={{ height: 120 }} />;
}

export default function AttentionLab() {
  const [inputText, setInputText] = useState('The cat sat on the mat and looked at the bird');
  const [temperature, setTemperature] = useState(1.0);
  const [numHeads, setNumHeads] = useState(4);
  const [activeHead, setActiveHead] = useState(-1);
  const [selectedToken, setSelectedToken] = useState(null);

  const tokens = useMemo(() => inputText.trim().split(/\s+/).filter(Boolean), [inputText]);
  const result = useMemo(() => tokens.length < 2 ? null : computeAttention(tokens, 32, numHeads, temperature), [tokens, numHeads, temperature]);
  const attn = result ? (activeHead === -1 ? result.combinedAttention : result.heads[activeHead]?.attention) : null;

  const presets = [
    'The cat sat on the mat and looked at the bird',
    'Machine learning models can understand language',
    'I went to the bank to deposit money yesterday',
    'The transformer architecture uses self attention',
  ];

  return (
    <div className="h-full flex flex-col">
      <div className="module-header">
        <div className="flex items-center gap-3">
          <div className="module-icon" style={{ background: 'linear-gradient(135deg, #fb7185, #f472b6)' }}>
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div>
            <h2 className="text-headline text-base text-text-primary">Attention Lab</h2>
            <p className="text-[11px] text-text-tertiary mt-0.5">How transformers decide what to pay attention to</p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-auto p-5 space-y-5">
        {/* Input */}
        <div>
          <label className="text-label">Input Sentence</label>
          <input type="text" value={inputText} onChange={(e) => setInputText(e.target.value)}
            className="control-input mt-2 text-mono" placeholder="Type a sentence..." />
          <div className="mt-2 flex gap-2 flex-wrap">
            {presets.map((p, i) => (
              <button key={i} onClick={() => setInputText(p)}
                className="text-[10px] px-2.5 py-1 rounded-lg sf-deep text-text-ghost hover:text-text-secondary transition-colors">
                {p.slice(0, 30)}...
              </button>
            ))}
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-6 flex-wrap">
          <div>
            <label className="text-[10px] text-text-tertiary flex items-center gap-1">
              <Thermometer className="w-3 h-3" /> Temperature
              <span className="text-mono text-nf-rose ml-1">{temperature.toFixed(1)}</span>
            </label>
            <input type="range" min="0.1" max="5" step="0.1" value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))} className="w-40" />
          </div>
          <div>
            <label className="text-[10px] text-text-tertiary">Heads</label>
            <select value={numHeads} onChange={(e) => setNumHeads(parseInt(e.target.value))}
              className="control-select ml-2 text-xs">
              {[1,2,4,8].map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
          <div>
            <label className="text-[10px] text-text-tertiary">View</label>
            <div className="flex gap-1 mt-0.5">
              <button onClick={() => setActiveHead(-1)}
                className={`px-2 py-1 rounded text-[10px] ${activeHead===-1 ? 'sf-raised text-nf-rose' : 'text-text-ghost'}`}>
                Combined
              </button>
              {result?.heads.map((_,i) => (
                <button key={i} onClick={() => setActiveHead(i)}
                  className={`px-2 py-1 rounded text-[10px] ${activeHead===i ? 'sf-raised text-nf-rose' : 'text-text-ghost'}`}>
                  H{i+1}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Token arcs */}
        {attn && (
          <div className="viz-container p-3">
            <div className="text-xs font-semibold text-text-secondary mb-2 flex items-center gap-1.5">
              <Eye className="w-3 h-3 text-nf-rose" /> Token Attention Flow
              <span className="text-[10px] text-text-ghost font-normal">(click a token in the heatmap)</span>
            </div>
            <TokenArcs tokens={tokens} attention={attn} selectedToken={selectedToken} />
          </div>
        )}

        {/* Heatmap */}
        {attn && (
          <div className="viz-container p-3 inline-block">
            <AttentionHeatmap attention={attn} tokens={tokens} selectedToken={selectedToken} onTokenSelect={setSelectedToken} />
          </div>
        )}

        {/* How it works */}
        <div className="callout callout-rose">
          <h4 className="text-xs font-bold text-nf-rose mb-2">🔍 How Self-Attention Works</h4>
          <div className="grid grid-cols-4 gap-3 text-[10px] text-text-secondary">
            {[
              { s:'1', t:'Embed', d:'Each token → a vector of numbers' },
              { s:'2', t:'Q × Kᵀ', d:'Compute similarity between all pairs' },
              { s:'3', t:'Softmax', d:'Normalize scores to probabilities' },
              { s:'4', t:'× V', d:'Weight values by attention scores' },
            ].map(s => (
              <div key={s.s} className="p-2 rounded-lg sf-deep">
                <div className="text-nf-rose font-bold text-sm mb-0.5">Step {s.s}</div>
                <div className="text-text-primary text-[11px] font-semibold text-mono">{s.t}</div>
                <div className="mt-0.5">{s.d}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
