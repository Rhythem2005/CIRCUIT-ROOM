// NeuroForge — Semantic Color System
// Every color carries meaning. Nothing is decorative.

// Activation intensity: cold (inactive) → hot (highly activated)
// Maps a normalized activation value [0, 1] to an HSL color
export function activationColor(value, alpha = 1) {
  const clamped = Math.max(0, Math.min(1, value));
  // Deep blue (inactive) → violet → amber → white-hot
  const hue = 220 - clamped * 180; // 220 (blue) → 40 (amber)
  const sat = 40 + clamped * 50;
  const light = 20 + clamped * 60;
  return `hsla(${hue}, ${sat}%, ${light}%, ${alpha})`;
}

// Activation color as hex for Three.js
export function activationHex(value) {
  const clamped = Math.max(0, Math.min(1, value));
  const hue = 220 - clamped * 180;
  const sat = 40 + clamped * 50;
  const light = 20 + clamped * 60;
  return hslToHex(hue, sat, light);
}

// Weight color: negative (rose) ↔ zero (dim) ↔ positive (blue)
export function weightColor(weight, alpha = 1) {
  const clamped = Math.max(-1, Math.min(1, weight));
  if (clamped >= 0) {
    const t = clamped;
    return `rgba(${Math.round(59 + (147 - 59) * (1 - t))}, ${Math.round(130 + (197 - 130) * (1 - t))}, ${Math.round(246)}, ${alpha * (0.15 + t * 0.85)})`;
  } else {
    const t = -clamped;
    return `rgba(${Math.round(244)}, ${Math.round(63 + (180 - 63) * (1 - t))}, ${Math.round(94 + (200 - 94) * (1 - t))}, ${alpha * (0.15 + t * 0.85)})`;
  }
}

export function weightHex(weight) {
  const clamped = Math.max(-1, Math.min(1, weight));
  if (clamped >= 0) return lerpHex('#1a1a2e', '#3b82f6', clamped);
  else return lerpHex('#1a1a2e', '#f43f5e', -clamped);
}

// Gradient magnitude: dim (vanishing) → bright (healthy) → red (exploding)
export function gradientColor(magnitude, maxMagnitude = 1) {
  const t = Math.min(magnitude / Math.max(maxMagnitude, 0.001), 2);
  if (t < 0.1) return '#1a1a2e';     // Dead
  if (t < 0.5) return '#6366f1';      // Weak
  if (t < 1.0) return '#8b5cf6';      // Healthy
  if (t < 1.5) return '#f59e0b';      // Strong
  return '#ef4444';                    // Exploding
}

// Layer type colors
export const LAYER_COLORS = {
  input: { primary: '#06b6d4', dim: '#06b6d433', label: 'Input' },
  hidden: { primary: '#8b5cf6', dim: '#8b5cf633', label: 'Hidden' },
  output: { primary: '#f59e0b', dim: '#f59e0b33', label: 'Output' },
  conv: { primary: '#3b82f6', dim: '#3b82f633', label: 'Conv' },
  pool: { primary: '#06b6d4', dim: '#06b6d433', label: 'Pool' },
  dense: { primary: '#8b5cf6', dim: '#8b5cf633', label: 'Dense' },
  dropout: { primary: '#6b7280', dim: '#6b728033', label: 'Dropout' },
  batchnorm: { primary: '#10b981', dim: '#10b98133', label: 'BatchNorm' },
  activation: { primary: '#f43f5e', dim: '#f43f5e33', label: 'Activation' },
};

// Classification colors — 8 maximally distinguishable classes
export const CLASS_PALETTE = [
  '#7c3aed', // Violet
  '#f59e0b', // Amber  
  '#06b6d4', // Cyan
  '#f43f5e', // Rose
  '#10b981', // Emerald
  '#3b82f6', // Blue
  '#ec4899', // Pink
  '#84cc16', // Lime
];

export const CLASS_PALETTE_DIM = CLASS_PALETTE.map(c => c + '22');
export const CLASS_PALETTE_MED = CLASS_PALETTE.map(c => c + '55');

// UI surface palette (layered depth)
export const SURFACE = {
  base: '#06070e',
  raised: '#0c0d1a',
  overlay: '#111327',
  elevated: '#181b36',
  accent: '#1e2144',
  border: 'rgba(139, 92, 246, 0.08)',
  borderHover: 'rgba(139, 92, 246, 0.18)',
  borderActive: 'rgba(139, 92, 246, 0.35)',
};

// Utility conversions
function hslToHex(h, s, l) {
  s /= 100; l /= 100;
  const a = s * Math.min(l, 1 - l);
  const f = n => {
    const k = (n + h / 30) % 12;
    const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    return Math.round(255 * color).toString(16).padStart(2, '0');
  };
  return `#${f(0)}${f(8)}${f(4)}`;
}

function lerpHex(a, b, t) {
  const parse = (hex) => [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16)
  ];
  const [ar, ag, ab] = parse(a);
  const [br, bg, bb] = parse(b);
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const bl = Math.round(ab + (bb - ab) * t);
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${bl.toString(16).padStart(2, '0')}`;
}

// Value-to-color ramp (for heatmaps)
export function heatmapColor(t, alpha = 1) {
  // 0 = deep navy, 0.25 = indigo, 0.5 = violet, 0.75 = amber, 1.0 = white
  const clamped = Math.max(0, Math.min(1, t));
  let r, g, b;
  if (clamped < 0.25) {
    const s = clamped / 0.25;
    r = lerp(10, 55, s); g = lerp(12, 30, s); b = lerp(30, 130, s);
  } else if (clamped < 0.5) {
    const s = (clamped - 0.25) / 0.25;
    r = lerp(55, 139, s); g = lerp(30, 92, s); b = lerp(130, 246, s);
  } else if (clamped < 0.75) {
    const s = (clamped - 0.5) / 0.25;
    r = lerp(139, 245, s); g = lerp(92, 158, s); b = lerp(246, 11, s);
  } else {
    const s = (clamped - 0.75) / 0.25;
    r = lerp(245, 255, s); g = lerp(158, 255, s); b = lerp(11, 255, s);
  }
  return `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, ${alpha})`;
}

function lerp(a, b, t) { return a + (b - a) * t; }
