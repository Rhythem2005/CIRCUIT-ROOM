// NeuroForge — Redesigned Home Page

import { motion } from 'framer-motion';
import { Brain, Play, Layers, Mountain, Sparkles, Beaker, ArrowRight, Zap, PenTool, BookOpen, Cpu } from 'lucide-react';
import useStore from '../../stores/useStore';
import { useEffect, useRef } from 'react';

const features = [
  {
    id: 'architect', icon: Brain, title: 'Network Architect',
    description: 'Build networks visually. Watch data flow through every neuron.',
    color: '#7c3aed',
  },
  {
    id: 'training', icon: Play, title: 'Training Arena',
    description: 'Live backpropagation with evolving decision boundaries.',
    color: '#f59e0b',
  },
  {
    id: 'cnn', icon: PenTool, title: 'CNN Digits',
    description: 'Draw a digit — watch a CNN process it layer by layer.',
    color: '#ec4899', badge: 'New',
  },
  {
    id: 'representations', icon: Layers, title: 'Representations',
    description: 'See how layers untangle data into separable clusters.',
    color: '#06b6d4',
  },
  {
    id: 'landscape', icon: Mountain, title: 'Loss Landscape',
    description: '3D optimizer trajectories on real loss surfaces.',
    color: '#10b981',
  },
  {
    id: 'attention', icon: Sparkles, title: 'Attention Lab',
    description: 'Interactive transformer attention visualization.',
    color: '#f43f5e',
  },
  {
    id: 'playground', icon: Beaker, title: 'Playground',
    description: 'Activations, convolutions, gradients, regularization.',
    color: '#6366f1',
  },
  {
    id: 'docs', icon: BookOpen, title: 'Documentation',
    description: 'How to interpret every visualization in the platform.',
    color: '#64748b',
  },
];

// Ambient background animation
function AmbientNetwork() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animId;
    let time = 0;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvas.offsetWidth * dpr;
      canvas.height = canvas.offsetHeight * dpr;
      ctx.scale(dpr, dpr);
    };
    resize();

    const W = () => canvas.offsetWidth;
    const H = () => canvas.offsetHeight;

    // Nodes
    const numNodes = 60;
    const nodes = Array.from({ length: numNodes }, () => ({
      x: Math.random(),
      y: Math.random(),
      vx: (Math.random() - 0.5) * 0.0003,
      vy: (Math.random() - 0.5) * 0.0003,
      radius: 1 + Math.random() * 2,
      phase: Math.random() * Math.PI * 2,
    }));

    const animate = () => {
      time += 0.004;
      const w = W(), h = H();
      ctx.clearRect(0, 0, w, h);

      // Update nodes
      for (const n of nodes) {
        n.x += n.vx;
        n.y += n.vy;
        if (n.x < 0 || n.x > 1) n.vx *= -1;
        if (n.y < 0 || n.y > 1) n.vy *= -1;
      }

      // Draw connections
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = (nodes[i].x - nodes[j].x) * w;
          const dy = (nodes[i].y - nodes[j].y) * h;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            const alpha = (1 - dist / 120) * 0.06;
            ctx.beginPath();
            ctx.moveTo(nodes[i].x * w, nodes[i].y * h);
            ctx.lineTo(nodes[j].x * w, nodes[j].y * h);
            ctx.strokeStyle = `rgba(139, 92, 246, ${alpha})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      // Draw nodes
      for (const n of nodes) {
        const pulse = Math.sin(time * 2 + n.phase) * 0.5 + 0.5;
        const r = n.radius + pulse;
        ctx.beginPath();
        ctx.arc(n.x * w, n.y * h, r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(139, 92, 246, ${0.15 + pulse * 0.15})`;
        ctx.fill();
      }

      animId = requestAnimationFrame(animate);
    };

    animate();
    const handleResize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvas.offsetWidth * dpr;
      canvas.height = canvas.offsetHeight * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    window.addEventListener('resize', handleResize);
    return () => { cancelAnimationFrame(animId); window.removeEventListener('resize', handleResize); };
  }, []);

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />;
}

export default function HomePage() {
  const { setActiveModule } = useStore();

  return (
    <div className="h-full overflow-y-auto">
      {/* Hero */}
      <div className="relative min-h-[420px] flex items-center justify-center overflow-hidden">
        <AmbientNetwork />
        <div className="absolute inset-0" style={{
          background: 'linear-gradient(to bottom, transparent 0%, var(--color-abyss) 100%)'
        }} />

        <motion.div
          className="relative z-10 text-center px-8 max-w-2xl"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className="flex items-center justify-center gap-2 mb-5">
            <div className="h-px w-8 bg-gradient-to-r from-transparent to-nf-violet/30" />
            <span className="text-[10px] font-semibold text-nf-amber tracking-[0.2em] uppercase">Real Computation Engine</span>
            <div className="h-px w-8 bg-gradient-to-l from-transparent to-nf-violet/30" />
          </div>

          <h1 className="text-display text-5xl mb-4">
            <span className="grad-text">NeuroForge</span>
          </h1>

          <p className="text-sm text-text-secondary leading-relaxed max-w-md mx-auto mb-8">
            Build, train, and visualize neural networks with real computation.
            Every gradient is calculated. Every weight is updated. Nothing is faked.
          </p>

          <div className="flex items-center justify-center gap-3">
            <motion.button
              onClick={() => setActiveModule('architect')}
              className="btn-primary flex items-center gap-2 text-sm px-6 py-3"
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
            >
              <Brain className="w-4 h-4" />
              Start Building
              <ArrowRight className="w-3.5 h-3.5 opacity-60" />
            </motion.button>
            <motion.button
              onClick={() => setActiveModule('cnn')}
              className="btn-ghost flex items-center gap-2 text-sm px-5 py-[11px]"
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
            >
              <PenTool className="w-4 h-4" />
              Try CNN Digits
            </motion.button>
          </div>
        </motion.div>
      </div>

      {/* Feature Grid */}
      <div className="px-8 pb-10 max-w-4xl mx-auto -mt-4">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {features.map((feat, i) => {
            const Icon = feat.icon;
            return (
              <motion.button
                key={feat.id}
                onClick={() => setActiveModule(feat.id)}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.05 * i, duration: 0.4 }}
                className="text-left p-4 rounded-xl sf-raised hover:sf-elevated transition-all duration-200 group relative overflow-hidden"
                whileHover={{ y: -3 }}
              >
                {feat.badge && (
                  <span className="absolute top-2.5 right-2.5 text-[8px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-md"
                    style={{ backgroundColor: feat.color + '20', color: feat.color }}>
                    {feat.badge}
                  </span>
                )}
                <div className="w-8 h-8 rounded-lg flex items-center justify-center mb-3"
                  style={{ backgroundColor: feat.color + '15' }}>
                  <Icon className="w-4 h-4" style={{ color: feat.color }} />
                </div>
                <h3 className="text-xs font-bold text-text-primary mb-1 group-hover:text-white transition-colors">
                  {feat.title}
                </h3>
                <p className="text-[10px] text-text-tertiary leading-relaxed">{feat.description}</p>
              </motion.button>
            );
          })}
        </div>

        {/* Computation banner */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="mt-6 p-4 rounded-xl sf-deep flex items-center gap-6 justify-center"
        >
          {[
            { icon: Cpu, label: 'Real Math', value: 'Float64 Tensors' },
            { icon: Zap, label: 'Live Training', value: 'Actual Backprop' },
            { icon: Brain, label: 'Activations', value: '7+ Functions' },
          ].map((stat, i) => (
            <div key={i} className="flex items-center gap-2">
              <stat.icon className="w-3.5 h-3.5 text-nf-violet" />
              <div>
                <div className="text-[10px] text-text-ghost">{stat.label}</div>
                <div className="text-xs font-semibold text-text-secondary text-mono">{stat.value}</div>
              </div>
            </div>
          ))}
        </motion.div>
      </div>
    </div>
  );
}
