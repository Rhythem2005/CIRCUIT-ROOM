// Circuit Room — Documentation Page

import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import {
  BookOpen, Brain, Play, Layers, Mountain, Sparkles, Beaker, PenTool,
  Palette, AlertTriangle, Boxes,
} from 'lucide-react';

/* ---------------------------------------------------------------------
   Content model: each section is a list of typed blocks instead of one
   markdown-flavored string. Lets sections actually differ in shape —
   a code block here, a plain list there — instead of forcing every
   module through the same "What you see / How to use it" template.
--------------------------------------------------------------------- */

const sections = [
  {
    id: 'overview',
    icon: Brain,
    title: 'What is Circuit Room?',
    color: '#8b5cf6',
    blocks: [
      { type: 'p', text: 'Circuit Room runs real computation in the browser — actual matrix multiplication, actual convolutions, actual backprop. Nothing you see is a pre-recorded animation standing in for the math; every module recomputes when you change an input.' },
      { type: 'p', text: 'One color convention holds across every module: color always encodes a specific quantity — activation strength, weight sign, gradient magnitude — never used decoratively. The legend below is the same one used everywhere else in the app.' },
    ],
  },
  {
    id: 'architect',
    icon: Brain,
    title: 'Network architect',
    color: '#7c3aed',
    blocks: [
      { type: 'h3', text: 'Building a network' },
      { type: 'p', text: 'Add or remove hidden layers, set neuron counts per layer from 1 to 64, and pick a per-layer activation: ReLU, Sigmoid, Tanh, GELU, Swish, or Leaky ReLU.' },
      { type: 'h3', text: 'Forward pass' },
      { type: 'p', text: 'Forward pass sends one 2D input through the network and animates it as particles traveling down each connection — so you can watch f(w·x + b) happen at every neuron, one layer at a time, instead of just seeing the final output.' },
      { type: 'h3', text: 'Reading the connections' },
      { type: 'p', text: 'Connection color is weight sign — blue is positive, red is negative — and thickness scales with magnitude. Neuron brightness scales with activation value. Layer color is fixed by role: input is cyan, hidden layers are violet, output is amber.' },
    ],
  },
  {
    id: 'training',
    icon: Play,
    title: 'Training arena',
    color: '#f59e0b',
    blocks: [
      { type: 'h3', text: 'What\u2019s actually running' },
      { type: 'p', text: 'This is real gradient descent, not a scripted playback. Loss gets computed, gradients get backpropagated, and weights get updated on every step, visualized live as it happens.' },
      { type: 'h3', text: 'Decision boundary' },
      { type: 'p', text: 'The heatmap is the network\u2019s current classification surface, recomputed as training progresses. Watch it sharpen from near-random noise into clean, separated regions.' },
      { type: 'h3', text: 'Gradient flow' },
      { type: 'p', text: 'Horizontal bars show gradient magnitude at each layer. Roughly even bars across all layers is healthy training. Bars that shrink toward the early layers are vanishing gradients — the classic symptom of a deep network struggling to update its earliest weights.' },
      { type: 'h3', text: 'Hyperparameters' },
      {
        type: 'list', items: [
          'Learning rate — log scale, 0.0001 to 0.3',
          'Optimizer — Adam, SGD, SGD with momentum, or RMSProp',
          'Batch size and training speed',
        ]
      },
      { type: 'p', text: 'All four are live — changing any of them mid-run takes effect on the next step, no restart needed.' },
    ],
  },
  {
    id: 'cnn',
    icon: PenTool,
    title: 'CNN digit recognition',
    color: '#ec4899',
    blocks: [
      { type: 'p', text: 'Draw a digit and it runs through an actual convolutional network, layer by layer, in real time.' },
      { type: 'h3', text: 'Architecture' },
      { type: 'code', text: 'Conv(1\u21928, 3\u00d73) \u2192 ReLU \u2192 MaxPool(2\u00d72)\n  \u2192 Conv(8\u219216, 3\u00d73) \u2192 ReLU \u2192 MaxPool(2\u00d72)\n  \u2192 Flatten(400) \u2192 Dense(32) \u2192 ReLU \u2192 Dense(10) \u2192 Softmax' },
      { type: 'p', text: 'The 280\u00d7280 canvas gets downsampled to 28\u00d728 \u2014 MNIST\u2019s native resolution \u2014 with a single canvas draw call, then read back as 784 grayscale values between 0 and 1.' },
      { type: 'h3', text: 'Feature maps' },
      { type: 'p', text: 'Conv layer 1\u2019s eight filters respond to simple, local patterns: edges and gradients at different orientations. Conv layer 2\u2019s sixteen filters take all eight of those maps as input, which is what lets them respond to compound shapes \u2014 corners, curves \u2014 rather than a single edge direction. Hover any tile in the feature map grid to see which filter produced it.' },
      { type: 'h3', text: 'Reading the pipeline' },
      { type: 'p', text: 'Click through any stage in the strip at the top. Heatmap color follows the same activation-strength ramp as the rest of the app \u2014 dark is inactive, violet is moderate, amber and white are the strongest responses. The final bars are softmax probabilities across digits 0\u20139; the highest is boxed in amber.' },
      {
        type: 'callout', tone: 'warning', title: 'Known limitations', text:
          'Two things this module doesn\u2019t do yet, both standard in real MNIST pipelines. First, there\u2019s no bounding-box crop or center-of-mass recentering before the downsample \u2014 a digit drawn small or off-center in the canvas will predict noticeably worse than one that fills it. Second, preprocessing is plain pixel/255; if the loaded weights were trained with z-score normalization (mean 0.1307, std 0.3081, MNIST\u2019s standard), that mismatch alone can throw predictions off independent of everything else being correct. Draw digits large and centered until both are fixed.'
      },
    ],
  },
  {
    id: 'representations',
    icon: Layers,
    title: 'Representation explorer',
    color: '#06b6d4',
    blocks: [
      { type: 'h3', text: 'Reading the plots' },
      { type: 'p', text: 'Same dataset, different lens each time. The leftmost plot is raw 2D input colored by true class. Every plot after it is that layer\u2019s activations for the same points \u2014 if a layer is doing its job, points sharing a class start landing closer together as you move deeper.' },
      { type: 'h3', text: 'Building a timeline' },
      { type: 'p', text: 'Train a few epochs, take a snapshot, repeat. The timeline slider scrubs through those snapshots, so instead of seeing only the trained end-state you can watch the clusters actually separate over the course of training.' },
      { type: 'h3', text: 'The point of this module' },
      { type: 'p', text: 'Classification is learning a representation where classes become linearly separable. A well-trained network\u2019s final-layer plot should show clean, separated point clouds even when the original input data was hopelessly entangled.' },
    ],
  },
  {
    id: 'landscape',
    icon: Mountain,
    title: 'Loss landscape explorer',
    color: '#10b981',
    blocks: [
      { type: 'h3', text: 'The surface' },
      { type: 'p', text: 'Loss as a function of two parameters, rendered as terrain. Valleys are good parameter regions, peaks are bad ones, and elevation is colored from dark (low loss) through violet to amber (high loss).' },
      { type: 'h3', text: 'Trajectories' },
      {
        type: 'list', items: [
          'SGD (red) \u2014 direct steps, prone to oscillating in narrow valleys',
          'Adam (violet) \u2014 adapts step size per parameter, generally the most efficient path',
          'SGD + momentum (amber) \u2014 carries inertia through flat regions',
          'RMSProp (cyan) \u2014 normalizes by a running average of gradient magnitude',
        ]
      },
      { type: 'h3', text: 'Presets' },
      { type: 'p', text: 'Simple bowl (convex, no tricks), saddle point (traps optimizers that don\u2019t escape flat curvature), ravine (narrow and steep \u2014 punishes plain SGD), multiple minima (starting point decides the outcome), Rosenbrock valley (the classic banana-shaped stress test), and Beale function (irregular, multiple local structures). Rotate and zoom freely; learning rate and step count are both live.' },
    ],
  },
  {
    id: 'attention',
    icon: Sparkles,
    title: 'Attention lab',
    color: '#f43f5e',
    blocks: [
      { type: 'h3', text: 'The heatmap' },
      { type: 'p', text: 'Rows are query tokens, columns are key tokens, brightness is attention score. Type your own sentence \u2014 it tokenizes and runs real self-attention over your exact text, not a canned example.' },
      { type: 'h3', text: 'Token arcs' },
      { type: 'p', text: 'Click a token and its outgoing attention becomes a set of curved arcs to every other token, arc thickness scaled to weight. It\u2019s the same information as one row of the heatmap \u2014 just easier to follow for a single token in isolation.' },
      { type: 'h3', text: 'Temperature' },
      { type: 'p', text: 'This is the literal softmax temperature. Low temperature collapses attention onto one dominant token; high temperature flattens it toward uniform across all tokens. The multi-head toggle switches between an individual head and the combined view \u2014 heads specialize in different relationships, so they rarely look alike.' },
    ],
  },
  {
    id: 'playground',
    icon: Beaker,
    title: 'Concept playground',
    color: '#6366f1',
    blocks: [
      { type: 'p', text: 'Four separate micro-labs, each isolating one concept rather than one big combined demo:' },
      {
        type: 'deflist', items: [
          { term: 'Activation functions', desc: 'Compare 7+ functions on one shared graph, toggle derivatives to see gradient behavior, and drag a test-input slider to evaluate any function at any point. Each card shows its formula alongside where it actually gets used in practice.' },
          { term: 'Convolution visualizer', desc: 'Step a 3\u00d73 kernel across an 8\u00d78 image one position at a time, watching the element-wise multiply and sum at each stop. Presets: edge detection, blur, sharpen, Sobel, Laplacian.' },
          { term: 'Gradient flow analyzer', desc: 'Stack up to 50 layers and watch gradients vanish under sigmoid, stay healthy under ReLU, and recover once you add skip connections or batch normalization.' },
          { term: 'Regularization lab', desc: 'Two identical networks train side by side on the same data, one regularized. Watch the unregularized one memorize noise while the other\u2019s decision boundary stays smooth.' },
        ]
      },
    ],
  },
];

const colorPrinciples = [
  { label: 'Activation strength', desc: 'Deep blue \u2192 violet \u2192 amber \u2192 white', preview: ['#0a0c1e', '#6366f1', '#8b5cf6', '#f59e0b', '#ffffff'] },
  { label: 'Weight sign', desc: 'Red is negative, blue is positive', preview: ['#f43f5e', '#1a1a2e', '#3b82f6'] },
  { label: 'Gradient health', desc: 'Dim is vanishing, bright is healthy, red is exploding', preview: ['#1a1a2e', '#6366f1', '#8b5cf6', '#f59e0b', '#ef4444'] },
  { label: 'Layer role', desc: 'Input is cyan, hidden is violet, output is amber, conv is blue', preview: ['#06b6d4', '#8b5cf6', '#f59e0b', '#3b82f6'] },
];

const stack = [
  { group: '3D rendering', items: ['react-three-fiber', '@react-three/drei (OrbitControls, Line, Html)', '@react-three/postprocessing (Bloom)', 'three.js r128'] },
  { group: '2D rendering', items: ['Canvas 2D \u2014 feature-map heatmaps, the digit-drawing surface, kernel previews'] },
  { group: 'Motion', items: ['Framer Motion \u2014 stage transitions and the drawing-canvas scan sweep'] },
  { group: 'State + data viz', items: ['Zustand', 'Recharts'] },
  { group: 'Icons', items: ['lucide-react'] },
];

/* ---------------------------------------------------------------------
   Block renderer
--------------------------------------------------------------------- */
function Block({ block }) {
  switch (block.type) {
    case 'p':
      return <p className="text-xs text-text-secondary leading-[1.75] mb-3 last:mb-0">{block.text}</p>;
    case 'h3':
      return <h3 className="text-xs font-semibold text-text-primary mt-4 mb-2 first:mt-0">{block.text}</h3>;
    case 'code':
      return (
        <pre className="text-mono text-[11px] text-text-secondary bg-black/30 rounded-lg px-3 py-2.5 mb-3 overflow-x-auto leading-[1.7] whitespace-pre">
          {block.text}
        </pre>
      );
    case 'list':
      return (
        <ul className="mb-3 space-y-1.5">
          {block.items.map((item, i) => (
            <li key={i} className="text-xs text-text-secondary leading-[1.6] flex gap-2">
              <span className="text-text-ghost flex-shrink-0">{'\u2013'}</span>
              <span>{item}</span>
            </li>
          ))}
        </ul>
      );
    case 'deflist':
      return (
        <dl className="space-y-3">
          {block.items.map((item, i) => (
            <div key={i}>
              <dt className="text-xs font-semibold text-text-primary mb-0.5">{item.term}</dt>
              <dd className="text-xs text-text-secondary leading-[1.7]">{item.desc}</dd>
            </div>
          ))}
        </dl>
      );
    case 'callout':
      return (
        <div
          className="rounded-xl px-4 py-3 mt-3 mb-1 flex gap-2.5"
          style={{ background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.2)' }}
        >
          <AlertTriangle className="w-3.5 h-3.5 text-nf-amber flex-shrink-0 mt-0.5" />
          <div>
            <div className="text-xs font-semibold mb-1" style={{ color: '#f59e0b' }}>{block.title}</div>
            <p className="text-xs text-text-secondary leading-[1.7] m-0">{block.text}</p>
          </div>
        </div>
      );
    default:
      return null;
  }
}

/* ---------------------------------------------------------------------
   Sidebar nav with scroll-spy via IntersectionObserver
--------------------------------------------------------------------- */
function SideNav({ activeId, onNavigate }) {
  return (
    <nav className="hidden lg:block w-52 flex-shrink-0 sticky top-0 self-start pt-10 pl-8">
      <div className="text-label mb-3 px-2">On this page</div>
      <ul className="space-y-0.5">
        {sections.map((s) => (
          <li key={s.id}>
            <a
              href={`#${s.id}`}
              onClick={(e) => { e.preventDefault(); onNavigate(s.id); }}
              className="block px-2 py-1.5 rounded-lg text-[12px] transition-colors"
              style={{
                color: activeId === s.id ? s.color : 'var(--text-tertiary, rgba(255,255,255,0.5))',
                background: activeId === s.id ? s.color + '14' : 'transparent',
                fontWeight: activeId === s.id ? 600 : 400,
              }}
            >
              {s.title}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}

export default function Documentation() {
  const [activeId, setActiveId] = useState(sections[0].id);
  const sectionRefs = useRef({});

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting);
        if (visible.length > 0) {
          const topMost = visible.reduce((a, b) => (a.boundingClientRect.top < b.boundingClientRect.top ? a : b));
          setActiveId(topMost.target.id);
        }
      },
      { rootMargin: '-15% 0px -70% 0px', threshold: 0 }
    );
    Object.values(sectionRefs.current).forEach((el) => el && observer.observe(el));
    return () => observer.disconnect();
  }, []);

  const scrollTo = (id) => {
    sectionRefs.current[id]?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto flex">
        <SideNav activeId={activeId} onNavigate={scrollTo} />

        <div className="flex-1 min-w-0 px-8 py-10">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="mb-12"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="module-icon" style={{ background: 'linear-gradient(135deg, #94a3b8, #64748b)', width: 36, height: 36 }}>
                <BookOpen className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-display text-2xl text-text-primary">Documentation</h1>
                <p className="text-sm text-text-tertiary mt-0.5">What every visualization in Circuit Room actually means</p>
              </div>
            </div>
          </motion.div>

          {/* Color system */}
          <div className="mb-10 p-5 rounded-2xl sf-raised">
            <div className="flex items-center gap-2 mb-4">
              <Palette className="w-4 h-4 text-nf-violet" />
              <h2 className="text-headline text-sm text-text-primary">Color system</h2>
            </div>
            <p className="text-xs text-text-secondary mb-4 leading-relaxed">
              Every color in Circuit Room encodes a specific mathematical quantity. Nothing is decorative.
            </p>
            <div className="grid grid-cols-2 gap-3">
              {colorPrinciples.map((cp, i) => (
                <div key={i} className="p-3 rounded-xl sf-deep">
                  <div className="text-xs font-semibold text-text-primary mb-1">{cp.label}</div>
                  <div className="text-[10px] text-text-tertiary mb-2">{cp.desc}</div>
                  <div className="flex gap-1">
                    {cp.preview.map((c, j) => (
                      <div key={j} className="w-5 h-5 rounded-md" style={{ backgroundColor: c, border: '1px solid rgba(255,255,255,0.06)' }} />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Module sections */}
          <div className="space-y-6">
            {sections.map((section) => {
              const Icon = section.icon;
              return (
                <div
                  key={section.id}
                  id={section.id}
                  ref={(el) => { sectionRefs.current[section.id] = el; }}
                  className="rounded-2xl sf-raised overflow-hidden scroll-mt-6"
                >
                  <div className="px-5 py-4 flex items-center gap-3 border-b border-white/[0.04]">
                    <div className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0" style={{ backgroundColor: section.color + '20' }}>
                      <Icon className="w-4 h-4" style={{ color: section.color }} />
                    </div>
                    <h2 className="text-headline text-sm text-text-primary">{section.title}</h2>
                  </div>
                  <div className="px-5 py-4">
                    {section.blocks.map((block, bi) => <Block key={bi} block={block} />)}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Stack */}
          <div className="mt-8 mb-16 p-5 rounded-2xl sf-raised">
            <div className="flex items-center gap-2 mb-4">
              <Boxes className="w-4 h-4 text-nf-cyan" />
              <h2 className="text-headline text-sm text-text-primary">Stack</h2>
            </div>
            <div className="grid grid-cols-2 gap-4">
              {stack.map((group, i) => (
                <div key={i}>
                  <div className="text-[10px] text-text-ghost uppercase tracking-wider mb-1.5">{group.group}</div>
                  <ul className="space-y-1">
                    {group.items.map((item, j) => (
                      <li key={j} className="text-xs text-text-secondary text-mono leading-[1.6]">{item}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}