// NeuroForge — Documentation Page

import { motion } from 'framer-motion';
import { BookOpen, Brain, Play, Layers, Mountain, Sparkles, Beaker, PenTool, ExternalLink, Cpu, Palette, Zap } from 'lucide-react';

const sections = [
  {
    id: 'overview',
    icon: Brain,
    title: 'What is NeuroForge?',
    color: '#8b5cf6',
    content: `NeuroForge is a professional-grade neural network simulator and visualization platform that runs entirely in your browser. Unlike typical educational demos that animate pre-recorded sequences, NeuroForge performs real computations — actual matrix multiplications, actual gradient calculations, actual backpropagation — and renders every intermediate result as an interactive visualization.

The goal is to make the invisible computations of modern deep learning visible, tangible, and deeply intuitive. Every neuron activation, every weight update, every gradient flow is computed in real-time and visualized with a semantic color system where every color carries mathematical meaning.`
  },
  {
    id: 'architect',
    icon: Brain,
    title: 'Network Architect',
    color: '#7c3aed',
    content: `Build neural network architectures visually. Add or remove hidden layers, adjust neuron counts per layer (1–64), and select activation functions (ReLU, Sigmoid, Tanh, GELU, Swish, Leaky ReLU).

**What you see:** Neurons are rendered with activation-dependent intensity — brighter colors indicate higher activation values. Connections between neurons are colored by weight sign: blue for positive weights, red for negative weights, with thickness proportional to magnitude.

**Forward Pass:** Click "Forward Pass" to watch a single data point flow through the network. Particles travel through each connection, showing how input values are multiplied by weights, summed, and transformed by activation functions at each layer. This is the core computation: f(w·x + b).

**How to read it:** The input layer (cyan) receives raw 2D coordinates. Each hidden layer (violet) transforms the data through learned weights. The output layer (amber) produces class probabilities.`
  },
  {
    id: 'training',
    icon: Play,
    title: 'Training Arena',
    color: '#f59e0b',
    content: `Watch neural networks learn in real-time. The Training Arena runs actual gradient descent — computing loss, backpropagating gradients, and updating weights — while visualizing everything simultaneously.

**Decision Boundary:** The colored heatmap shows the network's current classification surface. Each pixel's color represents which class the network would predict for that point in input space. Watch it evolve from random to precise as training progresses.

**Loss & Accuracy Curves:** Real-time charts show how the loss function decreases and accuracy increases over training epochs. The gradient beneath the line indicates trend momentum.

**Gradient Flow:** The horizontal bars show gradient magnitude at each layer. Healthy training shows roughly uniform gradient magnitudes. If early layers show very small bars (vanishing gradients), the network is struggling to learn deep features.

**Hyperparameters:** All hyperparameters can be adjusted live: learning rate (log scale from 0.0001 to 0.3), optimizer (Adam, SGD, SGD+Momentum, RMSProp), batch size, and training speed. Changes take effect immediately.`
  },
  {
    id: 'cnn',
    icon: PenTool,
    title: 'CNN Digit Recognition',
    color: '#ec4899',
    content: `Draw a digit (0–9) on the canvas, and watch a convolutional neural network process your exact drawing through every layer in real-time.

**How it works:** Your 280×280 drawing is downsampled to 28×28 pixels (matching MNIST format) and fed through a CNN with the architecture: Conv(1→8, 3×3) → ReLU → MaxPool(2×2) → Conv(8→16, 3×3) → ReLU → MaxPool(2×2) → Dense(400→32) → ReLU → Dense(32→10) → Softmax.

**Feature Maps:** At each convolutional layer, you can see what each filter detects. Early filters (Conv 1) detect simple edges and lines. Later filters (Conv 2) combine those edges into higher-level features like curves and corners.

**Stage Navigation:** Click through each stage in the pipeline to see the intermediate representations. The heatmap colors use the semantic color ramp: dark = inactive, violet = moderate, amber = high activation, white = maximum.

**Output Probabilities:** The final bars show the softmax probabilities for each digit 0–9. The predicted digit is highlighted in amber.`
  },
  {
    id: 'representations',
    icon: Layers,
    title: 'Representation Explorer',
    color: '#06b6d4',
    content: `Visualize how each layer of the network transforms data from tangled, overlapping distributions into cleanly separated clusters — the fundamental mechanism of deep learning.

**What you see:** Each scatter plot shows the same dataset, but projected through the lens of a different layer. The input plot (leftmost) shows raw 2D coordinates with their true class colors. Each subsequent plot shows how that layer's activations rearrange the data.

**How to use it:** Train for a few epochs, then take a snapshot. Repeat to build a timeline. Use the timeline slider to scrub through training history and watch clusters form, separate, and align.

**The key insight:** Classification = learning a representation where classes become linearly separable. A well-trained network's final layer should show cleanly separated point clouds, even if the original data was intricately entangled.`
  },
  {
    id: 'landscape',
    icon: Mountain,
    title: 'Loss Landscape Explorer',
    color: '#10b981',
    content: `Navigate the 3D mathematical terrain that optimizers must traverse to find good model parameters.

**Surface:** The 3D surface shows how loss varies as a function of two parameter dimensions. Valleys represent good parameter regions; peaks represent poor ones. The color maps elevation: dark valleys → violet midrange → amber peaks.

**Optimizer Trajectories:** Colored paths show how different optimizers navigate the same landscape. SGD (red) takes direct steps and can oscillate. Adam (violet) adapts step sizes and navigates efficiently. SGD+Momentum (amber) carries inertia. RMSProp (cyan) normalizes per-parameter.

**Presets:** Choose from multiple landscape topologies — Simple Bowl (convex, easy), Saddle Point (traps naive optimizers), Ravine (challenges SGD), Multiple Minima (starting point matters), Rosenbrock Valley (banana-shaped challenge), Beale Function (complex surface).

**Interactive:** Rotate and zoom the 3D view. Adjust learning rate and step count to see how optimizer behavior changes dramatically.`
  },
  {
    id: 'attention',
    icon: Sparkles,
    title: 'Attention Lab',
    color: '#f43f5e',
    content: `Understand how transformer self-attention works by visualizing attention patterns on your own text.

**Attention Heatmap:** The matrix shows attention scores between every pair of tokens. Brighter cells = higher attention. Each row shows what a query token attends to; each column shows which tokens receive attention from others.

**Token Arcs:** When you click a token, curved arcs show what it attends to, with arc thickness proportional to attention weight. This reveals the dynamic routing of information in transformers.

**Temperature:** Adjusting temperature changes how "sharp" or "diffuse" attention is. Low temperature → one token dominates (hard attention). High temperature → attention spreads evenly across all tokens (uniform attention). This directly controls the softmax concentration.

**Multi-Head:** Toggle between individual attention heads and the combined view. Different heads learn to attend to different types of relationships.`
  },
  {
    id: 'playground',
    icon: Beaker,
    title: 'Concept Playground',
    color: '#6366f1',
    content: `Four interactive micro-labs exploring fundamental deep learning concepts:

**Activation Functions:** Compare 7+ activation functions on a shared graph. Toggle derivatives to see gradient flow characteristics. Use the test input slider to evaluate each function at any point. Each function card shows its formula, pros, cons, and practical usage.

**Convolution Visualizer:** Step through convolution kernel-by-kernel. Watch a 3×3 filter slide across an 8×8 image, showing element-wise multiplications, summation, and the resulting output feature map. Choose from preset kernels (edge detection, blur, sharpen, Sobel, Laplacian) and input patterns.

**Gradient Flow Analyzer:** Simulate gradient backpropagation through networks up to 50 layers deep. See vanishing gradients with sigmoid, healthy flow with ReLU, and how skip connections (ResNet) and batch normalization fix gradient degradation.

**Regularization Lab:** Train two identical networks side-by-side — one with regularization, one without — and watch how regularization prevents overfitting by producing smoother, more generalizable decision boundaries.`
  },
];

const colorPrinciples = [
  { label: 'Activation Strength', desc: 'Deep blue → violet → amber → white', preview: ['#0a0c1e', '#6366f1', '#8b5cf6', '#f59e0b', '#ffffff'] },
  { label: 'Weight Sign', desc: 'Red = negative, Blue = positive', preview: ['#f43f5e', '#1a1a2e', '#3b82f6'] },
  { label: 'Gradient Health', desc: 'Dim = vanishing, Bright = healthy, Red = exploding', preview: ['#1a1a2e', '#6366f1', '#8b5cf6', '#f59e0b', '#ef4444'] },
  { label: 'Layer Types', desc: 'Input=Cyan, Hidden=Violet, Output=Amber, Conv=Blue', preview: ['#06b6d4', '#8b5cf6', '#f59e0b', '#3b82f6'] },
];

export default function Documentation() {
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-3xl mx-auto px-8 py-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-12"
        >
          <div className="flex items-center gap-3 mb-4">
            <div className="module-icon" style={{ background: 'linear-gradient(135deg, #94a3b8, #64748b)', width: 36, height: 36 }}>
              <BookOpen className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-display text-2xl text-text-primary">Documentation</h1>
              <p className="text-sm text-text-tertiary mt-0.5">Understanding NeuroForge — what every visualization means</p>
            </div>
          </div>
        </motion.div>

        {/* Color System */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-10 p-5 rounded-2xl sf-raised"
        >
          <div className="flex items-center gap-2 mb-4">
            <Palette className="w-4 h-4 text-nf-violet" />
            <h2 className="text-headline text-sm text-text-primary">Color System</h2>
          </div>
          <p className="text-xs text-text-secondary mb-4 leading-relaxed">
            Every color in NeuroForge encodes a specific mathematical quantity. Nothing is decorative.
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
        </motion.div>

        {/* Module Sections */}
        <div className="space-y-6">
          {sections.map((section, i) => {
            const Icon = section.icon;
            return (
              <motion.div
                key={section.id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 + i * 0.05 }}
                className="rounded-2xl sf-raised overflow-hidden"
              >
                <div className="px-5 py-4 flex items-center gap-3 border-b border-white/[0.04]">
                  <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ backgroundColor: section.color + '20' }}>
                    <Icon className="w-4 h-4" style={{ color: section.color }} />
                  </div>
                  <h2 className="text-headline text-sm text-text-primary">{section.title}</h2>
                </div>
                <div className="px-5 py-4">
                  {section.content.split('\n\n').map((para, pi) => (
                    <p key={pi} className="text-xs text-text-secondary leading-[1.75] mb-3 last:mb-0"
                      dangerouslySetInnerHTML={{
                        __html: para
                          .replace(/\*\*(.*?)\*\*/g, '<strong class="text-text-primary font-semibold">$1</strong>')
                      }}
                    />
                  ))}
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* Technical Details */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mt-8 mb-16 p-5 rounded-2xl sf-raised"
        >
          <div className="flex items-center gap-2 mb-4">
            <Cpu className="w-4 h-4 text-nf-cyan" />
            <h2 className="text-headline text-sm text-text-primary">Technical Architecture</h2>
          </div>
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: 'Computation', value: 'Real', desc: 'Actual matrix ops, not animations' },
              { label: 'Tensor System', value: 'Custom', desc: 'Float64Array with autograd' },
              { label: 'Rendering', value: 'Canvas + Three.js', desc: '2D canvas & WebGL 3D' },
              { label: 'State', value: 'Zustand', desc: 'Minimal reactive store' },
              { label: 'Animation', value: 'Framer Motion', desc: 'Physics-based transitions' },
              { label: 'Charts', value: 'Recharts', desc: 'Composable React charts' },
            ].map((item, i) => (
              <div key={i} className="p-3 rounded-xl sf-deep">
                <div className="text-[10px] text-text-ghost uppercase tracking-wider">{item.label}</div>
                <div className="text-sm font-bold text-text-primary mt-0.5 text-mono">{item.value}</div>
                <div className="text-[10px] text-text-tertiary mt-0.5">{item.desc}</div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
