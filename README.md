# NeuroForge

NeuroForge is a professional-grade neural network simulator and visualization platform that runs entirely in your browser. Unlike typical educational demos that animate pre-recorded sequences, NeuroForge performs **real computations**—actual matrix multiplications, actual gradient calculations, and actual backpropagation—and renders every intermediate result as an interactive, cinematic visualization.

The goal is to make the invisible computations of modern deep learning visible, tangible, and deeply intuitive. Every neuron activation, every weight update, every gradient flow is computed in real-time and visualized with a semantic color system where every color carries mathematical meaning.

## Core Features

- **Network Architect**: Build neural network architectures visually. Add or remove hidden layers, adjust neuron counts, and select activation functions (ReLU, Sigmoid, Tanh, GELU, Swish, Leaky ReLU) in real-time.
- **Training Arena**: Train your custom architectures on 2D datasets using real gradient descent. Watch the decision boundary evolve as weights and biases update continuously across the screen.
- **CNN Digit Recognition**: Draw a digit (0-9) and watch a real Convolutional Neural Network process it layer by layer in real-time. Features PyTorch-equivalent MNIST preprocessing and highly-optimized inference logic running locally in JS.
- **Loss Landscape Explorer**: Visualize the optimization journey as a particle navigating a highly non-convex, 3D loss terrain using physics-based momentum.
- **Interactive Playground**: Explore the mathematical concepts behind deep learning, including activation function shapes and regularization techniques, directly in the browser.

## Tech Stack

- **React + Vite**: Core framework and build tool for a blazing-fast user interface.
- **Zustand**: Lightweight and scalable state management for storing complex network architectures and weights.
- **Three.js & React Three Fiber (R3F)**: Powering the stunning, cinematic 3D visualizations, bloom effects, and network graphs.
- **Vanilla JavaScript Math Engine**: High-performance, zero-dependency neural network engine built from scratch utilizing flat `Float32Array` buffers for maximum inference speed.
- **Lucide React**: Beautiful, crisp, and consistent iconography.
- **Custom Design System**: Bespoke "dark glassmorphism" aesthetic with a highly intentional, semantic color palette styled entirely with vanilla CSS.

## Getting Started

To run the project locally on your machine:

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm run dev
   ```

3. **Explore**:
   Open `http://localhost:5173` (or the port specified in your terminal) in your browser to start exploring NeuroForge.

## Security & Privacy
NeuroForge runs entirely in the client's browser. No data, canvas drawings, or model weights are sent to any external server. The repository `.gitignore` has been rigorously configured to prevent the accidental exposure of environment variables (`.env`) and intermediate compilation files.
