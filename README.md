# Circuit Room

> **A browser-native neural network simulator that transforms deep learning computations into interactive, real-time 3D visualizations.**

Circuit Room is a professional-grade neural network visualization platform that runs entirely in the browser. Unlike traditional educational demonstrations that replay precomputed animations, Circuit Room performs **real neural network computation** in real time—including forward propagation, matrix multiplication, gradient computation, and backpropagation—and visualizes every stage of the learning process.

The project is designed to make the mathematics behind modern deep learning intuitive and observable. Every neuron activation, weight update, and gradient flow is computed live and rendered as fully interactive 3D geometry with physically inspired lighting and a semantic color system, allowing users to understand *why* a network behaves the way it does rather than simply seeing the final prediction.

---

## Highlights

* **Real computation** — No prerecorded animations or fake simulations. Every visualization is generated from live neural network computation.
* **Interactive 3D rendering** — Explore neural networks as dynamic 3D structures with cinematic lighting, animation, and camera controls.
* **Built entirely for the browser** — No backend or server-side inference required.
* **Educational and technically accurate** — Designed to bridge the gap between mathematical theory and intuitive understanding.

---

# Features

## Network Architect

Design and explore fully customizable feedforward neural networks in an immersive 3D environment.

Features include:

* Interactive visual architecture editor
* Adjustable hidden layers and neuron counts
* Multiple activation functions:

  * ReLU
  * Sigmoid
  * Tanh
  * GELU
  * Swish
  * Leaky ReLU
* Live forward propagation visualization
* Weight-aware animated connections
* Activation-driven neuron pulses
* Orbit, pan, and inspect the network from any angle

Each neuron is represented as a glowing energy core enclosed by a rotating shell, while connections dynamically change color and intensity according to their weight sign and magnitude.

---

## Training Arena

Train custom neural networks using real gradient descent while observing the learning process in real time.

Visualizations include:

* Forward propagation
* Backpropagation
* Live gradient updates
* Weight and bias evolution
* Continuously updating decision boundaries
* Interactive 2D datasets

Rather than visualizing a static result, Circuit Room exposes the complete optimization process as it happens.

---

## CNN Digit Recognition

Draw a handwritten digit and observe a real Convolutional Neural Network process it layer by layer.

The visualization includes:

* Live convolution operations
* Individual feature maps for every filter
* Progressive activation visualization
* Flattened feature vectors
* Final classification probabilities

The implementation uses PyTorch-equivalent MNIST preprocessing while running inference entirely in optimized JavaScript.

---

## Loss Landscape Explorer

Visualize optimization as movement across a complex three-dimensional loss surface.

Features include:

* Physics-inspired optimization
* Momentum-based movement
* Interactive camera controls
* Non-convex terrain visualization
* Optimization path tracking

This module provides an intuitive understanding of gradient descent dynamics beyond traditional 2D plots.

---

## Interactive Playground

Experiment with fundamental deep learning concepts directly in the browser.

Topics include:

* Activation functions
* Regularization techniques
* Mathematical intuition behind neural networks
* Interactive parameter exploration

---

# Technology Stack

### Frontend

* React
* Vite

### State Management

* Zustand

### 3D Graphics

* Three.js
* React Three Fiber (R3F)
* @react-three/postprocessing

These technologies power every visualization in Circuit Room, including neurons, weighted connections, particle systems, feature maps, and post-processing effects such as bloom and vignette.

### Neural Network Engine

A custom neural network engine built from scratch in vanilla JavaScript featuring:

* Zero runtime dependencies
* Flat `Float32Array` memory layouts
* Optimized matrix operations
* Real forward propagation
* Real backpropagation
* Efficient gradient computation

### Animation

* Framer Motion

Used throughout the interface for smooth transitions, panel animations, and navigation.

### Icons

* Lucide React

### Design System

A fully custom design language featuring:

* Dark glassmorphism aesthetic
* Semantic color system
* Responsive layouts
* Handcrafted CSS
* High-performance rendering

---

# Getting Started

## Prerequisites

* Node.js
* npm

## Installation

```bash
npm install
```

## Development

```bash
npm run dev
```

Open:

```text
http://localhost:5173
```

(or the port shown in your terminal)

---

# Project Philosophy

Circuit Room was built around a simple principle:

> **Understanding deep learning requires seeing the computation—not just the outcome.**

Most neural network visualizers animate abstract concepts or replay prerecorded sequences. Circuit Room instead computes every activation, every gradient, every weight update, and every prediction in real time, enabling users to inspect the actual mechanics of learning as they occur.

The objective is to transform neural networks from opaque mathematical abstractions into interactive systems that can be explored, understood, and experimented with.

---

# Privacy & Security

Circuit Room executes entirely within the user's browser.

* No model weights are transmitted to external servers.
* No canvas drawings or user inputs leave the device.
* No backend inference is performed.
* The repository includes a carefully configured `.gitignore` to prevent accidental exposure of environment variables, build artifacts, and temporary files.

---

# License

This project is open source. Refer to the repository's `LICENSE` file for licensing information.
