<div align="center">

# 🧠 Circuit Room

### A browser-native neural network simulator that turns deep learning into interactive, real-time 3D visualizations.

[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=white)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-Build-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev)
[![Three.js](https://img.shields.io/badge/Three.js-3D%20Engine-000000?style=for-the-badge&logo=three.js&logoColor=white)](https://threejs.org)
[![Zustand](https://img.shields.io/badge/Zustand-State-433E38?style=for-the-badge)](https://github.com/pmndrs/zustand)
[![Client Side Only](https://img.shields.io/badge/Inference-100%25%20Client--Side-success?style=for-the-badge)](#-privacy--security)

<sub>⭐ If Circuit Room helps you understand neural nets better, consider starring the repo — it genuinely helps.</sub>

</div>

---

Circuit Room is a professional-grade neural network visualization platform that runs **entirely in the browser**. Unlike educational demos that replay precomputed animations, Circuit Room performs **real neural network computation** in real time — forward propagation, matrix multiplication, gradient computation, backpropagation — and renders every stage of the learning process as it happens.

The goal is simple: make the mathematics behind deep learning **observable**, not just describable. Every activation, weight update, and gradient flow is computed live and rendered as interactive 3D geometry with physically inspired lighting, so you can see *why* a network behaves the way it does — not just its final prediction.

<br>

## 📚 Table of Contents

- [Live Demo](#-live-demo)
- [Why Circuit Room](#-why-circuit-room)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Privacy & Security](#-privacy--security)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

<br>

## 🎬 Live Demo

<div align="center">

**[→ Try Circuit Room Live](#)** &nbsp;·&nbsp; *(add your deployed URL here)*

<!-- Add a real screenshot or GIF for maximum impact, e.g.: -->
<!-- ![Circuit Room Demo](docs/demo.gif) -->
<sub>💡 Tip: a 10–15s screen-recorded GIF of the Training Arena or CNN visualizer here will do more for this README than any amount of text.</sub>

</div>

<br>

## 💡 Why Circuit Room

> **Understanding deep learning requires seeing the computation — not just the outcome.**

Most neural network visualizers animate abstract concepts or replay prerecorded sequences. Circuit Room computes every activation, every gradient, every weight update, and every prediction in real time — turning neural networks from opaque mathematical abstractions into systems you can explore, poke at, and actually understand.

<br>

## ✨ Features

| Module | What it does | Live computation |
|---|---|:---:|
| 🏗️ Network Architect | Design custom feedforward nets in 3D | ✅ |
| 🎯 Training Arena | Watch real gradient descent unfold | ✅ |
| ✍️ CNN Digit Recognition | Draw a digit, watch a CNN classify it layer-by-layer | ✅ |
| 🏔️ Loss Landscape Explorer | Navigate a 3D loss surface as optimization happens | ✅ |
| 🎛️ Interactive Playground | Explore activations & regularization hands-on | ✅ |

<details>
<summary><b>🏗️ Network Architect</b> — design & explore custom feedforward networks</summary>
<br>

Design and explore fully customizable feedforward neural networks in an immersive 3D environment.

**Includes:**
- Interactive visual architecture editor
- Adjustable hidden layers and neuron counts
- Multiple activation functions: ReLU, Sigmoid, Tanh, GELU, Swish, Leaky ReLU
- Live forward propagation visualization
- Weight-aware animated connections
- Activation-driven neuron pulses
- Orbit, pan, and inspect the network from any angle

Each neuron renders as a glowing energy core inside a rotating shell; connections shift color and intensity based on weight sign and magnitude.

</details>

<details>
<summary><b>🎯 Training Arena</b> — real gradient descent, visualized live</summary>
<br>

Train custom neural networks using real gradient descent while watching the learning process unfold in real time.

**Includes:**
- Forward propagation
- Backpropagation
- Live gradient updates
- Weight and bias evolution
- Continuously updating decision boundaries
- Interactive 2D datasets

Instead of a static end result, Circuit Room exposes the entire optimization process as it happens.

</details>

<details>
<summary><b>✍️ CNN Digit Recognition</b> — draw a digit, watch it get classified</summary>
<br>

Draw a handwritten digit and watch a real Convolutional Neural Network process it layer by layer.

**Includes:**
- Live convolution operations
- Individual feature maps for every filter
- Progressive activation visualization
- Flattened feature vectors
- Final classification probabilities

Uses PyTorch-equivalent MNIST preprocessing while running inference entirely in optimized JavaScript.

</details>

<details>
<summary><b>🏔️ Loss Landscape Explorer</b> — optimization as movement across a 3D surface</summary>
<br>

Visualize optimization as movement across a complex three-dimensional loss surface.

**Includes:**
- Physics-inspired optimization
- Momentum-based movement
- Interactive camera controls
- Non-convex terrain visualization
- Optimization path tracking

Gives an intuitive feel for gradient descent dynamics beyond the usual 2D plots.

</details>

<details>
<summary><b>🎛️ Interactive Playground</b> — hands-on deep learning fundamentals</summary>
<br>

Experiment with fundamental deep learning concepts directly in the browser.

**Includes:**
- Activation functions
- Regularization techniques
- Mathematical intuition behind neural networks
- Interactive parameter exploration

</details>

<br>

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React, Vite |
| **State Management** | Zustand |
| **3D Graphics** | Three.js, React Three Fiber (R3F), `@react-three/postprocessing` |
| **Neural Network Engine** | Custom vanilla JS engine — zero runtime dependencies, flat `Float32Array` memory layout, real forward/backward pass |
| **Animation** | Framer Motion |
| **Icons** | Lucide React |
| **Design System** | Custom dark glassmorphism, semantic color system, handcrafted CSS |

<br>

## 🚀 Getting Started

### Prerequisites

- Node.js
- npm

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Then open:

```
http://localhost:5173
```

*(or whichever port your terminal shows)*

<br>

## 🔒 Privacy & Security

Circuit Room executes entirely within your browser — nothing leaves your machine.

- 🚫 No model weights are transmitted to external servers
- 🚫 No canvas drawings or user inputs leave the device
- 🚫 No backend inference is performed
- ✅ A carefully configured `.gitignore` prevents accidental exposure of environment variables, build artifacts, and temporary files

<br>

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

<br>

## 📄 License

This project is open source. See the [`LICENSE`](./LICENSE) file for details.

<br>

## 👤 Author

**Rhythem**

[![GitHub](https://img.shields.io/badge/GitHub-Rhythem2005-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Rhythem2005)
[![Portfolio](https://img.shields.io/badge/Portfolio-View-000000?style=flat-square&logo=vercel&logoColor=white)](https://portfolio-pqfa.vercel.app)

<div align="center">
<sub>Built with a lot of Float32Arrays and probably too much coffee ☕</sub>
</div>