// NeuroForge — Network Architect (Redesigned with 3D Visualization)
// NOTE: this pass adds a new dependency — run:
//   npm install @react-three/postprocessing
// Everything else (store hooks, layout, panel logic) is untouched from the previous version.

import { useEffect, useRef, useCallback, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Line, Text, Trail } from '@react-three/drei';
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing';
import * as THREE from 'three';
import { Plus, Minus, Play, RotateCcw, Pause, Zap, Info, Brain } from 'lucide-react';
import useStore from '../../stores/useStore';
import { Tensor } from '../../engine/tensor.js';
import { activationHex, weightHex, LAYER_COLORS } from '../../components/ui/ColorSystem.js';

const NEURON_DETAIL = 1;   // icosahedron facet detail for the "gem" look
const CORE_RADIUS = 0.16;

// Deterministic per-neuron phase so every neuron breathes/pulses slightly
// out of sync with its neighbors instead of all pulsing in lockstep.
function seedFromPosition(position) {
  return Math.abs(Math.sin(position[0] * 12.9898 + position[1] * 78.233) * 43758.5453) % (Math.PI * 2);
}

/* ---------------------------------------------------------------------
   NEURON — faceted glowing core, rotating wireframe energy shell,
   soft additive halo, and a sonar-style pulse ring that fires while
   the neuron is active. This replaces the old flat "circle" neuron.
--------------------------------------------------------------------- */
function Neuron3D({ position, activation = 0, isInput, isOutput, radius = CORE_RADIUS }) {
  const coreRef = useRef();
  const shellRef = useRef();
  const glowRef = useRef();
  const ringRef = useRef();
  const phase = useMemo(() => seedFromPosition(position), [position]);
  const absAct = Math.min(Math.abs(activation), 1);

  const color = useMemo(() => {
    if (isInput) return '#06b6d4';
    if (isOutput) return '#f59e0b';
    return activationHex(absAct);
  }, [isInput, isOutput, absAct]);

  const colorObj = useMemo(() => new THREE.Color(color), [color]);

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    // layered sine waves = organic "breathing" instead of a single robotic pulse
    const breathe = Math.sin(t * 1.6 + phase) * 0.4 + Math.sin(t * 0.7 + phase * 1.7) * 0.6;
    const pulseScale = 1 + breathe * 0.035 + absAct * 0.14;

    if (coreRef.current) {
      coreRef.current.scale.setScalar(pulseScale);
      coreRef.current.material.emissiveIntensity = 0.35 + absAct * 1.5 + Math.max(0, breathe) * 0.15;
    }
    if (shellRef.current) {
      shellRef.current.rotation.x = t * 0.15 + phase;
      shellRef.current.rotation.y = t * 0.22 + phase;
      shellRef.current.material.opacity = 0.06 + absAct * 0.35;
    }
    if (glowRef.current) {
      glowRef.current.scale.setScalar(1 + Math.sin(t * 1.2 + phase) * 0.06);
      glowRef.current.material.opacity = 0.03 + absAct * 0.17;
    }
    if (ringRef.current) {
      if (absAct > 0.12) {
        const loop = ((t * 0.5 + phase) % 1.4) / 1.4;
        ringRef.current.visible = true;
        ringRef.current.scale.setScalar(1 + loop * 3.4);
        ringRef.current.material.opacity = (1 - loop) * absAct * 0.35;
      } else {
        ringRef.current.visible = false;
      }
    }
  });

  return (
    <group position={position}>
      {/* soft outer halo — additive blending so overlapping neurons glow together */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[radius * 3.2, 16, 16]} />
        <meshBasicMaterial color={color} transparent opacity={0.05} blending={THREE.AdditiveBlending} depthWrite={false} />
      </mesh>

      {/* rotating faceted "energy cage" around the core */}
      <mesh ref={shellRef}>
        <icosahedronGeometry args={[radius * 1.9, NEURON_DETAIL]} />
        <meshBasicMaterial color={color} wireframe transparent opacity={0.12} depthWrite={false} />
      </mesh>

      {/* sonar pulse ring — expands and fades each time the neuron fires */}
      <mesh ref={ringRef} rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[radius * 1.3, radius * 1.55, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>

      {/* faceted glowing core */}
      <mesh ref={coreRef}>
        <icosahedronGeometry args={[radius, NEURON_DETAIL + 1]} />
        <meshPhysicalMaterial
          color={colorObj}
          emissive={colorObj}
          emissiveIntensity={0.4}
          roughness={0.25}
          metalness={0.15}
          clearcoat={0.6}
          clearcoatRoughness={0.3}
          transparent
          opacity={0.95}
        />
      </mesh>
    </group>
  );
}

/* ---------------------------------------------------------------------
   CONNECTION — gently curved (not a rigid straight line), colored and
   thickened by weight sign/magnitude, with a flowing dash animation
   on strong connections so "energy flow" is visible even at rest.
--------------------------------------------------------------------- */
function Connection3D({ from, to, weight = 0 }) {
  const lineRef = useRef();
  const absWeight = Math.min(Math.abs(weight), 1.4);
  const isStrong = absWeight > 0.35;

  const points = useMemo(() => {
    const start = new THREE.Vector3(...from);
    const end = new THREE.Vector3(...to);
    const mid = start.clone().lerp(end, 0.5);
    mid.z += 0.12 * (weight >= 0 ? 1 : -1) * absWeight; // arc direction hints sign
    const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
    return curve.getPoints(16);
  }, [from, to, weight, absWeight]);

  const color = weight >= 0 ? '#3b82f6' : '#f43f5e';
  const baseOpacity = 0.035 + absWeight * 0.22;

  useFrame((state) => {
    if (isStrong && lineRef.current?.material) {
      lineRef.current.material.dashOffset = -(state.clock.elapsedTime * 0.6) % 1;
    }
  });

  return (
    <Line
      ref={lineRef}
      points={points}
      color={color}
      transparent
      opacity={baseOpacity}
      lineWidth={0.6 + absWeight * 1.8}
      dashed={isStrong}
      dashSize={0.12}
      dashScale={4}
      gapSize={0.08}
      depthWrite={false}
    />
  );
}

/* ---------------------------------------------------------------------
   DATA PARTICLE — a signal traveling along a connection, now with a
   real light trail instead of a bare dot.
--------------------------------------------------------------------- */
function DataParticle({ from, to, progress, positive = true }) {
  const t = Math.max(0, Math.min(1, progress));
  const pos = useMemo(() => [
    from[0] + (to[0] - from[0]) * t,
    from[1] + (to[1] - from[1]) * t,
    from[2] + (to[2] - from[2]) * t,
  ], [from, to, t]);

  const color = positive ? '#a78bfa' : '#fb923c';

  return (
    <Trail width={2.2} length={4} color={color} attenuation={(w) => w * w} decay={2}>
      <mesh position={pos}>
        <sphereGeometry args={[0.045, 8, 8]} />
        <meshBasicMaterial color={color} transparent opacity={0.95} blending={THREE.AdditiveBlending} depthWrite={false} />
      </mesh>
    </Trail>
  );
}

/* ---------------------------------------------------------------------
   CAMERA INTRO — a short cinematic dolly-in on mount instead of
   dropping the user straight in front of the network.
--------------------------------------------------------------------- */
function CameraIntro() {
  const { camera } = useThree();
  const animating = useRef(true);
  const from = useMemo(() => new THREE.Vector3(0, 1.2, 11), []);
  const to = useMemo(() => new THREE.Vector3(0, 0, 6), []);

  useFrame((state) => {
    if (!animating.current) return;
    const t = Math.min(state.clock.elapsedTime / 1.6, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    camera.position.lerpVectors(from, to, eased);
    camera.lookAt(0, 0, 0);
    if (t >= 1) animating.current = false;
  });

  return null;
}

function LayerLabel({ x, topY, text, color }) {
  return (
    <Text
      position={[x, topY + 0.5, 0]}
      fontSize={0.14}
      color={color}
      anchorX="center"
      anchorY="bottom"
      letterSpacing={0.08}
    >
      {text.toUpperCase()}
    </Text>
  );
}

// 3D Network Scene
function NetworkScene({ layerSizes, network, forwardPassData, animationProgress }) {
  const maxNeurons = 12;
  const layerSpacing = 2.0;
  const neuronSpacing = 0.5;

  // Calculate 3D positions for all neurons
  const nodePositions = useMemo(() => {
    return layerSizes.map((size, li) => {
      const displayCount = Math.min(size, maxNeurons);
      const x = (li - (layerSizes.length - 1) / 2) * layerSpacing;
      return Array.from({ length: displayCount }, (_, ni) => {
        const y = (ni - (displayCount - 1) / 2) * neuronSpacing;
        return [x, y, 0];
      });
    });
  }, [layerSizes]);

  // Get activations
  const getActivation = (layerIdx, neuronIdx) => {
    if (layerIdx === 0 || !network) return 0;
    const layer = network.layers[layerIdx - 1];
    if (!layer || !layer.output) return 0;
    return Math.min(1, Math.abs(layer.output.data[neuronIdx] || 0));
  };

  const getWeight = (layerIdx, fromIdx, toIdx) => {
    if (!network || layerIdx >= network.layers.length) return 0;
    const layer = network.layers[layerIdx];
    if (!layer || !layer.weights) return 0;
    const idx = fromIdx * layer.outputSize + toIdx;
    return layer.weights.data[idx] || 0;
  };

  // Forward pass particles
  const particles = useMemo(() => {
    if (!forwardPassData || animationProgress === undefined) return [];
    const currentLayerFloat = animationProgress * (layerSizes.length - 1);
    const currentLayer = Math.floor(currentLayerFloat);
    const layerProgress = currentLayerFloat - currentLayer;

    if (currentLayer >= nodePositions.length - 1) return [];

    const from = nodePositions[currentLayer];
    const to = nodePositions[currentLayer + 1];
    const result = [];

    for (let fi = 0; fi < Math.min(from.length, 4); fi++) {
      for (let ti = 0; ti < Math.min(to.length, 4); ti++) {
        const w = getWeight(currentLayer, fi, ti);
        result.push({ from: from[fi], to: to[ti], progress: layerProgress, positive: w >= 0 });
      }
    }
    return result;
  }, [forwardPassData, animationProgress, nodePositions, layerSizes, network]);

  return (
    <>
      <fog attach="fog" args={['#05050b', 7, 17]} />
      <ambientLight intensity={0.12} />
      <pointLight position={[5, 5, 5]} intensity={0.6} color="#8b5cf6" />
      <pointLight position={[-5, 3, -3]} intensity={0.35} color="#06b6d4" />
      <pointLight position={[0, -3, 5]} intensity={0.25} color="#f59e0b" />
      <pointLight position={[0, 0, -6]} intensity={0.2} color="#ffffff" />

      <gridHelper args={[20, 24, '#1a1a2e', '#12121c']} position={[0, -2.6, 0]} />

      <CameraIntro />

      {/* Connections */}
      {nodePositions.map((layer, li) =>
        li < nodePositions.length - 1 &&
        layer.map((fromPos, fi) =>
          nodePositions[li + 1].map((toPos, ti) => (
            <Connection3D
              key={`${li}-${fi}-${ti}`}
              from={fromPos}
              to={toPos}
              weight={getWeight(li, fi, ti)}
            />
          ))
        )
      )}

      {/* Neurons */}
      {nodePositions.map((layer, li) =>
        layer.map((pos, ni) => (
          <Neuron3D
            key={`n-${li}-${ni}`}
            position={pos}
            activation={getActivation(li, ni)}
            isInput={li === 0}
            isOutput={li === layerSizes.length - 1}
          />
        ))
      )}

      {/* Layer labels floating above each column */}
      {nodePositions.map((layer, li) => {
        const topY = Math.max(...layer.map((p) => p[1]));
        const isInput = li === 0;
        const isOutput = li === layerSizes.length - 1;
        const label = isInput ? 'Input' : isOutput ? 'Output' : `Hidden ${li}`;
        const color = isInput ? LAYER_COLORS.input.primary : isOutput ? LAYER_COLORS.output.primary : LAYER_COLORS.hidden.primary;
        return <LayerLabel key={`label-${li}`} x={layer[0][0]} topY={topY} text={label} color={color} />;
      })}

      {/* Forward pass particles */}
      {particles.map((p, i) => (
        <DataParticle key={`p-${i}`} from={p.from} to={p.to} progress={p.progress} positive={p.positive} />
      ))}

      <OrbitControls
        enableDamping
        dampingFactor={0.06}
        minDistance={3}
        maxDistance={13}
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.25}
      />

      {/* Post-processing glow — this is what makes the cores actually read as "energy" rather than plastic spheres */}
      <EffectComposer multisampling={4}>
        <Bloom intensity={0.9} luminanceThreshold={0.15} luminanceSmoothing={0.4} radius={0.6} />
        <Vignette eskil={false} offset={0.15} darkness={0.7} />
      </EffectComposer>
    </>
  );
}

export default function NetworkArchitect() {
  const {
    layerSizes, activationFn, dataset, datasetId, network,
    addLayer, removeLayer, updateLayerSize,
    setActivation, setDataset, initNetwork, networkVersion
  } = useStore();

  const [isRunningForward, setIsRunningForward] = useState(false);
  const [forwardPassData, setForwardPassData] = useState(null);
  const [animationProgress, setAnimationProgress] = useState(0);
  const animRef = useRef(null);

  useEffect(() => { if (!network) initNetwork(); }, []);

  const runForwardPass = useCallback(() => {
    if (!network || !dataset) return;
    const idx = Math.floor(Math.random() * dataset.points.length);
    const sample = dataset.points[idx];
    const input = new Tensor([sample], [1, 2]);
    network.forward(input);
    setForwardPassData({ input: sample, idx });
    setIsRunningForward(true);
    setAnimationProgress(0);

    let start = null;
    const duration = 2500;
    const step = (ts) => {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      setAnimationProgress(progress);
      if (progress < 1) animRef.current = requestAnimationFrame(step);
      else setIsRunningForward(false);
    };
    if (animRef.current) cancelAnimationFrame(animRef.current);
    animRef.current = requestAnimationFrame(step);
  }, [network, dataset]);

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="module-header">
        <div className="flex items-center gap-3">
          <div className="module-icon" style={{ background: 'linear-gradient(135deg, #7c3aed, #6d28d9)' }}>
            <Brain className="w-4 h-4 text-white" />
          </div>
          <div>
            <h2 className="text-headline text-base text-text-primary">Network Architect</h2>
            <p className="text-[11px] text-text-tertiary mt-0.5">Build and explore neural network architectures in 3D</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <motion.button
            onClick={runForwardPass}
            disabled={isRunningForward}
            className="btn-primary flex items-center gap-1.5 text-xs disabled:opacity-40"
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.97 }}
          >
            {isRunningForward ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            {isRunningForward ? 'Running...' : 'Forward Pass'}
          </motion.button>
          <button
            onClick={() => { initNetwork(); setForwardPassData(null); }}
            className="btn-ghost flex items-center gap-1.5 text-xs"
          >
            <RotateCcw className="w-3.5 h-3.5" /> Reset
          </button>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* 3D Canvas */}
        <div className="flex-1 relative" style={{ background: 'var(--color-void)' }}>
          <Canvas dpr={[1, 2]} gl={{ antialias: true, powerPreference: 'high-performance' }} camera={{ position: [0, 1.2, 11], fov: 45 }}>
            <NetworkScene
              layerSizes={layerSizes}
              network={network}
              forwardPassData={forwardPassData}
              animationProgress={animationProgress}
            />
          </Canvas>

          {/* Forward pass overlay */}
          <AnimatePresence>
            {forwardPassData && (
              <motion.div
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="absolute top-4 left-4 sf-glass rounded-xl px-4 py-3"
              >
                <div className="text-label mb-1">Forward Pass</div>
                <div className="text-mono text-xs text-text-primary">
                  Input: [{forwardPassData.input.map(v => v.toFixed(3)).join(', ')}]
                </div>
                {network && animationProgress >= 1 && network.layerActivations.length > 0 && (
                  <div className="text-mono text-xs text-nf-amber mt-1">
                    Output: [{Array.from(network.layerActivations[network.layerActivations.length - 1].data.data.slice(0, 4)).map(v => v.toFixed(3)).join(', ')}]
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Config Panel */}
        <div className="w-56 border-l border-white/[0.04] overflow-y-auto" style={{ background: 'var(--color-deep)' }}>
          {/* Dataset */}
          <div className="panel-section">
            <label className="text-label">Dataset</label>
            <select value={datasetId} onChange={(e) => setDataset(e.target.value)}
              className="control-select mt-2 w-full">
              <option value="spiral">🌀 Spiral</option>
              <option value="circles">⭕ Circles</option>
              <option value="moons">🌙 Moons</option>
              <option value="xor">✖️ XOR</option>
              <option value="gaussian">🔵 Gaussian</option>
              <option value="linear">📏 Linear</option>
            </select>
          </div>

          {/* Activation */}
          <div className="panel-section">
            <label className="text-label">Activation</label>
            <select value={activationFn} onChange={(e) => setActivation(e.target.value)}
              className="control-select mt-2 w-full">
              <option value="relu">ReLU</option>
              <option value="sigmoid">Sigmoid</option>
              <option value="tanh">Tanh</option>
              <option value="gelu">GELU</option>
              <option value="leakyRelu">Leaky ReLU</option>
              <option value="swish">Swish</option>
            </select>
          </div>

          {/* Architecture */}
          <div className="panel-section">
            <label className="text-label">Architecture</label>
            <div className="mt-3 space-y-2">
              {layerSizes.map((size, i) => {
                const isInput = i === 0;
                const isOutput = i === layerSizes.length - 1;
                const layerColor = isInput ? LAYER_COLORS.input : isOutput ? LAYER_COLORS.output : LAYER_COLORS.hidden;
                return (
                  <div key={i} className="flex items-center gap-2">
                    <span className="text-mono text-[10px] w-12" style={{ color: layerColor.primary }}>
                      {isInput ? 'Input' : isOutput ? 'Output' : `H-${i}`}
                    </span>
                    <div className="flex-1 flex items-center gap-1">
                      <button onClick={() => updateLayerSize(i, size - 1)} disabled={isInput || size <= 1}
                        className="w-5 h-5 rounded sf-deep flex items-center justify-center text-text-ghost hover:text-text-primary disabled:opacity-20 text-xs">−</button>
                      <div className="flex-1 text-center text-xs text-mono text-text-primary sf-deep rounded py-1">{size}</div>
                      <button onClick={() => updateLayerSize(i, size + 1)} disabled={isInput || size >= 64}
                        className="w-5 h-5 rounded sf-deep flex items-center justify-center text-text-ghost hover:text-text-primary disabled:opacity-20 text-xs">+</button>
                    </div>
                    {!isInput && !isOutput && (
                      <button onClick={() => removeLayer(i)} className="text-text-ghost hover:text-nf-rose text-xs">×</button>
                    )}
                  </div>
                );
              })}
            </div>
            <button onClick={() => addLayer(layerSizes.length - 1, 8)}
              className="mt-3 w-full py-1.5 rounded-lg border border-dashed border-white/[0.06] text-text-ghost text-[10px] hover:text-text-secondary hover:border-nf-violet/20 transition-all flex items-center justify-center gap-1">
              <Plus className="w-3 h-3" /> Add Hidden Layer
            </button>
          </div>

          {/* Network Stats */}
          {network && (
            <div className="panel-section">
              <label className="text-label">Network</label>
              <div className="mt-2 space-y-1.5">
                {[
                  { label: 'Layers', value: network.layers.length },
                  { label: 'Parameters', value: network.totalParams().toLocaleString() },
                  { label: 'Loss', value: network.lossFunction.name },
                ].map((row, i) => (
                  <div key={i} className="flex justify-between text-xs">
                    <span className="text-text-tertiary">{row.label}</span>
                    <span className="text-mono text-text-secondary">{row.value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Educational */}
          <div className="panel-section">
            <div className="callout callout-violet">
              <p className="text-[10px] text-text-secondary leading-relaxed">
                Each neuron computes <span className="text-mono text-nf-violet">f(w·x + b)</span> — a weighted sum passed through an activation. Deep learning = composing many such simple transforms.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}