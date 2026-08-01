// NeuroForge — CNN Digit Recognition Module
// Draw a digit → watch a CNN process it layer by layer in real-time
// NOTE: uses the same @react-three/postprocessing dependency already added for Network Architect.

import { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line, Html } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';
import { PenTool, Trash2, Cpu, ChevronRight, Eye } from 'lucide-react';
import { createDigitCNN, forwardPassCNN } from '../../engine/cnn.js';
import { heatmapColor, LAYER_COLORS, CLASS_PALETTE } from '../../components/ui/ColorSystem.js';

// Drawing canvas component
function DrawingCanvas({ onImageChange, canvasSize = 280 }) {
  const canvasRef = useRef(null);
  const isDrawing = useRef(false);
  const lastPos = useRef(null);
  const [active, setActive] = useState(false);
  const [scanTrigger, setScanTrigger] = useState(0);

  const getImageData = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const ctx = canvas.getContext('2d');

    // Downsample to 28x28
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    const imgData = tempCtx.getImageData(0, 0, 28, 28);

    const image = Array.from({ length: 28 }, (_, i) =>
      new Float32Array(Array.from({ length: 28 }, (_, j) => {
        const idx = (i * 28 + j) * 4;
        return imgData.data[idx] / 255; // Grayscale normalized
      }))
    );
    return image;
  }, []);

  const clear = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    lastPos.current = null;
    onImageChange(getImageData());
  }, [onImageChange, getImageData]);

  const handlePointerDown = (e) => {
    isDrawing.current = true;
    setActive(true);
    const rect = canvasRef.current.getBoundingClientRect();
    lastPos.current = {
      x: (e.clientX - rect.left) * (canvasRef.current.width / rect.width),
      y: (e.clientY - rect.top) * (canvasRef.current.height / rect.height)
    };
  };

  const handlePointerMove = (e) => {
    if (!isDrawing.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);

    if (lastPos.current) {
      ctx.beginPath();
      ctx.moveTo(lastPos.current.x, lastPos.current.y);
      ctx.lineTo(x, y);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 18;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.stroke();
    }
    lastPos.current = { x, y };
  };

  const handlePointerUp = () => {
    if (!isDrawing.current) return;
    isDrawing.current = false;
    lastPos.current = null;
    setActive(false);
    setScanTrigger((v) => v + 1); // fires the "reading your drawing" sweep
    onImageChange(getImageData());
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = 280;
    canvas.height = 280;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, 280, 280);
  }, []);

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="text-label mb-1">Draw a digit (0–9)</div>
      <motion.div
        className="relative viz-container overflow-hidden"
        style={{ width: canvasSize, height: canvasSize }}
        animate={{ boxShadow: active ? '0 0 30px rgba(139,92,246,0.4)' : '0 0 0px rgba(139,92,246,0)' }}
        transition={{ duration: 0.3 }}
      >
        <canvas
          ref={canvasRef}
          style={{ width: canvasSize, height: canvasSize, cursor: 'crosshair', touchAction: 'none' }}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerLeave={handlePointerUp}
        />
        {/* Grid overlay */}
        <div className="absolute inset-0 pointer-events-none" style={{
          backgroundImage: `
            linear-gradient(rgba(139, 92, 246, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(139, 92, 246, 0.05) 1px, transparent 1px)
          `,
          backgroundSize: `${canvasSize / 28}px ${canvasSize / 28}px`
        }} />
        {/* Scan sweep — plays once each time a stroke lifts, signaling "the network is reading this" */}
        <AnimatePresence>
          {scanTrigger > 0 && (
            <motion.div
              key={scanTrigger}
              initial={{ top: '0%', opacity: 0.9 }}
              animate={{ top: '100%', opacity: 0 }}
              transition={{ duration: 0.55, ease: 'easeOut' }}
              className="absolute left-0 right-0 h-px pointer-events-none"
              style={{
                background: 'linear-gradient(90deg, transparent, #8b5cf6, transparent)',
                boxShadow: '0 0 14px 2px rgba(139,92,246,0.65)'
              }}
            />
          )}
        </AnimatePresence>
      </motion.div>
      <button onClick={clear} className="btn-ghost flex items-center gap-1.5 text-xs">
        <Trash2 className="w-3.5 h-3.5" /> Clear Canvas
      </button>
    </div>
  );
}

/* ---------------------------------------------------------------------
   FEATURE MAP TEXTURE — paints one filter's activation grid onto a
   small offscreen canvas so it can be used as a GPU texture.
--------------------------------------------------------------------- */
function buildMapTexture(fmap, mapSize) {
  const [h, w] = mapSize;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');

  let min = Infinity, max = -Infinity;
  for (const row of fmap) for (let i = 0; i < row.length; i++) {
    min = Math.min(min, row[i]); max = Math.max(max, row[i]);
  }
  const range = max - min || 1;

  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      const val = fmap[i] ? (fmap[i][j] || 0) : 0;
      const norm = (val - min) / range;
      ctx.fillStyle = heatmapColor(norm);
      ctx.fillRect(j, i, 1, 1);
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  texture.needsUpdate = true;
  return texture;
}

/* ---------------------------------------------------------------------
   FEATURE MAP PLANE — a single filter rendered as a floating glowing
   "screen", with a hover label so a user can inspect individual filters.
--------------------------------------------------------------------- */
function FeatureMapPlane3D({ texture, position, size, color, label, index }) {
  const groupRef = useRef();
  const [hovered, setHovered] = useState(false);
  const phase = useMemo(() => (index * 1.37) % (Math.PI * 2), [index]);
  const frameGeo = useMemo(() => new THREE.EdgesGeometry(new THREE.PlaneGeometry(size[0], size[1])), [size]);

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (groupRef.current) {
      groupRef.current.position.y = position[1] + Math.sin(t * 0.8 + phase) * 0.02;
      groupRef.current.rotation.y = Math.sin(t * 0.3 + phase) * 0.05;
    }
  });

  return (
    <group
      ref={groupRef}
      position={position}
      onPointerOver={(e) => { e.stopPropagation(); setHovered(true); }}
      onPointerOut={() => setHovered(false)}
    >
      {/* glow backing */}
      <mesh position={[0, 0, -0.02]}>
        <planeGeometry args={[size[0] * 1.2, size[1] * 1.2]} />
        <meshBasicMaterial color={color} transparent opacity={hovered ? 0.4 : 0.16} blending={THREE.AdditiveBlending} depthWrite={false} />
      </mesh>
      {/* the feature map itself, unlit so the heatmap colors read accurately */}
      <mesh scale={hovered ? 1.07 : 1}>
        <planeGeometry args={size} />
        <meshBasicMaterial map={texture} toneMapped={false} />
      </mesh>
      {/* thin glowing frame */}
      <lineSegments geometry={frameGeo} scale={hovered ? 1.07 : 1}>
        <lineBasicMaterial color={color} transparent opacity={0.55} />
      </lineSegments>
      {hovered && (
        <Html center distanceFactor={8} position={[0, size[1] / 2 + 0.2, 0.05]}>
          <div
            className="px-1.5 py-0.5 rounded text-[9px] font-mono whitespace-nowrap"
            style={{ background: 'rgba(6,7,14,0.9)', color, border: `1px solid ${color}55` }}
          >
            {label}
          </div>
        </Html>
      )}
    </group>
  );
}

/* ---------------------------------------------------------------------
   FEATURE MAP WALL — every filter in this stage rendered as a receding
   wall of glowing screens, each showing what that filter detected.
--------------------------------------------------------------------- */
function FeatureMapWall3D({ maps, mapSize, type, stageName, maxDisplay = 16 }) {
  const layerColor = (LAYER_COLORS[type] || LAYER_COLORS.hidden).primary;
  const displayMaps = useMemo(() => (maps ? maps.slice(0, maxDisplay) : []), [maps, maxDisplay]);
  const textures = useMemo(
    () => displayMaps.map((fmap) => buildMapTexture(fmap, mapSize)),
    [displayMaps, mapSize]
  );

  useEffect(() => () => textures.forEach((t) => t.dispose()), [textures]);

  if (!displayMaps.length) return null;

  const cols = Math.min(displayMaps.length, 4) || 1;
  const rows = Math.ceil(displayMaps.length / cols);
  const aspect = mapSize[1] / mapSize[0];
  const planeH = 1.05;
  const planeW = planeH * aspect;
  const gapX = planeW + 0.4;
  const gapY = planeH + 0.4;

  return (
    <div className="relative w-full" style={{ height: 320 }}>
      <Canvas dpr={[1, 2]} camera={{ position: [0, 1.6, 6.5], fov: 42 }}>
        <fog attach="fog" args={['#05050b', 5, 12]} />
        <ambientLight intensity={0.28} />
        <pointLight position={[3, 3, 4]} intensity={0.55} color="#8b5cf6" />
        <pointLight position={[-3, 2, 2]} intensity={0.3} color="#06b6d4" />
        <pointLight position={[0, -1, 5]} intensity={0.2} color="#f59e0b" />

        {displayMaps.map((fmap, idx) => {
          const col = idx % cols;
          const row = Math.floor(idx / cols);
          const x = (col - (cols - 1) / 2) * gapX;
          const y = ((rows - 1) / 2 - row) * gapY;
          const z = -row * 0.55;
          return (
            <FeatureMapPlane3D
              key={idx}
              texture={textures[idx]}
              position={[x, y, z]}
              size={[planeW, planeH]}
              color={layerColor}
              label={`${stageName} · filter ${idx + 1}`}
              index={idx}
            />
          );
        })}

        <OrbitControls
          enableDamping
          dampingFactor={0.08}
          minDistance={3}
          maxDistance={11}
          autoRotate
          autoRotateSpeed={0.35}
          enablePan={false}
        />
        <EffectComposer multisampling={4}>
          <Bloom intensity={0.55} luminanceThreshold={0.2} luminanceSmoothing={0.4} radius={0.5} />
        </EffectComposer>
      </Canvas>
      {maps && maps.length > maxDisplay && (
        <span className="absolute bottom-2 right-3 text-[9px] text-text-ghost">
          +{maps.length - maxDisplay} more filters
        </span>
      )}
    </div>
  );
}

/* ---------------------------------------------------------------------
   DENSE NODE — one value in a flatten/dense vector, rendered with the
   same "energy node" language as the Network Architect neurons so the
   whole app reads as one consistent visual system.
--------------------------------------------------------------------- */
function DenseNode3D({ position, intensity, color, index }) {
  const ref = useRef();
  const phase = useMemo(() => (index * 0.71) % (Math.PI * 2), [index]);

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (ref.current) {
      const s = 1 + Math.sin(t * 1.4 + phase) * 0.05 + intensity * 0.3;
      ref.current.scale.setScalar(s);
      ref.current.material.emissiveIntensity = 0.3 + intensity * 1.7;
    }
  });

  return (
    <mesh ref={ref} position={position}>
      <sphereGeometry args={[0.045 + intensity * 0.04, 12, 12]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.4} roughness={0.3} metalness={0.2} />
    </mesh>
  );
}

function DenseColumn3D({ values, title, type }) {
  const layerColor = (LAYER_COLORS[type] || LAYER_COLORS.dense || LAYER_COLORS.hidden).primary;
  const display = useMemo(() => (values ? values.slice(0, 64) : []), [values]);
  const max = Math.max(...display.map((v) => Math.abs(v)), 0.01);

  const positions = useMemo(() => {
    const n = display.length;
    const spacing = Math.min(0.16, 9 / Math.max(n, 1));
    return display.map((_, i) => {
      const x = (i - (n - 1) / 2) * spacing;
      const z = Math.sin(i * 0.4) * 0.15;
      return [x, 0, z];
    });
  }, [display.length]);

  if (!display.length) return null;
  const points = positions.map((p) => new THREE.Vector3(...p));

  return (
    <div className="relative w-full" style={{ height: 220 }}>
      <Canvas dpr={[1, 2]} camera={{ position: [0, 0.6, 3.2], fov: 45 }}>
        <fog attach="fog" args={['#05050b', 3, 8]} />
        <ambientLight intensity={0.2} />
        <pointLight position={[2, 2, 3]} intensity={0.5} color="#8b5cf6" />
        <pointLight position={[-2, 1, 1]} intensity={0.3} color="#06b6d4" />

        <Line points={points} color={layerColor} transparent opacity={0.15} lineWidth={1} depthWrite={false} />

        {display.map((v, i) => (
          <DenseNode3D
            key={i}
            position={positions[i]}
            intensity={Math.abs(v) / max}
            color={heatmapColor(Math.abs(v) / max, 0.9)}
            index={i}
          />
        ))}

        <OrbitControls
          enableDamping
          dampingFactor={0.08}
          minDistance={1.5}
          maxDistance={6}
          autoRotate
          autoRotateSpeed={0.3}
          enablePan={false}
        />
        <EffectComposer multisampling={4}>
          <Bloom intensity={0.65} luminanceThreshold={0.15} luminanceSmoothing={0.5} radius={0.5} />
        </EffectComposer>
      </Canvas>
    </div>
  );
}

// Output probabilities
function OutputProbs({ probs }) {
  if (!probs) return null;
  const maxIdx = probs.indexOf(Math.max(...probs));

  return (
    <div className="w-full">
      <div className="flex items-center gap-1.5 mb-3">
        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: LAYER_COLORS.output.primary }} />
        <span className="text-xs font-semibold text-nf-amber">Prediction</span>
      </div>
      <div className="space-y-1.5">
        {probs.map((p, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className={`text-mono text-xs w-4 text-center ${i === maxIdx ? 'text-nf-amber font-bold' : 'text-text-tertiary'}`}>
              {i}
            </span>
            <div className="flex-1 h-5 rounded-md overflow-hidden" style={{ background: 'rgba(6, 7, 14, 0.6)' }}>
              <motion.div
                className="h-full rounded-md"
                initial={{ width: 0 }}
                animate={{ width: `${p * 100}%` }}
                transition={{ duration: 0.4, ease: 'easeOut' }}
                style={{
                  background: i === maxIdx
                    ? 'linear-gradient(90deg, #f59e0b, #f97316)'
                    : 'linear-gradient(90deg, rgba(139, 92, 246, 0.4), rgba(139, 92, 246, 0.2))',
                  boxShadow: i === maxIdx ? '0 0 12px rgba(245, 158, 11, 0.3)' : 'none'
                }}
              />
            </div>
            <span className={`text-mono text-[11px] w-12 text-right ${i === maxIdx ? 'text-nf-amber font-bold' : 'text-text-tertiary'}`}>
              {(p * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
      <div className="mt-4 text-center">
        <div className="text-text-tertiary text-[10px] uppercase tracking-wider">Predicted Digit</div>
        <div className="relative inline-block mt-1">
          <motion.div
            className="absolute inset-0 rounded-full"
            animate={{ opacity: [0.15, 0.35, 0.15], scale: [1, 1.18, 1] }}
            transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
            style={{ background: 'radial-gradient(circle, rgba(245,158,11,0.45), transparent 70%)' }}
          />
          <div className="relative text-5xl font-black text-nf-amber" style={{ textShadow: '0 0 30px rgba(245, 158, 11, 0.3)' }}>
            {maxIdx}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function CNNDigitRecognition() {
  const [image, setImage] = useState(null);
  const [stages, setStages] = useState(null);
  const [activeStage, setActiveStage] = useState(0);
  const model = useMemo(() => createDigitCNN(), []);

  const processImage = useCallback((img) => {
    setImage(img);
    if (!img) return;

    // Check if canvas has content
    let hasContent = false;
    for (let i = 0; i < 28 && !hasContent; i++)
      for (let j = 0; j < 28 && !hasContent; j++)
        if (img[i][j] > 0.1) hasContent = true;

    if (!hasContent) { setStages(null); return; }

    const result = forwardPassCNN(img, model);
    setStages(result);
  }, [model]);

  const currentStage = stages ? stages[activeStage] : null;

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="module-header">
        <div className="flex items-center gap-3">
          <div className="module-icon" style={{ background: 'linear-gradient(135deg, #ec4899, #8b5cf6)' }}>
            <PenTool className="w-4 h-4 text-white" />
          </div>
          <div>
            <h2 className="text-headline text-base text-text-primary">CNN Digit Recognition</h2>
            <p className="text-[11px] text-text-tertiary mt-0.5">Draw a digit — watch a convolutional neural network process it layer by layer</p>
          </div>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left: Drawing + Output */}
        <div className="w-[340px] border-r border-white/[0.04] flex flex-col p-5 gap-5 overflow-y-auto">
          <DrawingCanvas onImageChange={processImage} canvasSize={280} />

          {stages && stages.length > 0 && (
            <OutputProbs probs={stages[stages.length - 1].values} />
          )}
        </div>

        {/* Right: Pipeline visualization */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Stage navigation */}
          {stages && (
            <div className="px-5 py-3 border-b border-white/[0.04] flex items-center gap-1 overflow-x-auto flex-shrink-0">
              {stages.map((stage, i) => {
                const layerColor = LAYER_COLORS[stage.type] || LAYER_COLORS.hidden;
                const isActive = activeStage === i;
                return (
                  <div key={i} className="flex items-center flex-shrink-0">
                    <button
                      onClick={() => setActiveStage(i)}
                      className={`relative px-2.5 py-1.5 rounded-lg text-[10px] font-medium transition-colors whitespace-nowrap ${isActive ? 'text-white' : 'text-text-ghost hover:text-text-secondary'
                        }`}
                    >
                      {isActive && (
                        <motion.span
                          layoutId="stagePillHighlight"
                          className="absolute inset-0 rounded-lg"
                          style={{ backgroundColor: layerColor.dim, boxShadow: `0 0 12px ${layerColor.dim}` }}
                          transition={{ type: 'spring', stiffness: 480, damping: 38 }}
                        />
                      )}
                      <span className="relative" style={isActive ? { color: layerColor.primary } : {}}>
                        {stage.name}
                      </span>
                    </button>
                    {i < stages.length - 1 && (
                      <ChevronRight className="w-3 h-3 text-text-ghost mx-0.5 flex-shrink-0" />
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* Stage detail */}
          <div className="flex-1 overflow-auto p-5">
            {!stages && (
              <div className="h-full flex items-center justify-center">
                <div className="text-center">
                  <PenTool className="w-10 h-10 text-text-ghost mx-auto mb-3" />
                  <p className="text-sm text-text-tertiary">Draw a digit on the canvas</p>
                  <p className="text-xs text-text-ghost mt-1">The CNN will process it in real-time</p>
                </div>
              </div>
            )}

            {currentStage && (
              <AnimatePresence mode="wait">
                <motion.div
                  key={activeStage}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.2 }}
                  className="space-y-4"
                >
                  {/* Stage info */}
                  <div className="callout callout-violet">
                    <div className="flex items-center gap-2 mb-1">
                      <Eye className="w-3.5 h-3.5" style={{ color: (LAYER_COLORS[currentStage.type] || LAYER_COLORS.hidden).primary }} />
                      <span className="text-xs font-bold text-text-primary">{currentStage.name}</span>
                    </div>
                    <p className="text-[11px] text-text-secondary">{currentStage.description}</p>
                  </div>

                  {/* Feature maps — rendered as a 3D wall of glowing filter screens */}
                  {currentStage.maps && (
                    <FeatureMapWall3D
                      maps={currentStage.maps}
                      mapSize={currentStage.mapSize}
                      type={currentStage.type}
                      stageName={currentStage.name}
                      maxDisplay={currentStage.type === 'input' ? 1 : 16}
                    />
                  )}

                  {/* Dense/flatten values — rendered as a 3D glowing node column */}
                  {currentStage.values && currentStage.type !== 'output' && (
                    <DenseColumn3D
                      values={currentStage.values}
                      title={currentStage.name}
                      type={currentStage.type}
                    />
                  )}

                  {/* Output */}
                  {currentStage.type === 'output' && (
                    <OutputProbs probs={currentStage.values} />
                  )}
                </motion.div>
              </AnimatePresence>
            )}
          </div>

          {/* Full pipeline overview */}
          {stages && (
            <div className="px-5 py-3 border-t border-white/[0.04] flex-shrink-0">
              <div className="text-label mb-2">Pipeline Overview</div>
              <div className="flex items-center gap-1">
                {stages.map((stage, i) => {
                  const layerColor = LAYER_COLORS[stage.type] || LAYER_COLORS.hidden;
                  const width = stage.type === 'input' ? 28 :
                    stage.type === 'output' ? 24 :
                      stage.maps ? Math.max(12, stage.maps.length * 2) : 16;
                  const isActive = activeStage === i;
                  return (
                    <div key={i} className="flex items-center">
                      <motion.div
                        className="rounded-sm cursor-pointer transition-all"
                        style={{
                          width,
                          height: 20,
                          backgroundColor: isActive ? layerColor.primary : layerColor.dim,
                          opacity: isActive ? 1 : 0.5,
                          boxShadow: isActive ? `0 0 10px ${layerColor.primary}88` : 'none',
                        }}
                        onClick={() => setActiveStage(i)}
                        whileHover={{ opacity: 0.8, scale: 1.1 }}
                        title={stage.name}
                      />
                      {i < stages.length - 1 && (
                        <div className="w-2 h-[1px] bg-text-ghost" />
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}