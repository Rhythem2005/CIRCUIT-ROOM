// Circuit Room — Loss Landscape (Redesigned with new design system)

import { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { motion } from 'framer-motion';
import { Mountain, RotateCcw } from 'lucide-react';
import * as THREE from 'three';

const landscapes = {
  simple_bowl: { name: 'Simple Bowl', desc: 'Convex — easy for any optimizer',
    fn: (x, y) => x * x + y * y, range: [-2, 2], minima: [[0, 0]] },
  saddle: { name: 'Saddle Point', desc: 'Traps naive optimizers',
    fn: (x, y) => x * x - y * y, range: [-2, 2], minima: [] },
  ravine: { name: 'Ravine', desc: 'Challenges SGD without momentum',
    fn: (x, y) => 0.1 * x * x + 10 * y * y, range: [-3, 3], minima: [[0, 0]] },
  multi_minima: { name: 'Multiple Minima', desc: 'Starting point matters',
    fn: (x, y) => Math.sin(2*x)*Math.cos(2*y) + 0.1*(x*x+y*y), range: [-3, 3], minima: [] },
  rosenbrock: { name: 'Rosenbrock', desc: 'Banana-shaped valley',
    fn: (x, y) => (1-x)**2 + 100*(y-x*x)**2, range: [-2, 2], minima: [[1, 1]], scale: 0.001 },
  beale: { name: 'Beale Function', desc: 'Complex flat regions',
    fn: (x, y) => (1.5-x+x*y)**2 + (2.25-x+x*y*y)**2 + (2.625-x+x*y**3)**2,
    range: [-3, 3], minima: [[3, 0.5]], scale: 0.0005 },
};

function LossSurface({ landscape, colorScheme }) {
  const res = 80;
  const { geometry } = useMemo(() => {
    const geo = new THREE.PlaneGeometry(6, 6, res-1, res-1);
    const pos = geo.attributes.position.array;
    const range = landscape.range;
    const scale = landscape.scale || 1;
    const colors = new Float32Array(pos.length);
    const zVals = [];
    let minZ = Infinity, maxZ = -Infinity;

    for (let i = 0; i < pos.length; i += 3) {
      const x = (pos[i]/3)*(range[1]-range[0])/2;
      const y = (pos[i+1]/3)*(range[1]-range[0])/2;
      let z = landscape.fn(x, y)*scale;
      z = Math.min(z, 10);
      pos[i+2] = z;
      zVals.push(z);
      minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
    }

    const zRange = maxZ - minZ || 1;
    for (let i = 0; i < zVals.length; i++) {
      const t = (zVals[i] - minZ) / zRange;
      let r, g, b;
      if (colorScheme === 'thermal') {
        r = Math.min(1, t*2); g = Math.max(0, 1-Math.abs(t-0.5)*4); b = Math.max(0, 1-t*2);
      } else if (colorScheme === 'ocean') {
        r = t*0.3; g = 0.1+t*0.5; b = 0.4+(1-t)*0.6;
      } else { // violet-amber
        r = 0.04 + t*0.85; g = 0.04 + (1-t)*0.15 + t*0.45; b = 0.15 + (1-t)*0.75;
      }
      colors[i*3]=r; colors[i*3+1]=g; colors[i*3+2]=b;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    return { geometry: geo };
  }, [landscape, colorScheme]);

  return (
    <mesh rotation={[-Math.PI/2,0,0]} geometry={geometry}>
      <meshStandardMaterial vertexColors side={THREE.DoubleSide} transparent opacity={0.85} roughness={0.4} metalness={0.1} />
    </mesh>
  );
}

function OptimizerPath({ points, color }) {
  const geo = useMemo(() => {
    if (points.length < 2) return null;
    const curve = new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(p[0], p[2]+0.05, -p[1])));
    return new THREE.TubeGeometry(curve, points.length*4, 0.03, 8, false);
  }, [points]);
  if (!geo) return null;
  return <mesh geometry={geo}><meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5} /></mesh>;
}

function AnimatedBall({ position, color }) {
  const ref = useRef();
  useFrame((s) => { if (ref.current) ref.current.position.y = position[2]+0.1+Math.sin(s.clock.elapsedTime*3)*0.02; });
  return (
    <mesh ref={ref} position={[position[0], position[2]+0.1, -position[1]]}>
      <sphereGeometry args={[0.06,16,16]} /><meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.8} />
    </mesh>
  );
}

function Scene({ landscape, paths, colorScheme }) {
  return (
    <>
      <ambientLight intensity={0.3} />
      <directionalLight position={[5,10,5]} intensity={0.7} />
      <pointLight position={[-5,5,-5]} intensity={0.25} color="#8b5cf6" />
      <LossSurface landscape={landscape} colorScheme={colorScheme} />
      {paths.map((p,i) => <OptimizerPath key={i} points={p.points} color={p.color} />)}
      {paths.map((p,i) => p.points.length > 0 && <AnimatedBall key={`b-${i}`} position={p.points[p.points.length-1]} color={p.color} />)}
      <gridHelper args={[6,20,'rgba(255,255,255,0.03)','rgba(255,255,255,0.01)']} position={[0,-0.01,0]} />
      <OrbitControls enableDamping dampingFactor={0.05} minDistance={2} maxDistance={15} maxPolarAngle={Math.PI/2.1} />
    </>
  );
}

function simulate(landscape, sx, sy, lr, steps, type = 'sgd') {
  const pts = []; let x=sx, y=sy; const scale=landscape.scale||1; const eps=0.001; const range=landscape.range;
  let mX=0,mY=0,vX=0,vY=0,velX=0,velY=0;
  for (let t=1; t<=steps; t++) {
    const z = landscape.fn(x,y)*scale;
    pts.push([(x/((range[1]-range[0])/2))*3, (y/((range[1]-range[0])/2))*3, Math.min(z,10)]);
    const dx = (landscape.fn(x+eps,y)-landscape.fn(x-eps,y))/(2*eps)*scale;
    const dy = (landscape.fn(x,y+eps)-landscape.fn(x,y-eps))/(2*eps)*scale;
    if (type==='adam') {
      mX=0.9*mX+0.1*dx; mY=0.9*mY+0.1*dy; vX=0.999*vX+0.001*dx*dx; vY=0.999*vY+0.001*dy*dy;
      x -= lr*mX/(1-0.9**t)/(Math.sqrt(vX/(1-0.999**t))+1e-8);
      y -= lr*mY/(1-0.9**t)/(Math.sqrt(vY/(1-0.999**t))+1e-8);
    } else if (type==='momentum') {
      velX=0.9*velX-lr*dx; velY=0.9*velY-lr*dy; x+=velX; y+=velY;
    } else if (type==='rmsprop') {
      vX=0.9*vX+0.1*dx*dx; vY=0.9*vY+0.1*dy*dy;
      x-=lr*dx/(Math.sqrt(vX)+1e-8); y-=lr*dy/(Math.sqrt(vY)+1e-8);
    } else { x-=lr*dx; y-=lr*dy; }
    x=Math.max(range[0],Math.min(range[1],x)); y=Math.max(range[0],Math.min(range[1],y));
  }
  return pts;
}

export default function LossLandscape() {
  const [sel, setSel] = useState('simple_bowl');
  const [cs, setCs] = useState('default');
  const [lr, setLr] = useState(0.05);
  const [steps, setSteps] = useState(100);
  const [show, setShow] = useState({ sgd:true, adam:true, mom:false, rms:false });
  const [startPos] = useState([-1.5, 1.5]);
  const [v, setV] = useState(0);

  const landscape = landscapes[sel];
  const paths = useMemo(() => {
    const r = []; const [sx,sy] = startPos;
    if (show.sgd) r.push({ name:'SGD', color:'#ef4444', points:simulate(landscape,sx,sy,lr,steps,'sgd') });
    if (show.adam) r.push({ name:'Adam', color:'#8b5cf6', points:simulate(landscape,sx,sy,lr*0.5,steps,'adam') });
    if (show.mom) r.push({ name:'Momentum', color:'#f59e0b', points:simulate(landscape,sx,sy,lr,steps,'momentum') });
    if (show.rms) r.push({ name:'RMSProp', color:'#06b6d4', points:simulate(landscape,sx,sy,lr*0.3,steps,'rmsprop') });
    return r;
  }, [sel, lr, steps, show, startPos, v]);

  return (
    <div className="h-full flex flex-col">
      <div className="module-header">
        <div className="flex items-center gap-3">
          <div className="module-icon" style={{ background: 'linear-gradient(135deg, #34d399, #10b981)' }}>
            <Mountain className="w-4 h-4 text-white" />
          </div>
          <div>
            <h2 className="text-headline text-base text-text-primary">Loss Landscape Explorer</h2>
            <p className="text-[11px] text-text-tertiary mt-0.5">Navigate the terrain optimizers must traverse</p>
          </div>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        <div className="flex-1 relative" style={{ background: 'var(--color-void)' }}>
          <Canvas camera={{ position:[4,4,4], fov:50 }}><Scene landscape={landscape} paths={paths} colorScheme={cs} /></Canvas>
          <div className="absolute bottom-4 left-4 sf-glass rounded-xl px-3 py-2 space-y-1">
            {paths.map((p,i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor:p.color }} />
                <span className="text-text-secondary">{p.name}</span>
              </div>
            ))}
          </div>
          <div className="absolute top-4 left-4 sf-glass rounded-xl px-3 py-2">
            <div className="text-xs font-semibold text-text-primary">{landscape.name}</div>
            <div className="text-[10px] text-text-tertiary">{landscape.desc}</div>
          </div>
        </div>

        <div className="w-56 border-l border-white/[0.04] overflow-y-auto" style={{ background: 'var(--color-deep)' }}>
          <div className="panel-section">
            <label className="text-label">Landscape</label>
            <div className="mt-2 space-y-1">
              {Object.entries(landscapes).map(([k,l]) => (
                <button key={k} onClick={() => setSel(k)}
                  className={`w-full text-left px-3 py-1.5 rounded-lg text-xs transition-colors ${
                    sel===k ? 'sf-raised text-nf-emerald' : 'text-text-ghost hover:text-text-secondary'
                  }`}>{l.name}</button>
              ))}
            </div>
          </div>
          <div className="panel-section">
            <label className="text-label">Optimizers</label>
            <div className="mt-2 space-y-1.5">
              {[
                { k:'sgd', l:'SGD', c:'#ef4444' }, { k:'adam', l:'Adam', c:'#8b5cf6' },
                { k:'mom', l:'Momentum', c:'#f59e0b' }, { k:'rms', l:'RMSProp', c:'#06b6d4' },
              ].map(o => (
                <label key={o.k} className="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" checked={show[o.k]} onChange={(e) => setShow(s => ({...s,[o.k]:e.target.checked}))} className="accent-purple-500" />
                  <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor:o.c }} />
                  <span className="text-xs text-text-secondary">{o.l}</span>
                </label>
              ))}
            </div>
          </div>
          <div className="panel-section">
            <label className="text-[10px] text-text-tertiary flex justify-between">
              <span>Learning Rate</span><span className="text-mono text-nf-emerald">{lr.toFixed(3)}</span>
            </label>
            <input type="range" min="0.001" max="0.5" step="0.001" value={lr}
              onChange={(e) => setLr(parseFloat(e.target.value))} className="mt-1.5 w-full" />
          </div>
          <div className="panel-section">
            <label className="text-[10px] text-text-tertiary flex justify-between">
              <span>Steps</span><span className="text-mono text-nf-emerald">{steps}</span>
            </label>
            <input type="range" min="10" max="500" step="10" value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value))} className="mt-1.5 w-full" />
          </div>
          <div className="panel-section">
            <label className="text-label">Colors</label>
            <div className="mt-2 flex gap-2">
              {['default','thermal','ocean'].map(s => (
                <button key={s} onClick={() => setCs(s)}
                  className={`px-3 py-1 rounded-lg text-xs ${cs===s ? 'sf-raised text-text-primary' : 'text-text-ghost hover:text-text-secondary'}`}>
                  {s}
                </button>
              ))}
            </div>
          </div>
          <div className="panel-section">
            <button onClick={() => setV(n=>n+1)} className="btn-ghost w-full text-xs flex items-center justify-center gap-1.5">
              <RotateCcw className="w-3 h-3" /> Re-run
            </button>
          </div>
          <div className="panel-section">
            <div className="callout callout-emerald">
              <p className="text-[10px] text-text-secondary leading-relaxed">
                💡 The loss landscape shows how loss varies with parameters. Optimizers navigate this terrain to find
                <strong className="text-nf-emerald"> minimum values</strong>. Adam adapts step sizes; Momentum carries inertia.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
