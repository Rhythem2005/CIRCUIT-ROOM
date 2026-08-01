// NeuroForge — Concept Playground Module
// Interactive micro-labs for individual deep learning concepts

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Beaker, Waves, Grid3x3, TrendingDown, Shield } from 'lucide-react';
import ActivationExplorer from './ActivationExplorer';
import ConvolutionViz from './ConvolutionViz';
import GradientAnalyzer from './GradientAnalyzer';
import RegularizationLab from './RegularizationLab';

const subModules = [
  { id: 'activations', name: 'Activation Functions', icon: Waves, color: 'from-purple-500 to-indigo-500', description: 'Interactive graphs of activation functions and their effects' },
  { id: 'convolution', name: 'Convolution Visualizer', icon: Grid3x3, color: 'from-blue-500 to-cyan-500', description: 'Step-by-step kernel sliding animation' },
  { id: 'gradients', name: 'Gradient Flow', icon: TrendingDown, color: 'from-amber-500 to-red-500', description: 'Vanishing & exploding gradients explained' },
  { id: 'regularization', name: 'Regularization Lab', icon: Shield, color: 'from-emerald-500 to-green-500', description: 'Dropout, L1/L2, and overfitting visualization' },
];

export default function Playground() {
  const [activeSubModule, setActiveSubModule] = useState('activations');

  const renderSubModule = () => {
    switch (activeSubModule) {
      case 'activations': return <ActivationExplorer />;
      case 'convolution': return <ConvolutionViz />;
      case 'gradients': return <GradientAnalyzer />;
      case 'regularization': return <RegularizationLab />;
      default: return <ActivationExplorer />;
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="px-5 py-3 border-b border-white/5">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center">
            <Beaker className="w-4 h-4 text-white" />
          </div>
          Concept Playground
        </h2>
        <p className="text-xs text-gray-500 mt-0.5">Interactive micro-labs for core deep learning concepts</p>
      </div>

      {/* Sub-module tabs */}
      <div className="px-5 pt-3 flex gap-2 flex-wrap">
        {subModules.map((sm) => {
          const Icon = sm.icon;
          const isActive = activeSubModule === sm.id;
          return (
            <motion.button
              key={sm.id}
              onClick={() => setActiveSubModule(sm.id)}
              className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium transition-all ${isActive
                ? 'bg-white/10 text-white border border-white/10'
                : 'text-gray-500 hover:text-white hover:bg-white/5'
                }`}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className={`w-5 h-5 rounded-md flex items-center justify-center ${isActive ? `bg-gradient-to-r ${sm.color}` : 'bg-white/5'
                }`}>
                <Icon className="w-3 h-3 text-white" />
              </div>
              {sm.name}
            </motion.button>
          );
        })}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {renderSubModule()}
      </div>
    </div>
  );
}
