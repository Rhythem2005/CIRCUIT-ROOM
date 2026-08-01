// Circuit Room — Redesigned Sidebar

import { motion } from 'framer-motion';
import {
  Brain, Play, Layers, Mountain, Sparkles, Beaker,
  Home, ChevronLeft, ChevronRight, PenTool, BookOpen
} from 'lucide-react';
import { useState } from 'react';
import useStore from '../../stores/useStore';

const modules = [
  { id: 'home', name: 'Home', icon: Home, color: '#8b5cf6' },
  { id: 'architect', name: 'Architect', icon: Brain, color: '#7c3aed' },
  { id: 'training', name: 'Training', icon: Play, color: '#f59e0b' },
  { id: 'cnn', name: 'CNN Digits', icon: PenTool, color: '#ec4899' },
  { id: 'representations', name: 'Representations', icon: Layers, color: '#06b6d4' },
  { id: 'landscape', name: 'Loss Landscape', icon: Mountain, color: '#10b981' },
  { id: 'attention', name: 'Attention', icon: Sparkles, color: '#f43f5e' },
  { id: 'playground', name: 'Playground', icon: Beaker, color: '#6366f1' },
  { id: 'docs', name: 'Documentation', icon: BookOpen, color: '#64748b' },
];

export default function Sidebar() {
  const { activeModule, setActiveModule } = useStore();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <motion.nav
      className="h-full flex flex-col border-r border-white/[0.04] relative z-30 select-none"
      style={{ background: 'var(--color-deep)' }}
      animate={{ width: collapsed ? 56 : 200 }}
      transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
    >
      {/* Logo */}
      <div className="h-14 flex items-center gap-2.5 px-3.5 border-b border-white/[0.04] flex-shrink-0">
        <div className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
          style={{ background: 'linear-gradient(135deg, #7c3aed, #06b6d4)' }}>
          <Brain className="w-4 h-4 text-white" />
        </div>
        {!collapsed && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <div className="text-[13px] font-bold grad-text leading-none">Circuit Room</div>
            <div className="text-[9px] text-text-ghost mt-0.5 leading-none">Neural Simulator</div>
          </motion.div>
        )}
      </div>

      {/* Nav items */}
      <div className="flex-1 py-2 px-1.5 space-y-0.5 overflow-y-auto">
        {modules.map((mod) => {
          const Icon = mod.icon;
          const active = activeModule === mod.id;

          return (
            <motion.button
              key={mod.id}
              onClick={() => setActiveModule(mod.id)}
              className="w-full flex items-center gap-2.5 rounded-lg relative transition-colors duration-150"
              style={{
                padding: collapsed ? '8px 0' : '7px 10px',
                justifyContent: collapsed ? 'center' : 'flex-start',
                color: active ? '#e8e8f0' : '#5d6186',
                background: active ? 'rgba(139, 92, 246, 0.08)' : 'transparent',
              }}
              whileHover={{ backgroundColor: active ? 'rgba(139, 92, 246, 0.08)' : 'rgba(139, 92, 246, 0.04)' }}
              whileTap={{ scale: 0.97 }}
            >
              {active && (
                <motion.div
                  layoutId="sidebar-active"
                  className="absolute left-0 top-1 bottom-1 w-[3px] rounded-r-full"
                  style={{ backgroundColor: mod.color }}
                  transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                />
              )}
              <div
                className="w-5 h-5 rounded-md flex items-center justify-center flex-shrink-0 transition-colors"
                style={{
                  backgroundColor: active ? mod.color + '20' : 'transparent',
                }}
              >
                <Icon className="w-3.5 h-3.5" style={{ color: active ? mod.color : 'inherit' }} />
              </div>
              {!collapsed && (
                <span className="text-[12px] font-medium truncate">{mod.name}</span>
              )}
            </motion.button>
          );
        })}
      </div>

      {/* Collapse toggle */}
      <div className="px-1.5 py-2 border-t border-white/[0.04] flex-shrink-0">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-full flex items-center justify-center py-1.5 rounded-lg text-text-ghost hover:text-text-secondary transition-colors"
        >
          {collapsed ? <ChevronRight className="w-3.5 h-3.5" /> : <ChevronLeft className="w-3.5 h-3.5" />}
        </button>
      </div>
    </motion.nav>
  );
}
