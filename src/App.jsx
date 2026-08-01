// NeuroForge — Main App Shell (Redesigned)

import { AnimatePresence, motion } from 'framer-motion';
import Sidebar from './components/ui/Sidebar';
import HomePage from './modules/Home/HomePage';
import NetworkArchitect from './modules/NetworkArchitect/NetworkArchitect';
import TrainingArena from './modules/TrainingArena/TrainingArena';
import RepresentationExplorer from './modules/RepresentationExplorer/RepresentationExplorer';
import LossLandscape from './modules/LossLandscape/LossLandscape';
import AttentionLab from './modules/AttentionLab/AttentionLab';
import Playground from './modules/Playground/Playground';
import CNNDigitRecognition from './modules/CNNDigitRecognition/CNNDigitRecognition';
import Documentation from './modules/Documentation/Documentation';
import useStore from './stores/useStore';

const moduleComponents = {
  home: HomePage,
  architect: NetworkArchitect,
  training: TrainingArena,
  cnn: CNNDigitRecognition,
  representations: RepresentationExplorer,
  landscape: LossLandscape,
  attention: AttentionLab,
  playground: Playground,
  docs: Documentation,
};

export default function App() {
  const { activeModule } = useStore();
  const ModuleComponent = moduleComponents[activeModule] || HomePage;

  return (
    <div className="h-screen w-screen flex overflow-hidden" style={{ background: 'var(--color-abyss)' }}>
      <Sidebar />

      <main className="flex-1 overflow-hidden relative">
        {/* Ambient light effects */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden z-0">
          <div className="absolute -top-60 -right-60 w-[500px] h-[500px] rounded-full opacity-[0.015]"
            style={{ background: 'radial-gradient(circle, #8b5cf6, transparent 70%)' }} />
          <div className="absolute -bottom-60 -left-60 w-[500px] h-[500px] rounded-full opacity-[0.01]"
            style={{ background: 'radial-gradient(circle, #06b6d4, transparent 70%)' }} />
        </div>

        <AnimatePresence mode="wait">
          <motion.div
            key={activeModule}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
            className="h-full relative z-10"
          >
            <ModuleComponent />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
