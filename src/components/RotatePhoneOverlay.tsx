import React from 'react';
import { RotateCw, Smartphone } from 'lucide-react';

interface RotatePhoneOverlayProps {
  onDismiss: () => void;
}

export const RotatePhoneOverlay: React.FC<RotatePhoneOverlayProps> = ({
  onDismiss,
}) => {
  return (
    <div
      dir="rtl"
      className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-[#070b14]/95 backdrop-blur-md p-6 text-center safe-area-pad"
    >
      <div className="relative mb-6 flex items-center justify-center">
        <div className="w-24 h-24 rounded-3xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center shadow-[0_0_40px_rgba(6,182,212,0.2)]">
          <Smartphone className="w-11 h-11 text-cyan-300 animate-[spin_2.8s_ease-in-out_infinite] [animation-direction:alternate]" />
        </div>
        <div className="absolute -bottom-2 -left-2 w-9 h-9 rounded-full bg-emerald-500/20 border border-emerald-400/40 flex items-center justify-center">
          <RotateCw className="w-4 h-4 text-emerald-300" />
        </div>
      </div>

      <h2 className="text-xl font-bold text-white mb-2">
        סובב את הטלפון לרוחב
      </h2>
      <p className="text-sm text-slate-300 max-w-xs leading-relaxed mb-8">
        הלוח מותאם למשחק נוח במסך מלא במצב שוכב
      </p>

      <button
        type="button"
        onClick={onDismiss}
        className="px-4 py-2.5 rounded-xl bg-slate-900/90 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-700/80 text-xs font-semibold transition-colors cursor-pointer min-h-[44px]"
      >
        המשך במצב אנכי בכל זאת
      </button>
    </div>
  );
};
