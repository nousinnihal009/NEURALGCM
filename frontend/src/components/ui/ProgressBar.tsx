import { motion } from "framer-motion";

interface Props { value: number; }

export function ProgressBar({ value }: Props) {
  return (
    <div>
      <div className="flex justify-between text-xs text-text-muted mb-1.5">
        <span>Running NeuralGCM…</span>
        <span>{Math.round(value)}%</span>
      </div>
      <div className="w-full h-1.5 bg-bg-tertiary rounded-full overflow-hidden">
        <motion.div
          className="h-full bg-accent-blue rounded-full"
          initial={{ width: "0%" }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 0.4, ease: "easeOut" }}
        />
      </div>
      <div className="flex justify-between text-xs text-text-muted mt-2">
        <span>ERA5 → NeuralGCM 2.8°</span>
        <span>~45s on CPU</span>
      </div>
    </div>
  );
}
