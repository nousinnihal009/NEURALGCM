import { motion } from "framer-motion";

interface Props { isPolling: boolean; }

export function ForecastMarker({ isPolling }: Props) {
  return (
    <div className="relative flex items-center justify-center">
      {isPolling && (
        <>
          <motion.div
            className="absolute rounded-full border-2 border-accent-orange/60"
            initial={{ width: 20, height: 20, opacity: 0.8 }}
            animate={{ width: 60, height: 60, opacity: 0 }}
            transition={{ duration: 1.5, repeat: Infinity, ease: "easeOut" }}
          />
          <motion.div
            className="absolute rounded-full border border-accent-orange/40"
            initial={{ width: 20, height: 20, opacity: 0.6 }}
            animate={{ width: 80, height: 80, opacity: 0 }}
            transition={{
              duration: 1.5, repeat: Infinity,
              ease: "easeOut", delay: 0.4
            }}
          />
        </>
      )}

      <motion.div
        initial={{ scale: 0, y: -10 }}
        animate={{ scale: 1, y: 0 }}
        transition={{ type: "spring", stiffness: 400, damping: 20 }}
        className="relative"
      >
        <svg width="28" height="36" viewBox="0 0 28 36" fill="none">
          <path
            d="M14 0C6.268 0 0 6.268 0 14c0 10.5 14 22 14 22S28 24.5 28 14C28 6.268 21.732 0 14 0z"
            fill={isPolling ? "#EF9F27" : "#F78166"}
          />
          <circle cx="14" cy="14" r="6" fill="white" opacity="0.9" />
          {isPolling && (
            <circle cx="14" cy="14" r="4" fill="#EF9F27">
              <animate
                attributeName="r"
                values="3;5;3"
                dur="1s"
                repeatCount="indefinite"
              />
            </circle>
          )}
        </svg>
      </motion.div>
    </div>
  );
}
