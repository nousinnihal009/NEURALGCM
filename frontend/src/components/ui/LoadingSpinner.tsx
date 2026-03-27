export function LoadingSpinner({ size = 16 }: { size?: number }) {
  return (
    <div
      style={{ width: size, height: size }}
      className="rounded-full border-2 border-border
                 border-t-accent-blue animate-spin"
    />
  );
}
