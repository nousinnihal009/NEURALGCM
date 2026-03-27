/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: {
          primary:   "#0d1117",
          secondary: "#161b22",
          tertiary:  "#21262d",
        },
        border: {
          DEFAULT: "#30363d",
          hover:   "#58A6FF",
        },
        accent: {
          blue:   "#58A6FF",
          green:  "#3FB950",
          orange: "#F78166",
          yellow: "#EF9F27",
          purple: "#BD8EE6",
          teal:   "#1D9E75",
        },
        text: {
          primary:   "#e6edf3",
          secondary: "#8B949E",
          muted:     "#484F58",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4,0,0.6,1) infinite",
        "fade-in":    "fadeIn 0.3s ease-in-out",
        "slide-up":   "slideUp 0.4s ease-out",
      },
      keyframes: {
        fadeIn:  { "0%": { opacity: "0" }, "100%": { opacity: "1" } },
        slideUp: { "0%": { transform: "translateY(20px)", opacity: "0" },
                   "100%": { transform: "translateY(0)", opacity: "1" } },
      },
    },
  },
  plugins: [],
};
