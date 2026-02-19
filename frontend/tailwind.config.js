/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["'Space Grotesk'", "system-ui", "sans-serif"],
        body: ["'Manrope'", "system-ui", "sans-serif"],
      },
      boxShadow: {
        soft: "0 20px 60px -28px rgba(10, 24, 45, 0.35)",
      },
      keyframes: {
        wave: {
          "0%, 100%": { transform: "scaleY(0.35)" },
          "50%": { transform: "scaleY(1)" },
        },
        pulseGlow: {
          "0%, 100%": { opacity: "0.45", transform: "scale(0.96)" },
          "50%": { opacity: "0.95", transform: "scale(1.04)" },
        },
      },
      animation: {
        wave: "wave 1s ease-in-out infinite",
        pulseGlow: "pulseGlow 1.2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
