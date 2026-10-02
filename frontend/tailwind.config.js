/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#FFFFFF",
        surface: "#F9FAFB",
        border: "#E5E7EB",
        primary: "#111827",
        secondary: "#6B7280",
        accent: {
          green: "#16A34A",
          red: "#DC2626",
          amber: "#D97706",
        }
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
      },
      borderRadius: {
        DEFAULT: "8px",
      }
    },
  },
  plugins: [],
}
