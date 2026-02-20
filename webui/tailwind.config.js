/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0B0F14',
        panel: '#111827',
        panelAlt: '#0F172A',
        border: 'rgba(255,255,255,0.06)',
        textPrimary: '#E5E7EB',
        textSecondary: '#9CA3AF',
        bull: '#0ECB81',
        bear: '#F6465D',
        binanceYellow: '#F0B90B',
      },
    },
  },
  plugins: [],
}
