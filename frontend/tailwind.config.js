/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        red: {
          DEFAULT: '#e63939',
          dark: '#b52828',
          glow: 'rgba(230,57,57,0.35)',
        },
        surface: {
          DEFAULT: '#0d0d0d',
          2: '#111111',
          border: 'rgba(255,255,255,0.07)',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
