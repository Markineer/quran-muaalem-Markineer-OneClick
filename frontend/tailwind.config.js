/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Clean white background palette
        'night': {
          950: '#ffffff',
          900: '#f8f9fa',
          800: '#f0f2f5',
          700: '#e8ebed',
          600: '#dfe3e6',
        },
        // Fresh green accents (replacing gold)
        'gold': {
          100: '#f0fdf4',
          200: '#dcfce7',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
        },
        // Spiritual emerald
        'emerald': {
          400: '#34d399',
          500: '#10b981',
          600: '#059669',
        },
        // Error crimson
        'crimson': {
          400: '#f87171',
          500: '#ef4444',
          600: '#dc2626',
        },
        // Uncertain amber
        'amber': {
          400: '#fbbf24',
          500: '#f59e0b',
        },
      },
      fontFamily: {
        'arabic': ['"Amiri"', '"Scheherazade New"', '"Traditional Arabic"', 'serif'],
        'display': ['"Playfair Display"', 'serif'],
        'body': ['"Source Sans 3"', 'sans-serif'],
      },
      backgroundImage: {
        'geometric-pattern': `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%2322c55e' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
        'radial-glow': 'radial-gradient(ellipse at center, rgba(34,197,94,0.08) 0%, transparent 70%)',
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'ink-reveal': 'inkReveal 0.6s ease-out forwards',
        'fade-up': 'fadeUp 0.5s ease-out forwards',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        inkReveal: {
          '0%': { opacity: '0.1', filter: 'blur(2px)' },
          '100%': { opacity: '1', filter: 'blur(0)' },
        },
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        glow: {
          '0%': { boxShadow: '0 0 20px rgba(34,197,94,0.2)' },
          '100%': { boxShadow: '0 0 40px rgba(34,197,94,0.4)' },
        },
      },
    },
  },
  plugins: [],
}
