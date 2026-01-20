import React, { useState } from 'react'
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { BookOpen, Mic, Settings, Sparkles } from 'lucide-react'

import HomePage from './pages/HomePage'
import HarakatTrainer from './pages/HarakatTrainer'
import RealtimeTrainer from './pages/RealtimeTrainer'
import SettingsPage from './pages/SettingsPage'

function App() {
  const [moshafSettings, setMoshafSettings] = useState({
    rewaya: 'hafs',
    madd_monfasel_len: 4,
    madd_mottasel_len: 4,
    madd_mottasel_waqf: 4,
    madd_aared_len: 4,
  })

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-night-950 relative overflow-hidden">
        {/* Background effects */}
        <div className="fixed inset-0 bg-geometric-pattern opacity-30 pointer-events-none" />
        <div className="fixed inset-0 bg-radial-glow pointer-events-none" />

        {/* Ambient glow orbs */}
        <div className="fixed top-20 right-20 w-96 h-96 bg-gold-500/5 rounded-full blur-3xl pointer-events-none" />
        <div className="fixed bottom-20 left-20 w-80 h-80 bg-emerald-500/5 rounded-full blur-3xl pointer-events-none" />

        {/* Main content */}
        <div className="relative z-10">
          {/* Header */}
          <Header />

          {/* Page content */}
          <main className="container mx-auto px-4 py-8">
            <AnimatePresence mode="wait">
              <Routes>
                <Route path="/" element={<HomePage />} />
                <Route
                  path="/harakat"
                  element={<HarakatTrainer settings={moshafSettings} />}
                />
                <Route
                  path="/realtime"
                  element={<RealtimeTrainer settings={moshafSettings} />}
                />
                <Route
                  path="/settings"
                  element={
                    <SettingsPage
                      settings={moshafSettings}
                      onSettingsChange={setMoshafSettings}
                    />
                  }
                />
              </Routes>
            </AnimatePresence>
          </main>

          {/* Footer */}
          <Footer />
        </div>
      </div>
    </BrowserRouter>
  )
}

function Header() {
  const navItems = [
    { to: '/', icon: BookOpen, label: 'الرئيسية', labelEn: 'Home' },
    { to: '/harakat', icon: Sparkles, label: 'تدريب الحركات', labelEn: 'Harakat' },
    { to: '/realtime', icon: Mic, label: 'تدريب مباشر', labelEn: 'Live' },
    { to: '/settings', icon: Settings, label: 'الإعدادات', labelEn: 'Settings' },
  ]

  return (
    <header className="sticky top-0 z-50 backdrop-blur-xl bg-night-950/80 border-b border-white/5">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-20">
          {/* Logo */}
          <NavLink to="/" className="flex items-center gap-4 group">
            <div className="relative">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-gold-400 to-gold-600 flex items-center justify-center shadow-lg shadow-gold-500/20 group-hover:shadow-gold-500/40 transition-shadow">
                <span className="font-arabic text-2xl text-night-950 font-bold">ق</span>
              </div>
              <div className="absolute -inset-1 bg-gradient-to-br from-gold-400 to-gold-600 rounded-xl opacity-20 blur group-hover:opacity-40 transition-opacity" />
            </div>
            <div className="hidden sm:block">
              <h1 className="font-display text-xl text-white font-semibold tracking-wide">
                المعلم القرآني
              </h1>
              <p className="text-xs text-white/40 font-body">Quran Muaalem</p>
            </div>
          </NavLink>

          {/* Navigation */}
          <nav className="flex items-center gap-1">
            {navItems.map(({ to, icon: Icon, label, labelEn }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  `relative flex items-center gap-2 px-4 py-2 rounded-xl transition-all duration-300 ${
                    isActive
                      ? 'text-gold-400'
                      : 'text-white/60 hover:text-white hover:bg-white/5'
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    <Icon size={18} />
                    <span className="hidden md:inline text-sm font-medium">
                      {label}
                    </span>
                    {isActive && (
                      <motion.div
                        layoutId="activeTab"
                        className="absolute inset-0 bg-gold-500/10 border border-gold-500/20 rounded-xl -z-10"
                        transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
                      />
                    )}
                  </>
                )}
              </NavLink>
            ))}
          </nav>
        </div>
      </div>
    </header>
  )
}

function Footer() {
  return (
    <footer className="border-t border-white/5 py-8 mt-12">
      <div className="container mx-auto px-4 text-center">
        <p className="text-white/30 text-sm font-body">
          Built with{' '}
          <span className="text-gold-500">Quran Muaalem</span>
          {' '}| Model: obadx/muaalem-model-v3_2
        </p>
        <p className="text-white/20 text-xs mt-2 font-arabic">
          اللهم علّمنا ما ينفعنا وانفعنا بما علّمتنا
        </p>
      </div>
    </footer>
  )
}

export default App
