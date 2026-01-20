import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Save, RotateCcw, Check, BookOpen, Sliders } from 'lucide-react'

export default function SettingsPage({ settings, onSettingsChange }) {
  const [localSettings, setLocalSettings] = useState(settings)
  const [saved, setSaved] = useState(false)

  const handleChange = (key, value) => {
    setLocalSettings(prev => ({ ...prev, [key]: value }))
    setSaved(false)
  }

  const handleSave = () => {
    onSettingsChange(localSettings)
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const handleReset = () => {
    const defaults = {
      rewaya: 'hafs',
      madd_monfasel_len: 4,
      madd_mottasel_len: 4,
      madd_mottasel_waqf: 4,
      madd_aared_len: 4,
    }
    setLocalSettings(defaults)
    onSettingsChange(defaults)
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="max-w-3xl mx-auto"
    >
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="font-display text-4xl text-white mb-4">
          <span className="text-gradient-gold">إعدادات</span> المصحف
        </h1>
        <p className="text-white/50">
          قم بتعديل خصائص المصحف حسب التلاوة المطلوبة
        </p>
      </div>

      <div className="space-y-8">
        {/* Rewaya Selection */}
        <div className="glass-card p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-gold-500/20 flex items-center justify-center">
              <BookOpen className="text-gold-400" size={20} />
            </div>
            <div>
              <h3 className="font-display text-lg text-white">الرواية</h3>
              <p className="text-white/40 text-sm">اختر رواية التلاوة</p>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3">
            {['hafs', 'warsh', 'qalun'].map((rewaya) => (
              <button
                key={rewaya}
                onClick={() => handleChange('rewaya', rewaya)}
                className={`px-4 py-3 rounded-xl text-center transition-all ${
                  localSettings.rewaya === rewaya
                    ? 'bg-gold-500/20 border border-gold-500/30 text-gold-400'
                    : 'bg-white/5 border border-transparent hover:bg-white/10 text-white/70'
                }`}
              >
                <span className="font-arabic text-lg">
                  {rewaya === 'hafs' && 'حفص'}
                  {rewaya === 'warsh' && 'ورش'}
                  {rewaya === 'qalun' && 'قالون'}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Madd Settings */}
        <div className="glass-card p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
              <Sliders className="text-emerald-400" size={20} />
            </div>
            <div>
              <h3 className="font-display text-lg text-white">أطوال المدود</h3>
              <p className="text-white/40 text-sm">تحديد أطوال المدود بالحركات</p>
            </div>
          </div>

          <div className="space-y-6">
            <SliderSetting
              label="المد المنفصل"
              labelEn="Madd Monfasel"
              value={localSettings.madd_monfasel_len}
              onChange={(v) => handleChange('madd_monfasel_len', v)}
              min={2}
              max={6}
            />

            <SliderSetting
              label="المد المتصل"
              labelEn="Madd Mottasel"
              value={localSettings.madd_mottasel_len}
              onChange={(v) => handleChange('madd_mottasel_len', v)}
              min={4}
              max={6}
            />

            <SliderSetting
              label="المد المتصل عند الوقف"
              labelEn="Madd Mottasel Waqf"
              value={localSettings.madd_mottasel_waqf}
              onChange={(v) => handleChange('madd_mottasel_waqf', v)}
              min={4}
              max={6}
            />

            <SliderSetting
              label="المد العارض"
              labelEn="Madd Aared"
              value={localSettings.madd_aared_len}
              onChange={(v) => handleChange('madd_aared_len', v)}
              min={2}
              max={6}
            />
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-center gap-4">
          <button onClick={handleSave} className="btn-primary">
            {saved ? (
              <>
                <Check size={20} />
                <span>تم الحفظ!</span>
              </>
            ) : (
              <>
                <Save size={20} />
                <span>حفظ الإعدادات</span>
              </>
            )}
          </button>

          <button onClick={handleReset} className="btn-secondary">
            <RotateCcw size={20} />
            <span>إعادة التعيين</span>
          </button>
        </div>
      </div>
    </motion.div>
  )
}

function SliderSetting({ label, labelEn, value, onChange, min, max }) {
  const arabicNums = '٠١٢٣٤٥٦٧٨٩'
  const toArabic = (n) => n.toString().split('').map(d => arabicNums[parseInt(d)]).join('')

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <div>
          <span className="text-white font-medium">{label}</span>
          <span className="text-white/30 text-sm mr-2">({labelEn})</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-gold-400 font-arabic text-xl">{toArabic(value)}</span>
          <span className="text-white/40 text-sm">حركات</span>
        </div>
      </div>

      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          value={value}
          onChange={(e) => onChange(parseInt(e.target.value))}
          className="w-full h-2 bg-white/10 rounded-full appearance-none cursor-pointer
            [&::-webkit-slider-thumb]:appearance-none
            [&::-webkit-slider-thumb]:w-5
            [&::-webkit-slider-thumb]:h-5
            [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:bg-gradient-to-br
            [&::-webkit-slider-thumb]:from-gold-400
            [&::-webkit-slider-thumb]:to-gold-600
            [&::-webkit-slider-thumb]:shadow-lg
            [&::-webkit-slider-thumb]:shadow-gold-500/30
            [&::-webkit-slider-thumb]:cursor-pointer
            [&::-webkit-slider-thumb]:transition-transform
            [&::-webkit-slider-thumb]:hover:scale-110"
        />

        {/* Track fill */}
        <div
          className="absolute top-0 left-0 h-2 bg-gradient-to-r from-gold-500/50 to-gold-400 rounded-full pointer-events-none"
          style={{ width: `${((value - min) / (max - min)) * 100}%` }}
        />

        {/* Step markers */}
        <div className="flex justify-between mt-2 px-1">
          {Array.from({ length: max - min + 1 }, (_, i) => min + i).map((n) => (
            <span
              key={n}
              className={`text-xs ${
                n === value ? 'text-gold-400' : 'text-white/30'
              }`}
            >
              {toArabic(n)}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}
