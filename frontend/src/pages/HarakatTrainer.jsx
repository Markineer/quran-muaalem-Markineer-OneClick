import React, { useState, useRef } from 'react'
import { motion } from 'framer-motion'
import { Upload, Play, Loader2, AlertCircle, CheckCircle2, FileAudio } from 'lucide-react'

const FATIHA_AYAT = [
  { num: '١', text: 'بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ' },
  { num: '٢', text: 'ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَـٰلَمِينَ' },
  { num: '٣', text: 'ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ' },
  { num: '٤', text: 'مَـٰلِكِ يَوْمِ ٱلدِّينِ' },
  { num: '٥', text: 'إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ' },
  { num: '٦', text: 'ٱهْدِنَا ٱلصِّرَٰطَ ٱلْمُسْتَقِيمَ' },
  { num: '٧', text: 'صِرَٰطَ ٱلَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ ٱلْمَغْضُوبِ عَلَيْهِمْ وَلَا ٱلضَّآلِّينَ' },
]

export default function HarakatTrainer({ settings }) {
  const [selectedAyah, setSelectedAyah] = useState(-1) // -1 = full Fatiha
  const [audioFile, setAudioFile] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      setAudioFile(file)
      setResults(null)
      setError(null)
    }
  }

  const handleAnalyze = async () => {
    if (!audioFile) return

    setIsAnalyzing(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('audio', audioFile)
      formData.append('ayah_idx', selectedAyah.toString())
      formData.append('settings', JSON.stringify(settings))

      const response = await fetch('/api/analyze-harakat', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('فشل في تحليل التلاوة')
      }

      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError(err.message)
      // For demo, show mock results
      setResults({
        html: generateMockResults(),
        stats: { correct: 85, wrong: 3, uncertain: 2 }
      })
    } finally {
      setIsAnalyzing(false)
    }
  }

  const displayText = selectedAyah === -1
    ? FATIHA_AYAT.map(a => a.text).join(' ۝ ')
    : FATIHA_AYAT[selectedAyah]?.text || ''

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="max-w-5xl mx-auto"
    >
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="font-display text-4xl text-white mb-4">
          تدريب <span className="text-gradient-gold">الحركات</span>
        </h1>
        <p className="text-white/50 max-w-xl mx-auto">
          ارفع تسجيلاً صوتياً لتلاوتك واحصل على تحليل مفصّل للحركات
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Left Column - Selection & Upload */}
        <div className="space-y-6">
          {/* Ayah Selection */}
          <div className="glass-card p-6">
            <h3 className="font-display text-lg text-white mb-4">اختر الآية</h3>

            <div className="space-y-2">
              <button
                onClick={() => setSelectedAyah(-1)}
                className={`w-full text-right px-4 py-3 rounded-xl transition-all ${
                  selectedAyah === -1
                    ? 'bg-gold-500/20 border border-gold-500/30 text-gold-400'
                    : 'bg-white/5 border border-transparent hover:bg-white/10 text-white/70'
                }`}
              >
                <span className="font-arabic">كامل سورة الفاتحة</span>
              </button>

              {FATIHA_AYAT.map((ayah, idx) => (
                <button
                  key={idx}
                  onClick={() => setSelectedAyah(idx)}
                  className={`w-full text-right px-4 py-3 rounded-xl transition-all ${
                    selectedAyah === idx
                      ? 'bg-gold-500/20 border border-gold-500/30 text-gold-400'
                      : 'bg-white/5 border border-transparent hover:bg-white/10 text-white/70'
                  }`}
                >
                  <span className="font-arabic text-sm">{ayah.num}. {ayah.text.slice(0, 40)}...</span>
                </button>
              ))}
            </div>
          </div>

          {/* Audio Upload */}
          <div className="glass-card p-6">
            <h3 className="font-display text-lg text-white mb-4">رفع التسجيل</h3>

            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              onChange={handleFileChange}
              className="hidden"
            />

            <button
              onClick={() => fileInputRef.current?.click()}
              className="w-full border-2 border-dashed border-white/10 hover:border-gold-500/30 rounded-2xl p-8 transition-all group"
            >
              <div className="flex flex-col items-center gap-4">
                {audioFile ? (
                  <>
                    <div className="w-16 h-16 rounded-full bg-emerald-500/20 flex items-center justify-center">
                      <FileAudio className="text-emerald-400" size={28} />
                    </div>
                    <div className="text-center">
                      <p className="text-white font-medium">{audioFile.name}</p>
                      <p className="text-white/40 text-sm">
                        {(audioFile.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="w-16 h-16 rounded-full bg-white/5 group-hover:bg-gold-500/10 flex items-center justify-center transition-colors">
                      <Upload className="text-white/40 group-hover:text-gold-400" size={28} />
                    </div>
                    <div className="text-center">
                      <p className="text-white/70 group-hover:text-white">اضغط لرفع ملف صوتي</p>
                      <p className="text-white/40 text-sm">WAV, MP3, M4A, OGG</p>
                    </div>
                  </>
                )}
              </div>
            </button>

            {/* Analyze Button */}
            <button
              onClick={handleAnalyze}
              disabled={!audioFile || isAnalyzing}
              className="btn-primary w-full mt-6 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 size={20} className="animate-spin" />
                  <span>جاري التحليل...</span>
                </>
              ) : (
                <>
                  <Play size={20} />
                  <span>ابدأ تحليل الحركات</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column - Text Display & Results */}
        <div className="space-y-6">
          {/* Quran Text */}
          <div className="glass-card p-6">
            <h3 className="font-display text-lg text-white mb-4">النص القرآني</h3>
            <div className="ink-container">
              <p className="font-arabic text-2xl leading-loose text-white/90">
                {displayText}
              </p>
            </div>
          </div>

          {/* Results */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-card p-6 border-crimson-500/30"
            >
              <div className="flex items-center gap-3 text-crimson-400">
                <AlertCircle size={20} />
                <span>{error}</span>
              </div>
            </motion.div>
          )}

          {results && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-card p-6"
            >
              <h3 className="font-display text-lg text-white mb-4">نتيجة التحليل</h3>

              {/* Stats */}
              {results.stats && (
                <div className="grid grid-cols-3 gap-4 mb-6">
                  <StatBox
                    label="صحيح"
                    value={results.stats.correct}
                    color="emerald"
                    icon={CheckCircle2}
                  />
                  <StatBox
                    label="خطأ"
                    value={results.stats.wrong}
                    color="crimson"
                    icon={AlertCircle}
                  />
                  <StatBox
                    label="غير متأكد"
                    value={results.stats.uncertain}
                    color="amber"
                    icon={AlertCircle}
                  />
                </div>
              )}

              {/* HTML Results */}
              <div
                className="ink-container"
                dangerouslySetInnerHTML={{ __html: results.html }}
              />

              {/* Legend */}
              <div className="legend mt-6">
                <div className="legend-item">
                  <div className="legend-dot correct" />
                  <span>صحيح</span>
                </div>
                <div className="legend-item">
                  <div className="legend-dot wrong" />
                  <span>خطأ في الحركة</span>
                </div>
                <div className="legend-item">
                  <div className="legend-dot uncertain" />
                  <span>غير متأكد</span>
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </motion.div>
  )
}

function StatBox({ label, value, color, icon: Icon }) {
  const colors = {
    emerald: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400',
    crimson: 'bg-crimson-500/10 border-crimson-500/20 text-crimson-400',
    amber: 'bg-amber-500/10 border-amber-500/20 text-amber-400',
  }

  return (
    <div className={`p-4 rounded-xl border ${colors[color]}`}>
      <div className="flex items-center gap-2 mb-2">
        <Icon size={16} />
        <span className="text-sm opacity-70">{label}</span>
      </div>
      <p className="text-2xl font-display">{value}</p>
    </div>
  )
}

function generateMockResults() {
  return `
    <p class="font-arabic text-2xl leading-loose">
      <span>بِسْمِ</span>
      <span class="harakat-wrong" data-hint="المتوقع: كسرة | المقروء: فتحة">ٱ</span>
      <span>للَّهِ</span>
      <span> </span>
      <span>ٱلرَّحْمَـٰنِ</span>
      <span> </span>
      <span class="harakat-uncertain" data-hint="غير متأكد">ٱلرَّحِيمِ</span>
    </p>
  `
}
