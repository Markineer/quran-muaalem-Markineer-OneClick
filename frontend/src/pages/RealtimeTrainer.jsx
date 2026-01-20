import React, { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Mic, MicOff, RotateCcw, Volume2, Loader2 } from 'lucide-react'
import InkView from '../components/InkView'
import AudioVisualizer from '../components/AudioVisualizer'

const FATIHA_AYAT = [
  'بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ',
  'ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَـٰلَمِينَ',
  'ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ',
  'مَـٰلِكِ يَوْمِ ٱلدِّينِ',
  'إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ',
  'ٱهْدِنَا ٱلصِّرَٰطَ ٱلْمُسْتَقِيمَ',
  'صِرَٰطَ ٱلَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ ٱلْمَغْضُوبِ عَلَيْهِمْ وَلَا ٱلضَّآلِّينَ',
]

export default function RealtimeTrainer({ settings }) {
  const [isRecording, setIsRecording] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [status, setStatus] = useState('waiting') // waiting, detecting, detected, error
  const [activeAyahIdx, setActiveAyahIdx] = useState(null)
  const [confidence, setConfidence] = useState(0)
  const [wrongSlots, setWrongSlots] = useState(new Set())
  const [uncertainSlots, setUncertainSlots] = useState(new Set())
  const [hints, setHints] = useState({})
  const [audioLevel, setAudioLevel] = useState(0)
  const [error, setError] = useState(null)

  const wsRef = useRef(null)
  const mediaStreamRef = useRef(null)
  const audioContextRef = useRef(null)
  const analyserRef = useRef(null)
  const processorRef = useRef(null)

  // Start recording
  const startRecording = useCallback(async () => {
    try {
      setIsConnecting(true)
      setError(null)

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        }
      })

      mediaStreamRef.current = stream

      // Create audio context for visualization
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      })
      audioContextRef.current = audioContext

      const source = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser)
      analyserRef.current = analyser

      // Connect to WebSocket
      const ws = new WebSocket(`ws://${window.location.host}/ws/realtime`)
      wsRef.current = ws

      ws.onopen = () => {
        setIsConnecting(false)
        setIsRecording(true)
        setStatus('detecting')
        startAudioStreaming(stream, ws)
        startAudioLevelMonitor()
      }

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        handleServerMessage(data)
      }

      ws.onerror = () => {
        setError('خطأ في الاتصال بالخادم')
        stopRecording()
      }

      ws.onclose = () => {
        if (isRecording) {
          setStatus('waiting')
        }
      }

    } catch (err) {
      setIsConnecting(false)
      if (err.name === 'NotAllowedError') {
        setError('يرجى السماح بالوصول إلى الميكروفون')
      } else {
        setError('فشل في تشغيل الميكروفون')
      }

      // Demo mode - simulate detection after 3 seconds
      simulateDemoMode()
    }
  }, [isRecording])

  // Demo mode simulation
  const simulateDemoMode = () => {
    setIsConnecting(false)
    setIsRecording(true)
    setStatus('detecting')

    // Simulate ayah detection after 3 seconds
    setTimeout(() => {
      setStatus('detected')
      setActiveAyahIdx(0)
      setConfidence(0.87)
    }, 3000)

    // Simulate some errors after 5 seconds
    setTimeout(() => {
      setWrongSlots(new Set([3, 8]))
      setUncertainSlots(new Set([12]))
      setHints({
        3: 'المتوقع: كسرة | المقروء: فتحة',
        8: 'المتوقع: ضمة | المقروء: سكون',
        12: 'غير متأكد',
      })
    }, 5000)

    // Simulate moving to ayah 2
    setTimeout(() => {
      setActiveAyahIdx(1)
      setConfidence(0.92)
      setWrongSlots(new Set([2]))
      setUncertainSlots(new Set())
      setHints({
        2: 'المتوقع: فتحة | المقروء: كسرة',
      })
    }, 10000)
  }

  // Start audio streaming to server
  const startAudioStreaming = (stream, ws) => {
    const audioContext = audioContextRef.current
    const source = audioContext.createMediaStreamSource(stream)

    // Create script processor for sending audio chunks
    const processor = audioContext.createScriptProcessor(4096, 1, 1)
    processorRef.current = processor

    processor.onaudioprocess = (event) => {
      if (ws.readyState === WebSocket.OPEN) {
        const inputData = event.inputBuffer.getChannelData(0)
        const pcmData = new Int16Array(inputData.length)

        // Convert float32 to int16
        for (let i = 0; i < inputData.length; i++) {
          pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768))
        }

        ws.send(pcmData.buffer)
      }
    }

    source.connect(processor)
    processor.connect(audioContext.destination)
  }

  // Monitor audio level for visualization
  const startAudioLevelMonitor = () => {
    const analyser = analyserRef.current
    if (!analyser) return

    const dataArray = new Uint8Array(analyser.frequencyBinCount)

    const updateLevel = () => {
      if (!isRecording) return

      analyser.getByteFrequencyData(dataArray)
      const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length
      setAudioLevel(average / 255)

      requestAnimationFrame(updateLevel)
    }

    updateLevel()
  }

  // Handle server messages
  const handleServerMessage = (data) => {
    switch (data.type) {
      case 'status':
        setStatus(data.status)
        break

      case 'detection':
        setActiveAyahIdx(data.ayah_idx)
        setConfidence(data.confidence)
        setStatus('detected')
        break

      case 'tracking':
        setWrongSlots(new Set(data.wrong_slots || []))
        setUncertainSlots(new Set(data.uncertain_slots || []))
        setHints(data.hints || {})
        break

      case 'error':
        setError(data.message)
        break
    }
  }

  // Stop recording
  const stopRecording = useCallback(() => {
    setIsRecording(false)
    setStatus('waiting')
    setAudioLevel(0)

    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    // Stop media stream
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop())
      mediaStreamRef.current = null
    }

    // Close audio context
    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }

    // Disconnect processor
    if (processorRef.current) {
      processorRef.current.disconnect()
      processorRef.current = null
    }
  }, [])

  // Reset session
  const resetSession = () => {
    stopRecording()
    setActiveAyahIdx(null)
    setConfidence(0)
    setWrongSlots(new Set())
    setUncertainSlots(new Set())
    setHints({})
    setError(null)
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording()
    }
  }, [stopRecording])

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="max-w-5xl mx-auto"
    >
      {/* Header */}
      <div className="text-center mb-8">
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-crimson-500/10 border border-crimson-500/20 text-crimson-400 text-sm mb-6"
        >
          <span className="w-2 h-2 rounded-full bg-crimson-500 animate-pulse" />
          <span>تدريب مباشر</span>
        </motion.div>

        <h1 className="font-display text-4xl text-white mb-4">
          التدريب <span className="text-gradient-gold">المباشر</span>
        </h1>
        <p className="text-white/50 max-w-xl mx-auto">
          ابدأ القراءة من أي آية — سيتعرف النظام تلقائياً على الآية ويتتبع أخطاء الحركات
        </p>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-4 mb-8">
        {/* Record Button */}
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={isRecording ? stopRecording : startRecording}
          disabled={isConnecting}
          className={`btn-record ${isRecording ? 'recording' : ''}`}
        >
          {isConnecting ? (
            <Loader2 size={32} className="animate-spin" />
          ) : isRecording ? (
            <MicOff size={32} />
          ) : (
            <Mic size={32} />
          )}
        </motion.button>

        {/* Reset Button */}
        <button
          onClick={resetSession}
          className="btn-secondary p-4 rounded-full"
          title="إعادة تعيين"
        >
          <RotateCcw size={20} />
        </button>
      </div>

      {/* Status */}
      <div className="flex justify-center mb-8">
        <StatusPill status={status} ayahIdx={activeAyahIdx} confidence={confidence} />
      </div>

      {/* Audio Level */}
      <AnimatePresence>
        {isRecording && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-8"
          >
            <AudioVisualizer level={audioLevel} isActive={isRecording} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error Message */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="glass-card p-4 mb-8 border-crimson-500/30 text-center"
          >
            <p className="text-crimson-400">{error}</p>
            <p className="text-white/40 text-sm mt-2">
              سيتم استخدام الوضع التجريبي
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Ink View */}
      <motion.div
        layout
        className="glass-card p-8 border-glow"
      >
        <InkView
          ayat={FATIHA_AYAT}
          activeAyahIdx={activeAyahIdx}
          wrongSlots={wrongSlots}
          uncertainSlots={uncertainSlots}
          hints={hints}
          status={status}
        />

        {/* Legend */}
        <div className="legend mt-8">
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
          <div className="legend-item">
            <div className="legend-dot inkless" />
            <span>آية غير نشطة</span>
          </div>
        </div>
      </motion.div>

      {/* Instructions */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="mt-8 text-center"
      >
        <div className="inline-flex items-center gap-3 px-6 py-4 rounded-2xl bg-blue-500/10 border border-blue-500/20">
          <Volume2 className="text-blue-400" size={20} />
          <p className="text-blue-300 text-sm">
            اضغط على زر التسجيل ثم ابدأ القراءة بصوت واضح
          </p>
        </div>
      </motion.div>
    </motion.div>
  )
}

function StatusPill({ status, ayahIdx, confidence }) {
  const statusConfig = {
    waiting: {
      text: 'في انتظار الصوت...',
      icon: '🔇',
      className: 'status-waiting',
    },
    detecting: {
      text: 'جاري التعرف على الآية...',
      icon: '⏳',
      className: 'status-detecting',
    },
    detected: {
      text: `الآية ${toArabicNum(ayahIdx + 1)} (${Math.round(confidence * 100)}%)`,
      icon: '🎯',
      className: 'status-detected',
    },
    error: {
      text: 'خطأ في الاتصال',
      icon: '⚠️',
      className: 'status-error',
    },
  }

  const config = statusConfig[status] || statusConfig.waiting

  return (
    <motion.div
      layout
      className={`status-pill ${config.className}`}
    >
      <span>{config.icon}</span>
      <span>{config.text}</span>
    </motion.div>
  )
}

function toArabicNum(num) {
  const arabicNums = '٠١٢٣٤٥٦٧٨٩'
  return num.toString().split('').map(d => arabicNums[parseInt(d)]).join('')
}
