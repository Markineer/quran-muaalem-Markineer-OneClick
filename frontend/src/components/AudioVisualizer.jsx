import React, { useMemo } from 'react'
import { motion } from 'framer-motion'

/**
 * AudioVisualizer Component
 *
 * A beautiful audio level visualizer that shows audio input activity.
 * Uses staggered bars with golden gradient to match the overall theme.
 */
export default function AudioVisualizer({ level = 0, isActive = false }) {
  const barCount = 9 // Odd number for symmetry

  // Generate bar heights based on audio level
  // Higher bars in the center, creating a wave-like effect
  const bars = useMemo(() => {
    const center = Math.floor(barCount / 2)
    return Array.from({ length: barCount }, (_, i) => {
      const distanceFromCenter = Math.abs(i - center)
      const baseHeight = 1 - (distanceFromCenter / center) * 0.5
      const variation = Math.sin(Date.now() / 200 + i * 0.5) * 0.3

      if (!isActive) {
        return 0.15 // Minimal height when inactive
      }

      // Scale by audio level
      return Math.max(0.1, Math.min(1, baseHeight * level + variation * level))
    })
  }, [level, isActive, barCount])

  return (
    <div className="flex items-center justify-center gap-1 h-12">
      {bars.map((height, i) => (
        <motion.div
          key={i}
          className="audio-bar"
          initial={{ height: '8px' }}
          animate={{
            height: `${height * 40}px`,
            opacity: isActive ? 0.6 + height * 0.4 : 0.3,
          }}
          transition={{
            duration: 0.1,
            ease: 'easeOut',
          }}
          style={{
            background: isActive
              ? `linear-gradient(180deg,
                  rgba(245, 206, 79, ${0.8 + height * 0.2}) 0%,
                  rgba(212, 164, 24, ${0.6 + height * 0.4}) 50%,
                  rgba(166, 124, 18, ${0.4 + height * 0.6}) 100%)`
              : 'rgba(255, 255, 255, 0.1)',
          }}
        />
      ))}
    </div>
  )
}

/**
 * WaveformVisualizer Component
 *
 * Alternative visualizer using a continuous waveform
 */
export function WaveformVisualizer({ level = 0, isActive = false }) {
  const points = 50
  const height = 40

  // Generate wave path
  const wavePath = useMemo(() => {
    const segmentWidth = 100 / (points - 1)
    let path = `M 0 ${height / 2}`

    for (let i = 0; i < points; i++) {
      const x = i * segmentWidth
      const baseWave = Math.sin((i / points) * Math.PI * 4) * 0.3
      const noise = Math.sin(Date.now() / 100 + i * 0.5) * 0.2
      const amplitude = isActive ? level * 0.8 : 0.1
      const y = (height / 2) + (baseWave + noise) * amplitude * height

      path += ` L ${x} ${y}`
    }

    return path
  }, [level, isActive, points, height])

  return (
    <div className="flex items-center justify-center h-12 px-4">
      <svg
        viewBox={`0 0 100 ${height}`}
        className="w-full h-full max-w-md"
        preserveAspectRatio="none"
      >
        <defs>
          <linearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgba(212, 164, 24, 0.2)" />
            <stop offset="50%" stopColor="rgba(212, 164, 24, 0.8)" />
            <stop offset="100%" stopColor="rgba(212, 164, 24, 0.2)" />
          </linearGradient>
        </defs>

        {/* Center line */}
        <line
          x1="0"
          y1={height / 2}
          x2="100"
          y2={height / 2}
          stroke="rgba(255, 255, 255, 0.1)"
          strokeWidth="1"
        />

        {/* Wave path */}
        <motion.path
          d={wavePath}
          fill="none"
          stroke={isActive ? 'url(#waveGradient)' : 'rgba(255, 255, 255, 0.2)'}
          strokeWidth="2"
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.5 }}
        />
      </svg>
    </div>
  )
}

/**
 * CircularVisualizer Component
 *
 * Circular audio visualizer for a more unique look
 */
export function CircularVisualizer({ level = 0, isActive = false }) {
  const bars = 24
  const radius = 30
  const maxBarHeight = 15

  return (
    <div className="flex items-center justify-center h-24">
      <div className="relative" style={{ width: radius * 2 + maxBarHeight * 2, height: radius * 2 + maxBarHeight * 2 }}>
        {Array.from({ length: bars }, (_, i) => {
          const angle = (i / bars) * Math.PI * 2 - Math.PI / 2
          const variation = Math.sin(Date.now() / 150 + i * 0.3) * 0.3
          const barHeight = isActive
            ? maxBarHeight * (0.3 + level * 0.7 + variation * level)
            : maxBarHeight * 0.2

          const x1 = Math.cos(angle) * radius
          const y1 = Math.sin(angle) * radius
          const x2 = Math.cos(angle) * (radius + barHeight)
          const y2 = Math.sin(angle) * (radius + barHeight)

          return (
            <motion.div
              key={i}
              className="absolute bg-gradient-to-t from-green-600 to-green-400"
              style={{
                width: 3,
                left: '50%',
                top: '50%',
                transformOrigin: 'center bottom',
                borderRadius: 2,
              }}
              initial={{ height: 5 }}
              animate={{
                height: barHeight,
                opacity: isActive ? 0.5 + level * 0.5 : 0.3,
                transform: `translate(-50%, -100%) rotate(${(i / bars) * 360}deg) translateY(-${radius}px)`,
              }}
              transition={{ duration: 0.1 }}
            />
          )
        })}

        {/* Center dot */}
        <motion.div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-green-500"
          animate={{
            width: isActive ? 8 + level * 4 : 6,
            height: isActive ? 8 + level * 4 : 6,
            opacity: isActive ? 0.8 : 0.4,
          }}
          transition={{ duration: 0.1 }}
        />
      </div>
    </div>
  )
}
