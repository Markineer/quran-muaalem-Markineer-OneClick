import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'

/**
 * InkView Component
 *
 * The "ink" metaphor:
 * - Inkless: Faint, ghostly text - visible but not in focus
 * - Inked: Full, vibrant text - the active ayah being recited
 *
 * This creates a beautiful visual hierarchy where only the
 * ayah being recited is fully visible, guiding the reader's focus.
 */
export default function InkView({
  ayat = [],
  activeAyahIdx = null,
  wrongSlots = new Set(),
  uncertainSlots = new Set(),
  hints = {},
  status = 'waiting',
}) {
  return (
    <div className="ink-container relative">
      {/* Subtle geometric pattern overlay */}
      <div className="absolute inset-0 bg-geometric-pattern opacity-30 pointer-events-none rounded-xl" />

      {/* Ambient glow behind active ayah */}
      <AnimatePresence>
        {activeAyahIdx !== null && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 pointer-events-none"
            style={{
              background: `radial-gradient(ellipse at center ${(activeAyahIdx / ayat.length) * 100}%, rgba(212, 164, 24, 0.05) 0%, transparent 50%)`,
            }}
          />
        )}
      </AnimatePresence>

      {/* Ayat */}
      <div className="relative z-10 space-y-2">
        {ayat.map((ayahText, idx) => (
          <AyahRow
            key={idx}
            text={ayahText}
            ayahNum={idx + 1}
            isActive={idx === activeAyahIdx}
            wrongSlots={idx === activeAyahIdx ? wrongSlots : new Set()}
            uncertainSlots={idx === activeAyahIdx ? uncertainSlots : new Set()}
            hints={idx === activeAyahIdx ? hints : {}}
          />
        ))}
      </div>

      {/* Empty state message */}
      {status === 'waiting' && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="absolute inset-0 flex items-center justify-center pointer-events-none"
        >
          <div className="text-center px-8 py-6 rounded-2xl bg-night-800/80 backdrop-blur-sm">
            <p className="text-white/40 font-body">
              اضغط على زر التسجيل وابدأ القراءة
            </p>
            <p className="text-white/20 text-sm font-body mt-2">
              يمكنك البدء من أي آية
            </p>
          </div>
        </motion.div>
      )}
    </div>
  )
}

function AyahRow({ text, ayahNum, isActive, wrongSlots, uncertainSlots, hints }) {
  const arabicNums = '٠١٢٣٤٥٦٧٨٩'
  const toArabicNum = (n) => n.toString().split('').map(d => arabicNums[parseInt(d)]).join('')

  return (
    <motion.div
      layout
      initial={false}
      animate={{
        scale: isActive ? 1.02 : 1,
        opacity: isActive ? 1 : 0.7,
      }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className={`ayah-row ${isActive ? 'active' : 'inactive'}`}
    >
      {/* Verse badge */}
      <span className="verse-badge">
        ﴿{toArabicNum(ayahNum)}﴾
      </span>

      {/* Ayah text */}
      <span className={isActive ? 'ayah-inked' : 'ayah-inkless'}>
        {isActive ? (
          <HighlightedText
            text={text}
            wrongSlots={wrongSlots}
            uncertainSlots={uncertainSlots}
            hints={hints}
          />
        ) : (
          text
        )}
      </span>

      {/* Active indicator line */}
      <AnimatePresence>
        {isActive && (
          <motion.div
            initial={{ scaleX: 0, opacity: 0 }}
            animate={{ scaleX: 1, opacity: 1 }}
            exit={{ scaleX: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="absolute bottom-0 right-0 left-0 h-0.5 bg-gradient-to-l from-gold-500/50 via-gold-400/30 to-transparent origin-right"
          />
        )}
      </AnimatePresence>
    </motion.div>
  )
}

function HighlightedText({ text, wrongSlots, uncertainSlots, hints }) {
  // Parse the text into letter slots
  // This is a simplified version - the actual implementation uses the harakat_mode module
  const slots = parseTextToSlots(text)

  // Map letter-slot indices to actual slot indices
  const letterSlotIndices = slots
    .map((slot, idx) => ({ ...slot, idx }))
    .filter(slot => slot.isLetter)
    .map((slot, letterIdx) => ({ ...slot, letterIdx }))

  return (
    <>
      {slots.map((slot, idx) => {
        // Find if this slot corresponds to a wrong/uncertain letter
        const letterInfo = letterSlotIndices.find(ls => ls.idx === idx)
        const letterIdx = letterInfo?.letterIdx

        const isWrong = letterIdx !== undefined && wrongSlots.has(letterIdx)
        const isUncertain = letterIdx !== undefined && uncertainSlots.has(letterIdx)
        const hint = letterIdx !== undefined ? hints[letterIdx] : null

        if (isWrong) {
          return (
            <span
              key={idx}
              className="harakat-wrong"
              data-hint={hint || 'خطأ في الحركة'}
              title={hint || 'خطأ في الحركة'}
            >
              {slot.text}
            </span>
          )
        }

        if (isUncertain) {
          return (
            <span
              key={idx}
              className="harakat-uncertain"
              data-hint={hint || 'غير متأكد'}
              title={hint || 'غير متأكد'}
            >
              {slot.text}
            </span>
          )
        }

        return <span key={idx}>{slot.text}</span>
      })}
    </>
  )
}

/**
 * Parse Arabic text into slots (letters with their diacritics)
 * This is a simplified client-side parser
 */
function parseTextToSlots(text) {
  const slots = []
  const diacritics = new Set([
    '\u064B', '\u064C', '\u064D', '\u064E', '\u064F', '\u0650',
    '\u0651', '\u0652', '\u0653', '\u0654', '\u0655', '\u0656',
    '\u0657', '\u0658', '\u0670', '\u06D6', '\u06D7', '\u06D8',
    '\u06D9', '\u06DA', '\u06DB', '\u06DC', '\u06DF', '\u06E0',
    '\u06E1', '\u06E2', '\u06E3', '\u06E4', '\u06E7', '\u06E8',
    '\u06EA', '\u06EB', '\u06EC', '\u06ED',
  ])

  let i = 0
  while (i < text.length) {
    const char = text[i]

    // Check if it's a space or separator
    if (char === ' ' || char === '۝' || char === '۞') {
      slots.push({ text: char, isLetter: false })
      i++
      continue
    }

    // Check if it's an Arabic letter (not a diacritic)
    if (!diacritics.has(char)) {
      let slotText = char
      i++

      // Collect following diacritics
      while (i < text.length && diacritics.has(text[i])) {
        slotText += text[i]
        i++
      }

      slots.push({ text: slotText, isLetter: true })
    } else {
      // Standalone diacritic (shouldn't happen in proper text)
      slots.push({ text: char, isLetter: false })
      i++
    }
  }

  return slots
}
