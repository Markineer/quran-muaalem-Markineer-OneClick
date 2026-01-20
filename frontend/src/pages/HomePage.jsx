import React from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Mic, Upload, Sparkles, ArrowLeft, Star, Zap, Shield } from 'lucide-react'

const FATIHA_PREVIEW = 'بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ'

export default function HomePage() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="max-w-6xl mx-auto"
    >
      {/* Hero Section */}
      <section className="py-16 md:py-24 text-center relative">
        {/* Decorative elements */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-px h-24 bg-gradient-to-b from-transparent via-gold-500/30 to-transparent" />

        <motion.div
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
        >
          <span className="inline-block px-4 py-2 rounded-full bg-gold-500/10 border border-gold-500/20 text-gold-400 text-sm font-medium mb-8">
            معلم التلاوة الذكي
          </span>
        </motion.div>

        <motion.h1
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="font-display text-5xl md:text-7xl font-bold mb-6"
        >
          <span className="text-white">تعلّم القراءة</span>
          <br />
          <span className="text-gradient-gold">الصحيحة</span>
        </motion.h1>

        <motion.p
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="text-xl text-white/60 max-w-2xl mx-auto mb-12 font-body leading-relaxed"
        >
          نظام ذكي يستمع لتلاوتك ويكشف أخطاء الحركات والتجويد في الوقت الفعلي،
          ليساعدك على إتقان تلاوة القرآن الكريم
        </motion.p>

        {/* Preview Quran text */}
        <motion.div
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="relative inline-block mb-12"
        >
          <div className="absolute -inset-4 bg-gradient-to-r from-gold-500/10 via-gold-500/5 to-gold-500/10 rounded-2xl blur-xl" />
          <div className="relative glass-card px-12 py-8">
            <p className="font-arabic text-4xl md:text-5xl text-white/90 leading-relaxed">
              {FATIHA_PREVIEW}
            </p>
            <div className="corner-ornament top-right" />
            <div className="corner-ornament top-left" />
            <div className="corner-ornament bottom-right" />
            <div className="corner-ornament bottom-left" />
          </div>
        </motion.div>

        {/* CTA Buttons */}
        <motion.div
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="flex flex-col sm:flex-row items-center justify-center gap-4"
        >
          <Link to="/realtime" className="btn-primary group">
            <Mic size={20} />
            <span>ابدأ التدريب المباشر</span>
            <ArrowLeft size={16} className="group-hover:-translate-x-1 transition-transform" />
          </Link>
          <Link to="/harakat" className="btn-secondary">
            <Upload size={20} />
            <span>رفع تسجيل صوتي</span>
          </Link>
        </motion.div>
      </section>

      {/* Features Section */}
      <section className="py-16">
        <motion.div
          initial={{ y: 50, opacity: 0 }}
          whileInView={{ y: 0, opacity: 1 }}
          viewport={{ once: true }}
          className="grid md:grid-cols-3 gap-6"
        >
          <FeatureCard
            icon={Zap}
            title="تحليل فوري"
            titleEn="Real-Time Analysis"
            description="يكشف أخطاء الحركات والتجويد لحظة بلحظة أثناء تلاوتك"
            gradient="from-amber-500/20 to-orange-500/20"
            iconColor="text-amber-400"
            delay={0}
          />
          <FeatureCard
            icon={Star}
            title="ابدأ من أي آية"
            titleEn="Start Anywhere"
            description="يتعرف النظام تلقائياً على الآية التي تقرأها من أي موضع"
            gradient="from-emerald-500/20 to-teal-500/20"
            iconColor="text-emerald-400"
            delay={0.1}
          />
          <FeatureCard
            icon={Shield}
            title="دقة عالية"
            titleEn="High Accuracy"
            description="مدرّب على آلاف التلاوات لضمان دقة الكشف عن الأخطاء"
            gradient="from-blue-500/20 to-indigo-500/20"
            iconColor="text-blue-400"
            delay={0.2}
          />
        </motion.div>
      </section>

      {/* How it works */}
      <section className="py-16">
        <motion.div
          initial={{ y: 50, opacity: 0 }}
          whileInView={{ y: 0, opacity: 1 }}
          viewport={{ once: true }}
        >
          <h2 className="font-display text-3xl text-center text-white mb-4">
            كيف يعمل؟
          </h2>
          <p className="text-center text-white/50 mb-12 max-w-xl mx-auto">
            ثلاث خطوات بسيطة للبدء في تحسين تلاوتك
          </p>

          <div className="grid md:grid-cols-3 gap-8 relative">
            {/* Connecting line */}
            <div className="hidden md:block absolute top-16 left-1/4 right-1/4 h-px bg-gradient-to-r from-transparent via-gold-500/30 to-transparent" />

            <StepCard
              number="١"
              title="اختر الآية"
              description="حدد السورة والآية التي تريد التدرب عليها أو اقرأ من أي موضع"
              delay={0}
            />
            <StepCard
              number="٢"
              title="ابدأ التلاوة"
              description="اقرأ بصوت واضح وسيبدأ النظام في الاستماع والتحليل"
              delay={0.1}
            />
            <StepCard
              number="٣"
              title="راجع النتائج"
              description="شاهد الأخطاء مظللة باللون الأحمر مع شرح الحركة الصحيحة"
              delay={0.2}
            />
          </div>
        </motion.div>
      </section>
    </motion.div>
  )
}

function FeatureCard({ icon: Icon, title, titleEn, description, gradient, iconColor, delay }) {
  return (
    <motion.div
      initial={{ y: 30, opacity: 0 }}
      whileInView={{ y: 0, opacity: 1 }}
      viewport={{ once: true }}
      transition={{ delay }}
      className="glass-card p-8 group hover:border-white/10 transition-all duration-300"
    >
      <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${gradient} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
        <Icon className={iconColor} size={24} />
      </div>
      <h3 className="font-display text-xl text-white mb-2">{title}</h3>
      <p className="text-xs text-gold-500/60 mb-3 font-body">{titleEn}</p>
      <p className="text-white/50 font-body leading-relaxed">{description}</p>
    </motion.div>
  )
}

function StepCard({ number, title, description, delay }) {
  return (
    <motion.div
      initial={{ y: 30, opacity: 0 }}
      whileInView={{ y: 0, opacity: 1 }}
      viewport={{ once: true }}
      transition={{ delay }}
      className="text-center relative"
    >
      <div className="w-16 h-16 rounded-full bg-gradient-to-br from-gold-500/20 to-gold-600/20 border border-gold-500/30 flex items-center justify-center mx-auto mb-6 relative z-10">
        <span className="font-arabic text-3xl text-gold-400">{number}</span>
      </div>
      <h3 className="font-display text-lg text-white mb-3">{title}</h3>
      <p className="text-white/50 text-sm font-body max-w-xs mx-auto">{description}</p>
    </motion.div>
  )
}
