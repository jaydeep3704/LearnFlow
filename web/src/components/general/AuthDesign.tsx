"use client"

import { motion } from "framer-motion"

export function AuthDesign() {
  return (
    <div className="hidden lg:flex min-h-screen bg-gradient-to-br from-background via-background to-violet-950/20 relative overflow-hidden items-center justify-center">
      {/* Animated background orbs */}
      <motion.div
        className="absolute top-20 left-20 w-72 h-72 bg-violet-600/20 rounded-full blur-3xl"
        animate={{
          y: [0, 50, 0],
          x: [0, 30, 0],
        }}
        transition={{
          duration: 8,
          repeat: Number.POSITIVE_INFINITY,
          ease: "easeInOut",
        }}
      />

      <motion.div
        className="absolute bottom-20 right-20 w-96 h-96 bg-blue-600/20 rounded-full blur-3xl"
        animate={{
          y: [0, -50, 0],
          x: [0, -30, 0],
        }}
        transition={{
          duration: 10,
          repeat: Number.POSITIVE_INFINITY,
          ease: "easeInOut",
        }}
      />

      {/* Center geometric shapes */}
      <div className="relative z-10 flex items-center justify-center">
        {/* Rotating outer ring */}
        <motion.div
          className="absolute w-80 h-80 border-2 border-violet-500/30 rounded-full"
          animate={{ rotate: 360 }}
          transition={{
            duration: 20,
            repeat: Number.POSITIVE_INFINITY,
            ease: "linear",
          }}
        />

        {/* Rotating inner ring */}
        <motion.div
          className="absolute w-60 h-60 border-2 border-blue-500/30 rounded-full"
          animate={{ rotate: -360 }}
          transition={{
            duration: 15,
            repeat: Number.POSITIVE_INFINITY,
            ease: "linear",
          }}
        />

        {/* Center pulsing circle */}
        <motion.div
          className="absolute w-32 h-32 bg-gradient-to-br from-violet-600 to-blue-600 rounded-full"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.6, 0.8, 0.6],
          }}
          transition={{
            duration: 4,
            repeat: Number.POSITIVE_INFINITY,
            ease: "easeInOut",
          }}
        />

        {/* Floating geometric elements */}
        <motion.div
          className="absolute w-20 h-20 border-2 border-violet-400/50 rounded-lg"
          animate={{
            rotate: 360,
            y: [-100, 100, -100],
          }}
          transition={{
            duration: 12,
            repeat: Number.POSITIVE_INFINITY,
            ease: "easeInOut",
          }}
        />

        <motion.div
          className="absolute w-16 h-16 border-2 border-blue-400/50 rounded-full"
          animate={{
            rotate: -360,
            x: [100, -100, 100],
          }}
          transition={{
            duration: 10,
            repeat: Number.POSITIVE_INFINITY,
            ease: "easeInOut",
          }}
        />

        {/* Animated particles */}
        {[...Array(6)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-2 h-2 bg-violet-400 rounded-full"
            animate={{
              x: Math.cos((i / 6) * Math.PI * 2) * 150,
              y: Math.sin((i / 6) * Math.PI * 2) * 150,
              opacity: [0.3, 0.8, 0.3],
            }}
            transition={{
              duration: 6 + i,
              repeat: Number.POSITIVE_INFINITY,
              ease: "easeInOut",
            }}
          />
        ))}
      </div>

      {/* Grid pattern overlay */}
      <div className="absolute inset-0 bg-grid-pattern opacity-5" />
    </div>
  )
}
