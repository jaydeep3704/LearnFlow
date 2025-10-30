"use client"

import { motion } from "framer-motion"
import { Play, BookOpen, CheckCircle2 } from "lucide-react"

export function HeroImg() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.3,
      },
    },
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.8,
        ease: "easeOut",
      },
    },
  }

  const floatVariants = {
    animate: {
      y: [0, -20, 0],
      transition: {
        duration: 4,
        repeat: Number.POSITIVE_INFINITY,
        ease: "easeInOut",
      },
    },
  }

  const rotateVariants = {
    animate: {
      rotate: [0, 5, -5, 0],
      transition: {
        duration: 6,
        repeat: Number.POSITIVE_INFINITY,
        ease: "easeInOut",
      },
    },
  }

  return (
    <motion.div
      className="relative h-full min-h-72 flex items-center justify-end"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* Background gradient orbs */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-primary/5 rounded-3xl blur-3xl"
        animate={{
          scale: [1, 1.1, 1],
          opacity: [0.5, 0.8, 0.5],
        }}
        transition={{
          duration: 8,
          repeat: Number.POSITIVE_INFINITY,
          ease: "easeInOut",
        }}
      />

      {/* Main card container */}
      <motion.div className="relative z-10 w-full max-w-sm" variants={itemVariants}>
        {/* Video card */}
        <motion.div
          className="bg-gradient-to-br from-background to-muted border border-primary/20 rounded-2xl p-4 shadow-2xl backdrop-blur-sm"
          variants={floatVariants}
          animate="animate"
        >
          {/* Video thumbnail placeholder */}
          <div className="relative mb-3 overflow-hidden rounded-lg bg-muted aspect-video flex items-center justify-center group cursor-pointer">
            <motion.div
              className="absolute inset-0 bg-gradient-to-r from-primary/20 to-transparent"
              animate={{
                x: ["-100%", "100%"],
              }}
              transition={{
                duration: 3,
                repeat: Number.POSITIVE_INFINITY,
                ease: "linear",
              }}
            />
            <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.95 }} className="relative z-10">
              <Play className="w-10 h-10 text-primary fill-primary" />
            </motion.div>
          </div>

          {/* Card content */}
          <div className="space-y-2">
            <h3 className="font-semibold text-sm text-foreground">YouTube Video</h3>
            <p className="text-xs text-muted-foreground">Upload any YouTube video</p>
          </div>
        </motion.div>

        {/* Arrow animation */}
        <motion.div
          className="flex justify-center my-2"
          animate={{ y: [0, 8, 0] }}
          transition={{
            duration: 2,
            repeat: Number.POSITIVE_INFINITY,
            ease: "easeInOut",
          }}
        >
          <div className="text-primary text-lg">↓</div>
        </motion.div>

        {/* Transformation cards */}
        <div className="space-y-2">
          {[
            { icon: BookOpen, label: "Chapters", delay: 0.1 },
            { icon: CheckCircle2, label: "Quizzes", delay: 0.2 },
          ].map((item, index) => (
            <motion.div
              key={index}
              className="bg-gradient-to-r from-primary/10 to-primary/5 border border-primary/20 rounded-lg p-2 flex items-center gap-2 backdrop-blur-sm"
              variants={itemVariants}
              custom={item.delay}
              whileHover={{
                x: 8,
                backgroundColor: "rgba(var(--primary), 0.15)",
              }}
              transition={{
                type: "spring",
                stiffness: 300,
                damping: 30,
              }}
            >
              <motion.div
                animate={{ rotate: [0, 360] }}
                transition={{
                  duration: 20,
                  repeat: Number.POSITIVE_INFINITY,
                  ease: "linear",
                }}
              >
                <item.icon className="w-4 h-4 text-primary" />
              </motion.div>
              <span className="text-xs font-medium text-foreground">{item.label}</span>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Floating accent elements */}
      <motion.div
        className="absolute top-10 right-10 w-16 h-16 bg-primary/10 rounded-full blur-2xl"
        animate={{
          scale: [1, 1.2, 1],
          opacity: [0.3, 0.6, 0.3],
        }}
        transition={{
          duration: 5,
          repeat: Number.POSITIVE_INFINITY,
          ease: "easeInOut",
        }}
      />
      <motion.div
        className="absolute bottom-10 left-10 w-24 h-24 bg-primary/5 rounded-full blur-3xl"
        animate={{
          scale: [1.2, 1, 1.2],
          opacity: [0.2, 0.5, 0.2],
        }}
        transition={{
          duration: 7,
          repeat: Number.POSITIVE_INFINITY,
          ease: "easeInOut",
        }}
      />
    </motion.div>
  )
}
