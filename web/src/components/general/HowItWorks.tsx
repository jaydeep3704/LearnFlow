"use client"
import { motion } from "framer-motion"
import { Card, CardDescription, CardHeader, CardTitle } from "../ui/card"

interface HowItWorksCardProps {
  title: string
  description: string
  index: number
}

const list = [
  {
    title: "Paste Youtube Link",
    description: "Simply paste any YouTube video URL to get started",
  },
  {
    title: "AI Processes Content",
    description: "Our AI generates transcripts, identifies topics, and creates chapters",
  },
  {
    title: "Learn & Track Progress",
    description: "Take quizzes, make notes, and track your learning journey",
  },
]

export function HowItWorks() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.1,
      },
    },
  }

  return (
    <div className="px-[4%] lg:py-20 py-10 flex flex-col gap-4 max-w-7xl mx-auto">
      <motion.h1
        initial={{ opacity: 0, y: -20 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        viewport={{ once: true }}
        className="text-center text-4xl font-bold"
      >
        How It Works
      </motion.h1>
      <motion.p
        initial={{ opacity: 0, y: -10 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
        viewport={{ once: true }}
        className="text-center text-muted-foreground text-lg"
      >
        Our AI-powered platform makes learning from YouTube videos effortless and effective
      </motion.p>
      <motion.div
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        className="py-10 flex w-full gap-6 flex-wrap md:flex-row flex-col"
      >
        {list.map((item, index) => (
          <HowItWorksCard index={index} key={index} title={item.title} description={item.description} />
        ))}
      </motion.div>
    </div>
  )
}

const HowItWorksCard = ({ title, description, index }: HowItWorksCardProps) => {
  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5 },
    },
  }

  return (
    <motion.div variants={cardVariants} whileHover={{ y: -8, transition: { duration: 0.3 } }} className="flex-1">
      <Card className="flex-1 border-none h-full">
        <CardHeader>
          <motion.div
            initial={{ scale: 0, rotate: -180 }}
            whileInView={{ scale: 1, rotate: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
            viewport={{ once: true }}
            className="h-10 w-10 rounded-lg bg-gradient-to-br from-purple-500 to-purple-600 flex items-center justify-center text-white font-bold text-lg"
          >
            {index + 1}
          </motion.div>
          <CardTitle className="text-xl">{title}</CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
      </Card>
    </motion.div>
  )
}
