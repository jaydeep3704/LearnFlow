"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { CreateCourseModal } from "@/components/general/Course/CreateCourseModal"
import { Plus } from "lucide-react"

export function CreateCourseButton() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <>
      <Button onClick={() => setIsOpen(true)} size="lg" className="gap-2">
        <Plus className="w-5 h-5" />
        Create Course
      </Button>
      <CreateCourseModal isOpen={isOpen} onOpenChange={setIsOpen} />
    </>
  )
}
