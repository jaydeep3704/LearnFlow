"use client"

import type React from "react"

import { useState } from "react"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"

interface CreateCourseModalProps {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
}

export function CreateCourseModal({ isOpen, onOpenChange }: CreateCourseModalProps) {
  const [youtubeUrl, setYoutubeUrl] = useState("")
  const [courseName, setCourseName] = useState("")
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    try {
      // Add your course creation logic here
      console.log({
        youtubeUrl,
        courseName,
      })

      // Reset form and close modal
      setYoutubeUrl("")
      setCourseName("")
      onOpenChange(false)
    } catch (error) {
      console.error("Error creating course:", error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Create New Course</DialogTitle>
          <DialogDescription>Create a new course from a YouTube video. Fill in the details below.</DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="youtube-url">YouTube URL</Label>
            <Input
              id="youtube-url"
              placeholder="https://www.youtube.com/watch?v=..."
              value={youtubeUrl}
              onChange={(e) => setYoutubeUrl(e.target.value)}
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="course-name">Course Name</Label>
            <Input
              id="course-name"
              placeholder="Enter course name"
              value={courseName}
              onChange={(e) => setCourseName(e.target.value)}
              required
            />
          </div>
          <div className="flex justify-end gap-2 pt-4">
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={isLoading}>
              {isLoading ? "Creating..." : "Create Course"}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}
