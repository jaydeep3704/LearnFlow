"use client"

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { PlusIcon } from 'lucide-react';
import { CreateCourseModal } from './CreateCourseModal';
import type { Course } from '@/types/course';

interface CreateCourseButtonProps {
  onCourseCreated?: (course: Course) => void;
}

export function CreateCourseButton({ onCourseCreated }: CreateCourseButtonProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleCourseCreated = (course: Course) => {
    if (onCourseCreated) {
      onCourseCreated(course);
    }
    // Optionally refresh the page to show the new course
    window.location.reload();
  };

  return (
    <>
      <Button
        size="lg"
        onClick={() => setIsModalOpen(true)}
        className="text-lg px-8 py-6"
      >
        <PlusIcon className="mr-2 h-5 w-5" />
        Create New Course
      </Button>

      <CreateCourseModal
        isOpen={isModalOpen}
        onOpenChange={setIsModalOpen}
        onCourseCreated={handleCourseCreated}
      />
    </>
  );
}