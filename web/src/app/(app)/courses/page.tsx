"use client"

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2 } from 'lucide-react';
import { CreateCourseModal } from '@/components/general/Course/CreateCourseModal';
import { CourseCard } from '@/components/general/Course/CourseCard';
import type { Course } from '@/types/course';

export default function CoursesPage() {
  const [courses, setCourses] = useState<Course[]>([]);
  const [selectedCourse, setSelectedCourse] = useState<Course | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Load courses on mount
  useEffect(() => {
    loadCourses();
  }, []);

  const loadCourses = async () => {
    try {
      const response = await fetch('/api/courses');
      if (!response.ok) throw new Error('Failed to load courses');
      
      const data = await response.json();
      setCourses(data);
    } catch (error) {
      console.error('Error loading courses:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCourseCreated = (newCourse: Course) => {
    setCourses([newCourse, ...courses]);
    // Navigate to the new course page
    window.location.href = `/courses/${newCourse.id}`;
  };

  // If a course is selected, redirect to detail page
  if (selectedCourse) {
    window.location.href = `/courses/${selectedCourse.id}`;
    return null;
  }

  // Otherwise show the courses grid
  return (
    <section className="min-h-screen w-full px-[4%] py-4 ">
      {/* Hero Section */}
      <Card className="py-20 max-w-7xl h-[400px] bg-[url(/illustration.jpg)] bg-cover bg-center mx-auto relative rounded-3xl overflow-hidden mb-8">
        <div className="absolute inset-0 bg-black/60"></div>
        <div className="relative z-10">
          <CardHeader>
            <CardTitle className="text-center font-bold lg:text-5xl text-2xl drop-shadow-md text-white">
              Welcome Back!
            </CardTitle>
            <CardDescription className="text-center lg:text-2xl text-lg font-bold text-gray-300 drop-shadow-md">
              Continue your learning journey or create a new course from any YouTube video
            </CardDescription>
          </CardHeader>
          <CardContent className="flex justify-center items-center py-5">
            <Button
              size="lg"
              onClick={() => setIsModalOpen(true)}
              className="text-lg px-8 py-6"
            >
              + Create New Course
            </Button>
          </CardContent>
        </div>
      </Card>

      {/* Course Grid */}
      <div className="max-w-7xl mx-auto">
        <h2 className="text-2xl font-bold mb-6">Your Courses</h2>
        
        {isLoading ? (
          <div className="text-center py-12">
            <Loader2 className="h-8 w-8 animate-spin mx-auto" />
          </div>
        ) : courses.length === 0 ? (
          <Card className="p-12 text-center">
            <p className="text-gray-500 mb-4 text-xl">No courses yet</p>
            <Button onClick={() => setIsModalOpen(true)} variant={"outline"} className='w-sm mx-auto'>Create Your First Course</Button>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {courses.map((course) => (
              <CourseCard
                key={course.id}
                course={course}
                onClick={() => setSelectedCourse(course)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Create Course Modal */}
      <CreateCourseModal
        isOpen={isModalOpen}
        onOpenChange={setIsModalOpen}
        onCourseCreated={handleCourseCreated}
      />
    </section>
  );
}