"use client"

import { Card, CardContent } from '@/components/ui/card';
import { Loader2, CheckCircle2, XCircle } from 'lucide-react';
import { getYouTubeVideoId } from '@/lib/youtube-utils';
import type { Course } from '@/types/course';

interface CourseCardProps {
  course: Course;
  onClick: () => void;
}

export function CourseCard({ course, onClick }: CourseCardProps) {
  const videoId = getYouTubeVideoId(course.youtubeUrl);
  const thumbnail = `https://img.youtube.com/vi/${videoId}/mqdefault.jpg`;

  const handleClick = () => {
    // Navigate to course detail page
    window.location.href = `/courses/${course.id}`;
  };

  return (
    <Card 
      className="cursor-pointer hover:shadow-lg transition-shadow"
      onClick={handleClick}
    >
      <div className="relative aspect-video">
        <img 
          src={thumbnail} 
          alt={course.title}
          className="w-full h-full object-cover rounded-t-lg"
        />
        {course.status === 'PROCESSING' && (
          <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
            <Loader2 className="h-8 w-8 text-white animate-spin" />
          </div>
        )}
        {course.status === 'COMPLETED' && course.chapters && (
          <div className="absolute bottom-2 right-2 bg-black/80 text-white text-xs px-2 py-1 rounded">
            {course.chapters.length} chapters
          </div>
        )}
      </div>
      <CardContent className="p-4">
        <h3 className="font-semibold text-lg truncate">{course.title}</h3>
        <div className="flex items-center gap-2 mt-2">
          {course.status === 'PROCESSING' && (
            <>
              <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
              <span className="text-sm text-blue-500">Processing...</span>
            </>
          )}
          {course.status === 'COMPLETED' && (
            <>
              <CheckCircle2 className="h-4 w-4 text-green-500" />
              <span className="text-sm text-green-500">Ready</span>
            </>
          )}
          {course.status === 'FAILED' && (
            <>
              <XCircle className="h-4 w-4 text-red-500" />
              <span className="text-sm text-red-500">Failed</span>
            </>
          )}
          {course.status === 'PENDING' && (
            <>
              <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
              <span className="text-sm text-gray-400">Pending...</span>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
}