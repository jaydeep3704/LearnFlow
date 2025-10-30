"use client"

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Loader2, PlayCircle } from 'lucide-react';
import { getYouTubeVideoId } from '@/lib/youtube-utils';
import { ChapterList } from './ChapterList';
import type { Course, Chapter, MLJobStatus, MLTopic } from '@/types/course';

interface CourseViewerProps {
  course: Course;
  onBack: () => void;
}

export function CourseViewer({ course, onBack }: CourseViewerProps) {
  const [chapters, setChapters] = useState<Chapter[]>(course.chapters || []);
  const [currentChapter, setCurrentChapter] = useState<Chapter | null>(null);
  const [isProcessing, setIsProcessing] = useState(course.status === 'PROCESSING');
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');

  const videoId = getYouTubeVideoId(course.youtubeUrl);

  // Poll for processing status
  useEffect(() => {
    if (!course.jobId || course.status === 'COMPLETED') {
      setIsProcessing(false);
      return;
    }

    const pollStatus = async () => {
      try {
        const response = await fetch(`http://localhost:5000/api/status/${course.jobId}`);
        const status: MLJobStatus = await response.json();

        setProgress(status.progress || 0);
        setStatusMessage(status.message || '');

        if (status.status === 'completed') {
          // Fetch results from ML server
          const resultResponse = await fetch(`http://localhost:5000/api/result/${course.jobId}`);
          const result = await resultResponse.json();

          // Convert ML topics to chapters
          const newChapters: Chapter[] = result.topics.map((topic: MLTopic) => ({
            id: `chapter-${topic.index}`,
            title: topic.title,
            start: topic.start_time,
            end: topic.end_time,
            index: topic.index,
            keywords: topic.keywords,
            transcript: topic.transcript
          }));

          // Save chapters to database
          await fetch(`/api/courses/${course.id}/chapters`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chapters: newChapters }),
          });

          // Update course status
          await fetch(`/api/courses/${course.id}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: 'COMPLETED' }),
          });

          setChapters(newChapters);
          setIsProcessing(false);
        } else if (status.status === 'error') {
          setIsProcessing(false);
          setStatusMessage('Processing failed: ' + (status.error || 'Unknown error'));
          
          // Update course status to FAILED
          await fetch(`/api/courses/${course.id}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: 'FAILED' }),
          });
        }
      } catch (error) {
        console.error('Error polling status:', error);
      }
    };

    // Poll every 2 seconds
    const interval = setInterval(pollStatus, 2000);
    
    // Initial poll
    pollStatus();

    return () => clearInterval(interval);
  }, [course.jobId, course.id, course.status]);

  const handleChapterClick = (chapter: Chapter) => {
    setCurrentChapter(chapter);
  };

  const videoUrl = currentChapter
    ? `https://www.youtube.com/embed/${videoId}?start=${Math.floor(currentChapter.start)}&autoplay=1`
    : `https://www.youtube.com/embed/${videoId}`;

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto">
        <Button onClick={onBack} variant="outline" className="mb-4">
          ← Back to Courses
        </Button>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Video Player */}
          <div className="lg:col-span-2">
            <Card>
              <CardContent className="p-0">
                <div className="aspect-video">
                  <iframe
                    src={videoUrl}
                    className="w-full h-full rounded-t-lg"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                  />
                </div>
                <div className="p-4">
                  <h1 className="text-2xl font-bold">{course.title}</h1>
                  {currentChapter && (
                    <div className="mt-2 p-3 bg-blue-50 rounded-lg">
                      <p className="text-sm text-gray-600">Currently playing:</p>
                      <p className="font-semibold">{currentChapter.title}</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Processing Status */}
            {isProcessing && (
              <Card className="mt-4">
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-3">
                    <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
                    <h3 className="font-semibold">Processing video...</h3>
                  </div>
                  <Progress value={progress} className="mb-2" />
                  <p className="text-sm text-gray-600">{statusMessage}</p>
                  <p className="text-xs text-gray-500 mt-2">
                    This may take a few minutes. You can watch the video while we process it!
                  </p>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Chapters Sidebar */}
          <div>
            <Card className="sticky top-4">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <PlayCircle className="h-5 w-5" />
                  Chapters
                </CardTitle>
                <CardDescription>
                  {chapters.length > 0 
                    ? `${chapters.length} chapters available`
                    : 'Generating chapters from video...'}
                </CardDescription>
              </CardHeader>
              <CardContent className="max-h-[600px] overflow-y-auto">
                <ChapterList
                  chapters={chapters}
                  onChapterClick={handleChapterClick}
                  currentChapter={currentChapter}
                />
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}