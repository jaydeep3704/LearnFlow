"use client"

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Loader2, PlayCircle } from 'lucide-react';
import { getYouTubeVideoId } from '@/lib/youtube-utils';
import { ChapterList } from './ChapterList';
import type { Course, Chapter, MLJobStatus, MLTopic } from '@/types/course';

interface CourseViewerClientProps {
  course: Course;
}

export function CourseViewerClient({ course: initialCourse }: CourseViewerClientProps) {
  const router = useRouter();
  const [course, setCourse] = useState(initialCourse);
  const [chapters, setChapters] = useState<Chapter[]>(course.chapters || []);
  const [currentChapter, setCurrentChapter] = useState<Chapter | null>(null);
  const [isProcessing, setIsProcessing] = useState(course.status === 'PROCESSING');
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');

  const videoId = getYouTubeVideoId(course.youtubeUrl);

  // Poll for processing status
  useEffect(() => {
    if (!course.jobId || course.status === 'COMPLETED' || course.status === 'FAILED') {
      setIsProcessing(false);
      return;
    }

    let isMounted = true;

    const pollStatus = async () => {
      try {
        const response = await fetch(`http://localhost:5000/api/status/${course.jobId}`);
        
        if (!response.ok) {
          console.error('Failed to fetch status');
          return;
        }

        const status: MLJobStatus = await response.json();

        if (!isMounted) return;

        setProgress(status.progress || 0);
        setStatusMessage(status.message || 'Processing...');

        console.log('Status:', status.status, 'Progress:', status.progress);

        if (status.status === 'completed') {
          console.log('ML processing completed, fetching results...');
          
          // Fetch results from ML server
          const resultResponse = await fetch(`http://localhost:5000/api/result/${course.jobId}`);
          
          if (!resultResponse.ok) {
            throw new Error('Failed to fetch results');
          }

          const result = await resultResponse.json();
          console.log('ML Results:', result);

          if (!result.topics || result.topics.length === 0) {
            throw new Error('No topics returned from ML server');
          }

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

          console.log('Converted chapters:', newChapters);

          // Save chapters to database
          const saveResponse = await fetch(`/api/courses/${course.id}/chapters`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chapters: newChapters }),
          });

          if (!saveResponse.ok) {
            throw new Error('Failed to save chapters');
          }

          console.log('Chapters saved to database');

          // Update course status
          const updateResponse = await fetch(`/api/courses/${course.id}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: 'COMPLETED' }),
          });

          if (!updateResponse.ok) {
            throw new Error('Failed to update course status');
          }

          console.log('Course status updated to COMPLETED');

          setChapters(newChapters);
          setCourse(prev => ({ ...prev, status: 'COMPLETED', chapters: newChapters }));
          setIsProcessing(false);
          setStatusMessage('Processing complete!');
          
          // Refresh the page to get updated data from server
          router.refresh();
          
        } else if (status.status === 'error') {
          console.error('ML processing error:', status.error);
          setIsProcessing(false);
          setStatusMessage('Processing failed: ' + (status.error || 'Unknown error'));
          
          // Update course status to FAILED
          await fetch(`/api/courses/${course.id}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: 'FAILED' }),
          });

          setCourse(prev => ({ ...prev, status: 'FAILED' }));
        }
      } catch (error) {
        console.error('Error polling status:', error);
        if (isMounted) {
          setStatusMessage('Error checking status. Will retry...');
        }
      }
    };

    // Initial poll
    pollStatus();

    // Poll every 3 seconds
    const interval = setInterval(pollStatus, 3000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [course.jobId, course.id, course.status, router]);

  const handleChapterClick = (chapter: Chapter) => {
    setCurrentChapter(chapter);
  };

  const videoUrl = currentChapter
    ? `https://www.youtube.com/embed/${videoId}?start=${Math.floor(currentChapter.start)}&autoplay=1`
    : `https://www.youtube.com/embed/${videoId}`;

  return (
    <div className="min-h-screen  p-4">
      <div className="max-w-7xl mx-auto">
        <Button 
          onClick={() => router.push('/courses')} 
          variant="outline" 
          className="mb-4"
        >
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
                    <div className="mt-2 p-3  rounded-lg">
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
                    {progress < 50 
                      ? 'Transcribing audio...' 
                      : 'Generating chapters...'}
                  </p>
                </CardContent>
              </Card>
            )}

            {/* Failed Status */}
            {course.status === 'FAILED' && (
              <Card className="mt-4 border-red-200">
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="h-5 w-5 rounded-full bg-red-500 flex items-center justify-center text-white text-xs">
                      !
                    </div>
                    <h3 className="font-semibold text-red-600">Processing Failed</h3>
                  </div>
                  <p className="text-sm text-gray-600">
                    The video processing encountered an error. Please try creating the course again.
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
                    : isProcessing 
                    ? 'Generating chapters...'
                    : 'No chapters yet'}
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