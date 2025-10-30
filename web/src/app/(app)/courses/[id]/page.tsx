import { redirect } from 'next/navigation';
import { auth } from '@/utils/auth';
import prisma from '@/lib/prisma';
import { CourseViewerClient } from '@/components/general/Course/CourseViewerClient';

interface PageProps {
  params: Promise<{ id: string }>;
}

export default async function CoursePage({ params }: PageProps) {
  const session = await auth();
  
  if (!session?.user?.id) {
    redirect('/login');
  }

  const { id } = await params;

  // Fetch course from database
  const course = await prisma.course.findUnique({
    where: {
      id: id,
      userId: session.user.id,
    },
    include: {
      chapters: {
        orderBy: {
          index: 'asc',
        },
      },
    },
  });

  if (!course) {
    redirect('/courses');
  }

  // Convert chapters to the format expected by CourseViewer
  const formattedCourse = {
    ...course,
    chapters: course.chapters.map(chapter => ({
      ...chapter,
      start: parseFloat(chapter.start),
      end: parseFloat(chapter.end),
      keywords: [], // Add if you store keywords in DB
    })),
  };

  return <CourseViewerClient course={formattedCourse} />;
}