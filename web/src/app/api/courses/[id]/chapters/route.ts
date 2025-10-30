import { NextRequest, NextResponse } from 'next/server';
import { auth } from '@/utils/auth';
import prisma from '@/lib/prisma';

// POST /api/courses/:id/chapters - Save chapters from ML processing
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const session = await auth();
    
    if (!session?.user?.id) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { id } = await params;
    const body = await request.json();
    const { chapters } = body;

    if (!chapters || !Array.isArray(chapters)) {
      return NextResponse.json(
        { error: 'Chapters array is required' },
        { status: 400 }
      );
    }

    // Verify course ownership
    const course = await prisma.course.findUnique({
      where: {
        id: id,
        userId: session.user.id,
      },
    });

    if (!course) {
      return NextResponse.json({ error: 'Course not found' }, { status: 404 });
    }

    // Delete existing chapters
    await prisma.chapter.deleteMany({
      where: {
        courseId: id,
      },
    });

    // Create new chapters
    const createdChapters = await prisma.chapter.createMany({
      data: chapters.map((chapter: any) => ({
        courseId: id,
        title: chapter.title,
        start: chapter.start.toString(),
        end: chapter.end.toString(),
        index: chapter.index,
        keywords: chapter.keywords || [],
      })),
    });

    return NextResponse.json({ 
      success: true, 
      count: createdChapters.count 
    });
  } catch (error) {
    console.error('Error saving chapters:', error);
    return NextResponse.json(
      { error: 'Failed to save chapters' },
      { status: 500 }
    );
  }
}