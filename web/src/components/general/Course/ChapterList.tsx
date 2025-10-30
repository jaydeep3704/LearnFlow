"use client"

import { Clock, Loader2 } from 'lucide-react';
import { formatTime } from '@/lib/time-utils';
import type { Chapter } from '@/types/course';

interface ChapterListProps {
  chapters: Chapter[];
  onChapterClick: (chapter: Chapter) => void;
  currentChapter: Chapter | null;
}

export function ChapterList({ chapters, onChapterClick, currentChapter }: ChapterListProps) {
  if (!chapters || chapters.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <Loader2 className="h-8 w-8 animate-spin mx-auto mb-2" />
        <p>Generating chapters...</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {chapters.map((chapter, idx) => (
        <button
          key={chapter.id}
          onClick={() => onChapterClick(chapter)}
          className={`w-full text-left p-3 rounded-lg transition-colors ${
            currentChapter?.id === chapter.id
              ? ' border-2 bg-blue-400/20  border-blue-500'
              : ' border-2 border-transparent'
          }`}
        >
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-semibold">
              {idx + 1}
            </div>
            <div className="flex-1 min-w-0">
              <h4 className="font-medium text-sm truncate">{chapter.title}</h4>
              <div className="flex items-center gap-2 mt-1 text-xs text-gray-500">
                <Clock className="h-3 w-3" />
                <span>{formatTime(chapter.start)}</span>
                <span>•</span>
                <span>{Math.round((chapter.end - chapter.start) / 60)} min</span>
              </div>
              {/* {chapter.keywords && chapter.keywords.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {chapter.keywords.slice(0, 3).map((keyword, i) => (
                    <span key={i} className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
                      {keyword}
                    </span>
                  ))}
                </div>
              )} */}
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}