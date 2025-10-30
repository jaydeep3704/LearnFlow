export interface Chapter {
  id: string;
  title: string;
  start: number;
  end: number;
  index: number;
  keywords?: string[];
  transcript?: TranscriptWord[];
}

export interface TranscriptWord {
  word: string;
  start: number;
  end: number;
}

export interface Course {
  id: string;
  title: string;
  youtubeUrl: string;
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
  jobId?: string;
  chapters?: Chapter[];
  createdAt?: string;
  updatedAt?: string;
}

export interface MLJobStatus {
  id: string;
  status: 'queued' | 'transcribing' | 'transcribed' | 'segmenting' | 'completed' | 'error';
  progress: number;
  message: string;
  error?: string;
}

export interface MLTopic {
  index: number;
  title: string;
  start_time: number;
  end_time: number;
  duration_seconds: number;
  keywords: string[];
  transcript: TranscriptWord[];
}