import threading
import time
import uuid
import re
from datetime import datetime
from services.transcription_service import TranscriptionService
from services.segmentation_service import SegmentationService

class JobOrchestrator:
    """Manages the job dictionary, threads, and coordinates transcription and segmentation services."""

    def __init__(self):
        self.jobs = {}
        self.transcription_service = TranscriptionService()
        self.segmentation_service = SegmentationService()

    def _monitor_and_segment(self, job_id):
        """Waits for transcription to finish, then starts segmentation."""
        print(f"Job {job_id}: Monitor thread started, waiting for transcription...") 

        while self.jobs[job_id]['status'] not in ['transcribed', 'error']:
            time.sleep(1) 
            
        if self.jobs[job_id]['status'] == 'transcribed':
            print(f"Job {job_id}: Transcription complete. Starting segmentation.")
            self.segmentation_service.segment_topics(job_id, self.jobs)

    def start_job(self, youtube_url):
        """Initializes a job, checks description, and launches threads."""
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            'id': job_id,
            'url': youtube_url,
            'status': 'queued',
            'progress': 0,
            'message': 'Starting...',
            'created_at': datetime.now().isoformat()
        }
        
        print(f"Job {job_id}: Processing started for URL: {youtube_url}")

        # ----------------------------------------------------
        # CRITICAL STEP 1: Check Description for Chapters (Fast Check)
        # ----------------------------------------------------
        self.jobs[job_id]['message'] = 'Checking description for embedded chapters...'
        print(f"Job {job_id}: Executing fast check for embedded chapters...")
        
        # NOTE: This call is synchronous and fast, as it only fetches metadata.
        # This uses the new method in the TranscriptionService
        direct_chapters = self.transcription_service._extract_chapters_from_description(youtube_url)
        
        if direct_chapters:
            # ----------------------------------------------------
            # SUCCESS PATH: Skip all slow processing
            # ----------------------------------------------------
            self.jobs[job_id]['topics'] = direct_chapters
            self.jobs[job_id]['status'] = 'completed'
            self.jobs[job_id]['progress'] = 100
            self.jobs[job_id]['message'] = f'SUCCESS: {len(direct_chapters)} chapters extracted directly from description.'
            print(f"Job {job_id}: SUCCESS - Chapters found in description. Processing complete.")
            return job_id
        
        # ----------------------------------------------------
        # FALLBACK PATH: Start slow ASR/Clustering pipeline
        # ----------------------------------------------------
        self.jobs[job_id]['message'] = 'No chapters found. Starting ASR/Clustering fallback...'
        print(f"Job {job_id}: No chapters found. Starting slow ASR/Clustering fallback...")

        # Thread 1: Transcription (ASR/Download work)
        transcribe_thread = threading.Thread(
            target=self.transcription_service.transcribe_video, 
            args=(job_id, self.jobs, youtube_url)
        )
        transcribe_thread.daemon = True
        transcribe_thread.start()
        
        # Thread 2: Segmentation Monitor
        monitor_thread = threading.Thread(
            target=self._monitor_and_segment, 
            args=(job_id,)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return job_id

    def get_status(self, job_id):
        return self.jobs.get(job_id)

    def get_result(self, job_id):
        return self.jobs.get(job_id)
    
    def get_active_jobs(self):
        return len([j for j in self.jobs.values() if j['status'] in ['queued', 'transcribing', 'segmenting']])