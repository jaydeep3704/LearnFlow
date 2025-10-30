import whisper
import yt_dlp
import os
import subprocess
import torch
import threading
import re # Needed for chapter extraction
from datetime import datetime

class TranscriptionService:
    """Handles video downloading, audio chunking, and GPU-accelerated Whisper transcription."""

    def __init__(self):
        self.whisper_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"TranscriptionService initialized. Using device: {self.device}")

    def _load_model(self, model_name="base.en"): # Confirmed: Using base.en for balanced speed/accuracy
        if self.whisper_model is None:
            print(f"Loading Whisper model ({model_name}) to {self.device}...")
            self.whisper_model = whisper.load_model(model_name, device=self.device)
        return self.whisper_model

    def _get_video_duration(self, youtube_url):
        ydl_opts = {
            'quiet': True, 'no_warnings': True, 'extract_flat': False,
            'extractor_args': {'youtube': {'player_client': ['android', 'web'],}},
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                return info.get('duration', 0)
        except Exception as e:
            return 3600

    def _get_video_id(self, youtube_url):
        # Implementation to safely extract video ID
        match = re.search(r"(?<=v=)[\w-]+|(?<=youtu\.be/)[\w-]+", youtube_url)
        return match.group(0) if match else youtube_url

    def _download_audio_chunk(self, youtube_url, start_time, duration, chunk_file):
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best', 'outtmpl': chunk_file.replace('.mp3', ''), 
            'quiet': True, 'no_warnings': True,
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '128',}],
            'extractor_args': {'youtube': {'player_client': ['android', 'web'], 'player_skip': ['webpage', 'configs'],}},
            'download_ranges': yt_dlp.utils.download_range_func(None, [(start_time, start_time + duration)]),
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([youtube_url])
            
            if os.path.exists(chunk_file): return chunk_file
            
            base_name = chunk_file.replace('.mp3', '')
            if os.path.exists(base_name + '.mp3'):
                if base_name + '.mp3' != chunk_file: os.rename(base_name + '.mp3', chunk_file)
                return chunk_file
            return None
        except Exception as e:
            return None

    def _transcribe_chunk(self, chunk_file, offset_sec):
        if not os.path.exists(chunk_file) or os.path.getsize(chunk_file) < 10000: return []

        try:
            result = self.whisper_model.transcribe(
                chunk_file, word_timestamps=True, fp16=False, language='en'
            )
            if not result or 'segments' not in result: return []

            words = []
            for seg in result.get('segments', []):
                if 'words' in seg and seg['words']:
                    for word in seg['words']:
                        word_text = word.get('word', '').strip()
                        if word_text:
                            words.append({
                                "word": word_text, "start": word.get('start', 0) + offset_sec,
                                "end": word.get('end', 0) + offset_sec
                            })
            return words
        except Exception as e:
            print(f"Error transcribing chunk: {e}")
            return []

    def _extract_chapters_from_description(self, youtube_url):
        """
        Retrieves the video description and extracts chapter markers (timestamps) using yt-dlp metadata.
        """
        
        ydl_opts = {'quiet': True, 'no_warnings': True, 'skip_download': True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                description = info.get('description', '')
                video_duration = info.get('duration', 0)
        except Exception as e:
            return []

        # Regex to find timestamp patterns (HH:MM:SS or MM:SS)
        pattern = re.compile(r'(\d{1,2}:\d{2}(:\d{2})?)\s+([^\n]+)', re.MULTILINE)
        matches = pattern.findall(description)
        
        chapters = []
        
        def time_to_seconds(time_str):
            parts = [int(p) for p in time_str.split(':')]
            if len(parts) == 3: # HH:MM:SS
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2: # MM:SS
                return parts[0] * 60 + parts[1]
            return 0

        for i, match in enumerate(matches):
            time_str = match[0] 
            title = match[2].strip() 
            start_time = time_to_seconds(time_str)
            
            # Simple validation: First timestamp must be 0:00
            if i == 0 and start_time != 0:
                return []
            
            chapters.append({'time': start_time, 'title': title})

        # Process chapters to assign end times and ensure valid length
        final_chapters = []
        for i, chap in enumerate(chapters):
            # Assign end time: start of next chapter, or video duration for the last chapter
            end_time = chapters[i+1]['time'] if i + 1 < len(chapters) else video_duration
            
            # Only include chapters that have a duration > 0
            if end_time > chap['time']:
                final_chapters.append({
                    "index": i + 1,
                    "title": chap['title'],
                    "start_time": float(chap['time']),
                    "end_time": float(end_time),
                    "duration_seconds": float(end_time - chap['time']),
                    "keywords": [], # Empty list since we skip keyword extraction
                    "transcript": [], # Empty list since we skip ASR/transcription
                })

        # Require a minimum of 3 chapters for a structured output
        return final_chapters if len(final_chapters) > 2 else []

    def transcribe_video(self, job_id, jobs_dict, youtube_url):
        """Main sequential transcription pipeline with bug fix."""
        jobs_dict[job_id]['status'] = 'transcribing'
        print(f"Job {job_id}: Status -> transcribing")
        
        try:
            self._load_model()
            jobs_dict[job_id]['message'] = 'Getting video info...'
            total_duration = self._get_video_duration(youtube_url) or 3600
            jobs_dict[job_id]['total_duration'] = total_duration
            
            chunk_duration = 300
            start_time = 0
            all_words = []
            chunk_index = 0

            # --- Sequential Loop ---
            while start_time < total_duration:
                
                current_chunk_duration = min(chunk_duration, total_duration - start_time)
                
                if current_chunk_duration <= 0: break 

                chunk_index += 1
                chunk_file = f"chunk_{job_id}_{chunk_index}.mp3"
                
                jobs_dict[job_id]['message'] = f'Processing chunk {chunk_index} (Download & Transcribe)'
                jobs_dict[job_id]['progress'] = (start_time / total_duration) * 50
                print(f"Job {job_id}: Progress {jobs_dict[job_id]['progress']:.0f}% -> {jobs_dict[job_id]['message']}")

                downloaded = self._download_audio_chunk(youtube_url, start_time, current_chunk_duration, chunk_file)

                if downloaded and os.path.exists(downloaded) and os.path.getsize(downloaded) >= 10000:
                    
                    words = self._transcribe_chunk(downloaded, start_time)
                    
                    # Clip Timestamps (Bug Fix)
                    final_words = []
                    for word in words:
                        word['end'] = min(word['end'], total_duration)
                        word['start'] = min(word['start'], total_duration)
                        if word['start'] < word['end']:
                             final_words.append(word)

                    if final_words:
                        all_words.extend(final_words)
                        print(f"Job {job_id}: Chunk {chunk_index} - transcribed {len(final_words)} words")
                    
                else:
                    print(f"Job {job_id}: Chunk {chunk_index} skipped (download failed or file too small)")
                
                try: os.remove(chunk_file)
                except: pass
                
                start_time += chunk_duration

            if not all_words: raise Exception("No speech detected.")
            
            jobs_dict[job_id]['transcript_words'] = all_words
            jobs_dict[job_id]['status'] = 'transcribed'
            jobs_dict[job_id]['progress'] = 50
            jobs_dict[job_id]['message'] = f'Transcribed {len(all_words)} words'
            print(f"Job {job_id}: Status -> Transcribed. Total words: {len(all_words)}")
            
        except Exception as e:
            jobs_dict[job_id]['status'] = 'error'
            jobs_dict[job_id]['error'] = str(e)
            print(f"Job {job_id}: ERROR in transcription -> {e}")