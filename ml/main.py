# pip install flask flask-cors yt-dlp openai-whisper bertopic sentence-transformers
# pip install umap-learn hdbscan transformers torch scikit-learn
from flask import Flask, request, jsonify
from flask_cors import CORS
import yt_dlp
import whisper
import os
import json
import subprocess
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import threading
import uuid
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global storage for processing jobs
jobs = {}
whisper_model = None
llm_generator = None

# ============================================================================
# TRANSCRIPTION FUNCTIONS
# ============================================================================

def get_video_duration(youtube_url):
    """Get video duration without downloading"""
    ydl_opts = {
        'quiet': True, 
        'no_warnings': True, 
        'extract_flat': False,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],
            }
        },
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            if 'entries' in info:
                info = info['entries'][0]
            return info.get('duration', 0)
    except Exception as e:
        print(f"Warning: Could not get duration: {e}")
        return None

def download_audio_chunk(youtube_url, start_time, duration, chunk_file):
    """Download a specific time range using yt-dlp"""
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best',
        'outtmpl': chunk_file.replace('.mp3', ''),
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],
                'player_skip': ['webpage', 'configs'],
            }
        },
        'download_ranges': yt_dlp.utils.download_range_func(None, [(start_time, start_time + duration)]),
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        # Check for mp3 file
        if os.path.exists(chunk_file):
            return chunk_file
        
        # Check without extension
        base_name = chunk_file.replace('.mp3', '')
        if os.path.exists(base_name + '.mp3'):
            if base_name + '.mp3' != chunk_file:
                os.rename(base_name + '.mp3', chunk_file)
            return chunk_file
        
        return None
    except Exception as e:
        print(f"Error downloading chunk: {e}")
        return None

def extract_audio_segment(input_file, start_time, duration, output_file):
    """Extract a segment from full audio file using FFmpeg"""
    try:
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-ss', str(start_time),
            '-t', str(duration),
            '-acodec', 'libmp3lame',
            '-q:a', '2',
            '-y',
            output_file
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60
        )
        
        if result.returncode == 0 and os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            if file_size > 1000:  # At least 1KB
                return output_file
        
        return None
    except Exception as e:
        print(f"Error extracting segment: {e}")
        return None

def transcribe_chunk(chunk_file, offset_sec, model):
    """Transcribe a single audio chunk with better validation"""
    try:
        # Validate file exists
        if not os.path.exists(chunk_file):
            print(f"Chunk file does not exist: {chunk_file}")
            return []
        
        # Validate file size (at least 10KB)
        file_size = os.path.getsize(chunk_file)
        if file_size < 10000:
            print(f"Chunk file too small ({file_size} bytes)")
            return []
        
        print(f"Transcribing {chunk_file} ({file_size} bytes)...")
        
        # Transcribe with CPU mode to avoid GPU tensor issues
        result = model.transcribe(
            chunk_file, 
            word_timestamps=True, 
            fp16=False,  # Force CPU mode
            language='en'  # Set language to avoid detection issues
        )
        
        if not result or 'segments' not in result:
            print(f"No result from transcription")
            return []
        
        words = []
        for seg in result.get('segments', []):
            if 'words' in seg and seg['words']:
                for word in seg['words']:
                    word_text = word.get('word', '').strip()
                    if word_text:
                        words.append({
                            "word": word_text,
                            "start": word.get('start', 0) + offset_sec,
                            "end": word.get('end', 0) + offset_sec
                        })
        
        print(f"Transcribed {len(words)} words")
        return words
    
    except Exception as e:
        print(f"Error transcribing chunk: {e}")
        import traceback
        traceback.print_exc()
        return []

def transcribe_video(job_id, youtube_url):
    """Background task to transcribe video using chunked download"""
    global whisper_model
    jobs[job_id]['status'] = 'transcribing'
    
    try:
        # Load Whisper model
        if whisper_model is None:
            jobs[job_id]['message'] = 'Loading Whisper model...'
            print(f"Job {job_id}: Loading Whisper model...")
            whisper_model = whisper.load_model("base")
        
        # Get duration
        jobs[job_id]['message'] = 'Getting video info...'
        print(f"Job {job_id}: Getting video info...")
        total_duration = get_video_duration(youtube_url)
        if not total_duration:
            total_duration = 3600  # Default to 1 hour max
        
        jobs[job_id]['total_duration'] = total_duration
        print(f"Job {job_id}: Video duration: {total_duration}s")
        
        # Process in chunks
        chunk_duration = 300  # 5 minutes
        start_time = 0
        all_words = []
        chunk_index = 0
        consecutive_failures = 0
        
        while start_time < total_duration:
            chunk_index += 1
            chunk_file = f"chunk_{job_id}_{chunk_index}.mp3"
            
            jobs[job_id]['message'] = f'Processing chunk {chunk_index} ({start_time:.0f}s / {total_duration:.0f}s)'
            jobs[job_id]['progress'] = (start_time / total_duration) * 50
            
            print(f"Job {job_id}: Downloading chunk {chunk_index} at {start_time}s")
            
            # Download chunk
            downloaded = download_audio_chunk(youtube_url, start_time, chunk_duration, chunk_file)
            
            if not downloaded or not os.path.exists(downloaded):
                consecutive_failures += 1
                print(f"Job {job_id}: Chunk {chunk_index} download failed (attempt {consecutive_failures}/3)")
                
                if consecutive_failures >= 3:
                    print(f"Job {job_id}: Too many failures, stopping")
                    break
                
                start_time += chunk_duration
                continue
            
            # Validate file size
            file_size = os.path.getsize(downloaded)
            if file_size < 10000:  # Less than 10KB indicates problem
                print(f"Job {job_id}: Chunk {chunk_index} too small ({file_size} bytes), skipping")
                os.remove(downloaded)
                start_time += chunk_duration
                continue
            
            # Reset failure counter on success
            consecutive_failures = 0
            print(f"Job {job_id}: Chunk {chunk_index} downloaded ({file_size} bytes)")
            
            # Transcribe segment
            words = transcribe_chunk(downloaded, start_time, whisper_model)
            
            if words:
                all_words.extend(words)
                print(f"Job {job_id}: Chunk {chunk_index} - transcribed {len(words)} words")
            else:
                print(f"Job {job_id}: Chunk {chunk_index} - no words (silent/empty)")
            
            # Clean up chunk
            try:
                os.remove(downloaded)
            except:
                pass
            
            start_time += chunk_duration
        
        # Check results
        if not all_words:
            raise Exception("No speech detected in the video. The video might be silent or music-only.")
        
        jobs[job_id]['transcript_words'] = all_words
        jobs[job_id]['status'] = 'transcribed'
        jobs[job_id]['progress'] = 50
        jobs[job_id]['message'] = f'Transcribed {len(all_words)} words'
        print(f"Job {job_id}: Transcription complete - {len(all_words)} words total")
        
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        print(f"Job {job_id}: Error in transcription - {e}")

# ============================================================================
# TOPIC SEGMENTATION FUNCTIONS (unchanged)
# ============================================================================

class LLMTitleGenerator:
    """Generate topic titles using a small LLM"""
    
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        print(f"Loading LLM model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def extract_keywords(self, text, top_n=5):
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words='english',
                max_features=top_n * 2
            )
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            top_indices = scores.argsort()[-top_n:][::-1]
            keywords = [feature_names[idx] for idx in top_indices if scores[idx] > 0]
            return keywords[:top_n]
        except:
            return []
    
    def generate_title(self, text, max_length=1000):
        words = text.split()
        if len(words) > max_length:
            sample_text = " ".join(words[:600]) + " ... " + " ".join(words[-200:])
        else:
            sample_text = text
        
        prompt = f"""Read this transcript and create a short title describing the main topic.

Transcript:
{sample_text}

Create a concise title (2-6 words) that captures the specific topic being discussed. Output ONLY the title.

Title:"""

        try:
            if self.model is not None:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=20,
                        temperature=0.3,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if "Title:" in generated_text:
                    title = generated_text.split("Title:")[-1].strip()
                else:
                    lines = [l.strip() for l in generated_text.split("\n") if l.strip()]
                    title = lines[-1] if lines else ""
                
                title = title.split("\n")[0].strip().strip('"\'.,;:!?')
                
                if len(title) < 3 or len(title.split()) > 10 or len(title.split()) < 2:
                    keywords = self.extract_keywords(text, top_n=5)
                    title = self._create_keyword_title(keywords)
            else:
                keywords = self.extract_keywords(text, top_n=5)
                title = self._create_keyword_title(keywords)
        
        except Exception as e:
            print(f"Error generating title: {e}")
            keywords = self.extract_keywords(text, top_n=5)
            title = self._create_keyword_title(keywords)
        
        keywords = self.extract_keywords(text, top_n=5)
        return title, keywords
    
    def _create_keyword_title(self, keywords):
        if not keywords:
            return "Discussion Topic"
        multi_word = [kw for kw in keywords if len(kw.split()) > 1]
        if multi_word:
            return multi_word[0].title()
        if len(keywords) >= 2:
            return f"{keywords[0].title()} and {keywords[1].title()}"
        return keywords[0].title()

def create_time_windows(transcript_words, window_seconds=45, overlap_seconds=15):
    if not transcript_words:
        return []
    
    windows = []
    start_time = transcript_words[0]["start"]
    end_time = transcript_words[-1]["end"]
    current_time = start_time
    step = window_seconds - overlap_seconds
    
    while current_time < end_time:
        window_end = current_time + window_seconds
        window_words = [
            w for w in transcript_words 
            if w["start"] >= current_time and w["start"] < window_end
        ]
        
        if window_words and len(window_words) > 10:
            text = " ".join([w["word"] for w in window_words])
            windows.append({
                "text": text,
                "start": window_words[0]["start"],
                "end": window_words[-1]["end"],
            })
        current_time += step
    
    return windows

def cluster_topics(windows, min_topic_size=5):
    texts = [w["text"] for w in windows]
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        max_features=500,
        min_df=2
    )
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        verbose=False,
        calculate_probabilities=False,
        nr_topics="auto"
    )
    
    topics, _ = topic_model.fit_transform(texts)
    
    for i, window in enumerate(windows):
        window["topic"] = topics[i]
    
    topic_model.reduce_topics(texts, nr_topics="auto")
    reduced_topics = topic_model.topics_
    
    for i, window in enumerate(windows):
        window["topic"] = reduced_topics[i]
    
    return windows

def merge_segments(windows, min_duration=90):
    if not windows:
        return []
    
    segments = []
    current = {
        "topic": windows[0]["topic"],
        "texts": [windows[0]["text"]],
        "start": windows[0]["start"],
        "end": windows[0]["end"]
    }
    
    for window in windows[1:]:
        if window["topic"] == current["topic"]:
            current["texts"].append(window["text"])
            current["end"] = window["end"]
        else:
            if current["end"] - current["start"] >= min_duration:
                segments.append(current)
            current = {
                "topic": window["topic"],
                "texts": [window["text"]],
                "start": window["start"],
                "end": window["end"]
            }
    
    if current["texts"] and current["end"] - current["start"] >= min_duration:
        segments.append(current)
    
    for seg in segments:
        seg["text"] = " ".join(seg["texts"])
        del seg["texts"]
    
    merged = [segments[0]] if segments else []
    for seg in segments[1:]:
        if seg["topic"] == merged[-1]["topic"]:
            merged[-1]["text"] += " " + seg["text"]
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)
    
    return merged

def get_segment_transcript(segment, transcript_words):
    words = [
        w for w in transcript_words
        if w["start"] >= segment["start"] and w["end"] <= segment["end"]
    ]
    return words

class LLMTranscriptCleaner:
    """Cleans a transcript by removing filler words and disfluencies using an LLM."""

    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        print(f"Loading LLM for transcript cleaning: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            self.model.eval()
        except Exception as e:
            print(f"Error loading cleaner model: {e}")
            self.model = None

    def clean_text(self, text, max_length=1500):
        if len(text.split()) > max_length:
            text = " ".join(text.split()[:max_length])

        prompt = f"""You are an expert transcription cleaner.
Rewrite the following transcript by removing filler words, repetitions, and false starts.
Do NOT change the meaning or important words.

Transcript:
{text}

Cleaned Transcript:"""

        try:
            if self.model is None:
                return text  # fallback if model not loaded

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.4,
                    top_p=0.9,
                    repetition_penalty=1.15,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Cleaned Transcript:" in result:
                cleaned = result.split("Cleaned Transcript:")[-1].strip()
            else:
                cleaned = result.strip()
            return cleaned
        except Exception as e:
            print(f"Error during cleaning: {e}")
            return text






def segment_topics(job_id):
    global llm_generator, llm_cleaner
    jobs[job_id]['status'] = 'segmenting'

    try:
        # ───────────────────────────────
        # Load models
        # ───────────────────────────────
        if llm_cleaner is None:
            jobs[job_id]['message'] = 'Loading LLM cleaner...'
            llm_cleaner = LLMTranscriptCleaner()

        if llm_generator is None:
            jobs[job_id]['message'] = 'Loading LLM title generator...'
            llm_generator = LLMTitleGenerator()

        # ───────────────────────────────
        # Fetch and clean transcript
        # ───────────────────────────────
        transcript_words = jobs[job_id]['transcript_words']
        if not transcript_words:
            raise Exception("No transcript words available")

        jobs[job_id]['message'] = 'Cleaning transcript with LLM...'
        jobs[job_id]['progress'] = 52

        # Convert transcript to plain text
        raw_text = " ".join([w["word"] for w in transcript_words])

        # Clean using LLM
        cleaned_text = llm_cleaner.clean_text(raw_text)

        # Reconstruct cleaned transcript into pseudo-timed word objects
        cleaned_words = []
        for i, word in enumerate(cleaned_text.split()):
            cleaned_words.append({
                "word": word,
                "start": i * 0.5,      # rough timestamp estimate
                "end": i * 0.5 + 0.5
            })

        transcript_words = cleaned_words

        # ───────────────────────────────
        # Segment transcript into topics
        # ───────────────────────────────
        jobs[job_id]['message'] = 'Creating time windows...'
        jobs[job_id]['progress'] = 55
        windows = create_time_windows(transcript_words, 45, 15)

        if not windows:
            raise Exception("Could not create time windows")

        jobs[job_id]['message'] = 'Clustering topics...'
        jobs[job_id]['progress'] = 65
        windows = cluster_topics(windows, min_topic_size=5)

        jobs[job_id]['message'] = 'Merging segments...'
        jobs[job_id]['progress'] = 75
        segments = merge_segments(windows, min_duration=90)

        if not segments:
            raise Exception("Could not create segments")

        # ───────────────────────────────
        # Generate titles per segment
        # ───────────────────────────────
        jobs[job_id]['message'] = 'Generating titles...'
        for i, seg in enumerate(segments):
            jobs[job_id]['progress'] = 75 + (i / len(segments)) * 20

            # Generate concise, contextual titles
            title, keywords = llm_generator.generate_title(seg["text"])
            seg["title"] = title
            seg["keywords"] = keywords
            seg["transcript"] = get_segment_transcript(seg, transcript_words)

        # ───────────────────────────────
        # Prepare final output
        # ───────────────────────────────
        output = []
        for i, seg in enumerate(segments, 1):
            output.append({
                "index": i,
                "title": seg["title"],
                "start_time": seg["start"],
                "end_time": seg["end"],
                "duration_seconds": seg["end"] - seg["start"],
                "keywords": seg["keywords"][:5],
                "transcript": seg["transcript"]
            })

        jobs[job_id]['topics'] = output
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['message'] = f'Generated {len(output)} topic segments'
        print(f"Job {job_id}: Completed with {len(output)} topics")

    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        print(f"Job {job_id}: Segmentation error - {e}")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/process', methods=['POST'])
def process_video():
    data = request.json
    youtube_url = data.get('url')
    
    if not youtube_url:
        return jsonify({'error': 'No URL provided'}), 400
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'id': job_id,
        'url': youtube_url,
        'status': 'queued',
        'progress': 0,
        'message': 'Starting...',
        'created_at': datetime.now().isoformat()
    }
    
    thread = threading.Thread(target=transcribe_video, args=(job_id, youtube_url))
    thread.daemon = True
    thread.start()
    
    def monitor_and_segment():
        while jobs[job_id]['status'] not in ['transcribed', 'error']:
            threading.Event().wait(1)
        if jobs[job_id]['status'] == 'transcribed':
            segment_topics(job_id)
    
    monitor_thread = threading.Thread(target=monitor_and_segment)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    return jsonify({'job_id': job_id, 'status': 'processing'}), 202

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    response = {
        'id': job['id'],
        'status': job['status'],
        'progress': job.get('progress', 0),
        'message': job.get('message', '')
    }
    
    if job['status'] == 'error':
        response['error'] = job.get('error', 'Unknown error')
    
    return jsonify(response)

@app.route('/api/result/<job_id>', methods=['GET'])
def get_result(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({
            'status': job['status'],
            'message': 'Processing not completed yet'
        }), 202
    
    return jsonify({
        'status': 'completed',
        'topics': job['topics']
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'whisper': whisper_model is not None,
            'llm': llm_generator is not None
        },
        'active_jobs': len([j for j in jobs.values() if j['status'] in ['queued', 'transcribing', 'segmenting']])
    })

if __name__ == '__main__':
    print("Starting YouTube Topic Segmentation Server...")
    print("Server will be available at http://localhost:5000")
    print("\nEndpoints:")
    print("  POST /api/process - Start processing (body: {url: 'youtube_url'})")
    print("  GET  /api/status/<job_id> - Check status")
    print("  GET  /api/result/<job_id> - Get results")
    print("  GET  /api/health - Health check")
    
    app.run(debug=True, host='0.0.0.0', port=5000)