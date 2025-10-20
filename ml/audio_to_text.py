# pip install yt-dlp openai-whisper
import yt_dlp
import whisper
import os
import json
import subprocess

# -----------------------------
# Get video duration (with better error handling)
# -----------------------------
def get_video_duration(youtube_url):
    """Get video duration without downloading"""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'format': 'worst',  # Just get info, don't care about quality
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            
            # Handle playlists
            if 'entries' in info:
                # If it's a playlist, get the first video
                info = info['entries'][0]
            
            return info.get('duration', 0)
    except Exception as e:
        print(f"Warning: Could not get duration: {e}")
        return None

# -----------------------------
# Download a single chunk using FFmpeg directly
# -----------------------------
def download_audio_chunk_direct(youtube_url, start_time, duration, chunk_file="chunk.mp3"):
    """
    Download a specific time range using yt-dlp with better format handling
    """
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best',
        'outtmpl': 'temp_download',
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        # Better format selection to avoid YouTube issues
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
            # Handle playlists
            info = ydl.extract_info(youtube_url, download=False)
            if 'entries' in info:
                # If playlist URL, extract first video URL
                video_url = info['entries'][0]['url']
            else:
                video_url = youtube_url
            
            ydl.download([video_url])
        
        # After postprocessing, the file should be temp_download.mp3
        temp_file = 'temp_download.mp3'
        
        # If mp3 doesn't exist, check for other formats
        if not os.path.exists(temp_file):
            for ext in ['m4a', 'webm', 'opus']:
                if os.path.exists(f'temp_download.{ext}'):
                    temp_file = f'temp_download.{ext}'
                    break
        
        if temp_file and os.path.exists(temp_file):
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
            os.rename(temp_file, chunk_file)
            return chunk_file
        
        return None
        
    except Exception as e:
        print(f"  Error downloading chunk: {e}")
        return None

# -----------------------------
# Fallback: Download full audio then extract chunk
# -----------------------------
def download_and_extract_chunk(youtube_url, start_time, duration, chunk_file="chunk.mp3", full_audio="full_audio.mp3"):
    """
    Fallback method: Download full audio once, then extract chunks with FFmpeg
    """
    # Download full audio if not already downloaded
    if not os.path.exists(full_audio):
        print("  Downloading full audio (one-time)...")
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'outtmpl': 'temp_full',
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
                }
            },
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                if 'entries' in info:
                    video_url = info['entries'][0]['url']
                else:
                    video_url = youtube_url
                ydl.download([video_url])
            
            # After postprocessing, the file should be temp_full.mp3
            if os.path.exists('temp_full.mp3'):
                os.rename('temp_full.mp3', full_audio)
            else:
                # Check for other formats
                for ext in ['m4a', 'webm']:
                    if os.path.exists(f'temp_full.{ext}'):
                        os.rename(f'temp_full.{ext}', full_audio)
                        break
        except Exception as e:
            print(f"  Error downloading full audio: {e}")
            return None
    
    # Extract chunk using FFmpeg
    if os.path.exists(full_audio):
        cmd = [
            'ffmpeg', '-i', full_audio,
            '-ss', str(start_time),
            '-t', str(duration),
            '-acodec', 'libmp3lame',
            '-y',
            chunk_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 1000:
            return chunk_file
    
    return None

# -----------------------------
# Transcribe a single chunk
# -----------------------------
def transcribe_chunk(chunk_file, offset_sec, model):
    """Transcribe a single audio chunk and return words with adjusted timestamps"""
    print(f"  Transcribing...")
    result = model.transcribe(chunk_file, word_timestamps=True)
    
    words = []
    for seg in result.get('segments', []):
        if 'words' in seg:
            for word in seg['words']:
                words.append({
                    "word": word.get('word', '').strip(),
                    "start": word.get('start', 0) + offset_sec,
                    "end": word.get('end', 0) + offset_sec
                })
    
    return words

# -----------------------------
# Append words to transcript file
# -----------------------------
def append_to_transcript(words, filename="transcript.json"):
    """Append new words to transcript file (create if doesn't exist)"""
    existing_words = []
    
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                existing_words = json.load(f)
        except:
            existing_words = []
    
    existing_words.extend(words)
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(existing_words, f, ensure_ascii=False, indent=2)
    
    return len(existing_words)

# -----------------------------
# Main streaming pipeline
# -----------------------------
if __name__ == "__main__":
    youtube_url = input("Enter YouTube URL: ").strip()
    
    if not youtube_url:
        print("Error: No URL provided")
        exit(1)
    
    chunk_duration = 300  # 5 minutes per chunk
    transcript_file = "transcript.json"
    chunk_file = "current_chunk.mp3"
    full_audio_file = "full_audio.mp3"
    use_fallback = False
    
    # Clean up old files
    if os.path.exists(transcript_file):
        response = input(f"{transcript_file} exists. Overwrite? (y/n): ").strip().lower()
        if response == 'y':
            os.remove(transcript_file)
            print(f"Removed old {transcript_file}")
    
    try:
        # Get total video duration
        print("\nGetting video info...")
        total_duration = get_video_duration(youtube_url)
        
        if total_duration:
            print(f"Video duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        else:
            print("Could not determine duration, will process until end")
            total_duration = 10000  # Large number as fallback
        
        # Load Whisper model once
        print("\nLoading Whisper model...")
        model = whisper.load_model("base")
        
        # Process chunks one at a time
        start_time = 0
        chunk_index = 0
        total_words = 0
        consecutive_failures = 0
        
        while start_time < total_duration:
            chunk_index += 1
            print(f"\n--- Chunk {chunk_index} (starting at {start_time:.1f}s) ---")
            
            # Try direct download first
            if not use_fallback:
                print(f"  Downloading chunk...")
                downloaded_file = download_audio_chunk_direct(
                    youtube_url, 
                    start_time, 
                    chunk_duration, 
                    chunk_file
                )
                
                # If direct download fails, switch to fallback
                if not downloaded_file:
                    print("  Direct download failed, switching to fallback method...")
                    use_fallback = True
            
            # Use fallback method
            if use_fallback:
                downloaded_file = download_and_extract_chunk(
                    youtube_url,
                    start_time,
                    chunk_duration,
                    chunk_file,
                    full_audio_file
                )
            
            # Check if download succeeded
            if not downloaded_file or not os.path.exists(downloaded_file):
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print(f"  Multiple failures, stopping")
                    break
                start_time += chunk_duration
                continue
            
            if os.path.getsize(downloaded_file) < 1000:
                print(f"  Chunk too small, reached end")
                os.remove(downloaded_file)
                break
            
            consecutive_failures = 0
            
            # Transcribe
            words = transcribe_chunk(downloaded_file, start_time, model)
            print(f"  Found {len(words)} words")
            
            # Append to transcript
            total_words = append_to_transcript(words, transcript_file)
            print(f"  Total words so far: {total_words}")
            
            # Delete chunk
            os.remove(downloaded_file)
            
            # Move to next chunk
            start_time += chunk_duration
        
        # Clean up full audio if used
        if os.path.exists(full_audio_file):
            os.remove(full_audio_file)
            print(f"\nCleaned up temporary files")
        
        print(f"\n✓ Transcription complete!")
        print(f"✓ Total words transcribed: {total_words}")
        print(f"✓ Saved to: {transcript_file}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if os.path.exists(full_audio_file):
            os.remove(full_audio_file)
        exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(full_audio_file):
            os.remove(full_audio_file)
        exit(1)