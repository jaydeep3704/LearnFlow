# pip install bertopic sentence-transformers umap-learn hdbscan transformers torch
import json
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# For LLM-based title generation
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# ============================================================================
# LOAD TRANSCRIPT
# ============================================================================

def load_transcript(filename="transcript.json"):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

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

# ============================================================================
# LLM TITLE GENERATOR
# ============================================================================

class LLMTitleGenerator:
    """Generate topic titles using a small LLM"""
    
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        """
        Initialize LLM for title generation
        
        Recommended lightweight models:
        - microsoft/Phi-3-mini-4k-instruct (3.8B params, very good)
        - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B params, faster)
        - HuggingFaceTB/SmolLM-360M-Instruct (360M params, fastest)
        """
        print(f"Loading LLM model: {model_name}...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            # Load tokenizer and model
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
            print("✓ Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to summarization pipeline...")
            self.model = None
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if self.device == "cuda" else -1)
    
    def extract_keywords(self, text, top_n=5):
        """Extract important keywords from text"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
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
        """Generate a concise title for the text segment using LLM"""
        
        # Use a good chunk of text for context
        words = text.split()
        if len(words) > max_length:
            # Take from beginning and end for better context
            sample_text = " ".join(words[:600]) + " ... " + " ".join(words[-200:])
        else:
            sample_text = text
        
        # Create a generic prompt that works for any domain
        prompt = f"""Read this transcript segment and create a short, specific title that describes the main topic being discussed.

Transcript:
{sample_text}

Create a concise title (2-6 words) that captures what specific topic, concept, or technique is being explained. Output ONLY the title.

Title:"""

        try:
            if self.model is not None:
                # Use the loaded LLM
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
                
                # Extract title from response
                if "Title:" in generated_text:
                    title = generated_text.split("Title:")[-1].strip()
                else:
                    # Get text after the prompt
                    prompt_end_markers = [sample_text[-50:], "Title:", "\n\n"]
                    for marker in prompt_end_markers:
                        if marker in generated_text:
                            parts = generated_text.split(marker)
                            if len(parts) > 1:
                                title = parts[-1].strip()
                                break
                    else:
                        # Take the last line
                        lines = [l.strip() for l in generated_text.split("\n") if l.strip()]
                        title = lines[-1] if lines else ""
                
                # Clean basic formatting
                title = title.split("\n")[0].strip()
                title = title.strip('"\'.,;:!?')
                
                # Basic validation - just check it's not empty or too long
                if len(title) < 3 or len(title.split()) > 10 or len(title.split()) < 2:
                    print(f"  Invalid title length, using keyword fallback")
                    keywords = self.extract_keywords(text, top_n=5)
                    title = self._create_keyword_title(keywords)
                
            else:
                # Fallback: use keywords
                keywords = self.extract_keywords(text, top_n=5)
                title = self._create_keyword_title(keywords)
        
        except Exception as e:
            print(f"Error generating title: {e}")
            keywords = self.extract_keywords(text, top_n=5)
            title = self._create_keyword_title(keywords)
        
        # Extract keywords for output (separate from title generation)
        keywords = self.extract_keywords(text, top_n=5)
        
        return title, keywords
    
    def _create_keyword_title(self, keywords):
        """Create a title from keywords - works for any domain"""
        if not keywords:
            return "Discussion Topic"
        
        # Prioritize multi-word phrases (more specific)
        multi_word = [kw for kw in keywords if len(kw.split()) > 1]
        if multi_word:
            return multi_word[0].title()
        
        # Combine top 2 single keywords
        if len(keywords) >= 2:
            return f"{keywords[0].title()} and {keywords[1].title()}"
        
        return keywords[0].title()
    
    def _is_good_title(self, title):
        """Check if generated title is meaningful"""
        # Check minimum length
        if len(title) < 5 or len(title.split()) < 2:
            return False
        
        # Check if it's too long (probably includes extra text)
        if len(title.split()) > 8:
            return False
        
        # Check if it's too generic or contains unwanted phrases
        bad_phrases = [
            "how a", "how to", "what is", "introduction to",
            "learning about", "understanding the", "exploring",
            "beginner", "guide to", "overview of", "let's learn",
            "in this", "we will", "chapter", "transcript",
            "segment", "section", "lecture"
        ]
        
        title_lower = title.lower()
        for phrase in bad_phrases:
            if phrase in title_lower:
                return False
        
        # Check for meaningless titles
        meaningless = [
            "machine learning", "data science", "learning models",
            "and", "divided", "equals", "data points", "data set"
        ]
        
        if title_lower in meaningless:
            return False
        
        return True
    
    def _clean_title(self, title):
        """Clean and format the generated title"""
        # Remove common prefixes
        prefixes = [
            "title:", "topic:", "subject:", "chapter:", 
            "the topic is", "this is about", "this segment covers",
            "in this section", "we discuss", "we will learn"
        ]
        title_lower = title.lower()
        for prefix in prefixes:
            if title_lower.startswith(prefix):
                title = title[len(prefix):].strip()
                title_lower = title.lower()
        
        # Remove quotes and brackets
        title = title.strip('"\'[](){}')
        
        # Remove trailing punctuation except when it's part of abbreviation
        title = title.rstrip('.,;:!?')
        
        # Remove generic wrappers
        unwanted_starts = ["a ", "an ", "the "]
        title_lower = title.lower()
        for start in unwanted_starts:
            if title_lower.startswith(start):
                title = title[len(start):]
        
        # Capitalize properly - preserve technical terms
        if not any(c.isupper() for c in title):  # Only if no capitals exist
            title = title.title()
        
        # Limit length - ensure complete thought
        words = title.split()
        if len(words) > 8:
            title = " ".join(words[:8])
        
        # Ensure it's not empty
        if not title or len(title) < 3:
            title = "Machine Learning Concepts"
        
        return title.strip()

# ============================================================================
# BERTOPIC CLUSTERING
# ============================================================================

def cluster_topics_with_bertopic(windows, min_topic_size=5):
    texts = [w["text"] for w in windows]
    
    print("Loading sentence transformer...")
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
    
    print("Clustering topics...")
    topics, _ = topic_model.fit_transform(texts)
    
    for i, window in enumerate(windows):
        window["topic"] = topics[i]
    
    print("Reducing similar topics...")
    topic_model.reduce_topics(texts, nr_topics="auto")
    reduced_topics = topic_model.topics_
    
    for i, window in enumerate(windows):
        window["topic"] = reduced_topics[i]
    
    return windows, topic_model

# ============================================================================
# MERGE SEGMENTS
# ============================================================================

def merge_into_segments(windows, min_duration=90):
    if not windows:
        return []
    
    segments = []
    current_segment = {
        "topic": windows[0]["topic"],
        "texts": [windows[0]["text"]],
        "start": windows[0]["start"],
        "end": windows[0]["end"]
    }
    
    for window in windows[1:]:
        if window["topic"] == current_segment["topic"]:
            current_segment["texts"].append(window["text"])
            current_segment["end"] = window["end"]
        else:
            duration = current_segment["end"] - current_segment["start"]
            if duration >= min_duration:
                segments.append(current_segment)
            
            current_segment = {
                "topic": window["topic"],
                "texts": [window["text"]],
                "start": window["start"],
                "end": window["end"]
            }
    
    if current_segment["texts"]:
        duration = current_segment["end"] - current_segment["start"]
        if duration >= min_duration:
            segments.append(current_segment)
    
    for seg in segments:
        seg["text"] = " ".join(seg["texts"])
        del seg["texts"]
    
    return segments

def merge_adjacent_similar(segments):
    if len(segments) <= 1:
        return segments
    
    merged = [segments[0]]
    
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["topic"] == prev["topic"]:
            prev["text"] += " " + seg["text"]
            prev["end"] = seg["end"]
        else:
            merged.append(seg)
    
    return merged

# ============================================================================
# GENERATE TITLES WITH LLM
# ============================================================================

def generate_titles_with_llm(segments, llm_generator):
    """Generate proper titles for all segments using LLM"""
    print("\nGenerating topic titles with LLM...")
    
    for i, seg in enumerate(segments):
        print(f"  Processing segment {i+1}/{len(segments)}...", end="\r")
        
        title, keywords = llm_generator.generate_title(seg["text"])
        
        seg["title"] = title
        seg["keywords"] = keywords
    
    print("\n✓ Generated titles")
    return segments

def calculate_title_similarity(title1, title2):
    """Calculate similarity between two titles"""
    words1 = set(title1.lower().split())
    words2 = set(title2.lower().split())
    
    if not words1 or not words2:
        return 0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def merge_similar_topics(segments, similarity_threshold=0.5):
    """Merge segments with similar titles that are close together"""
    if len(segments) <= 1:
        return segments
    
    print("\nMerging similar topics...")
    merged = []
    i = 0
    
    while i < len(segments):
        current = segments[i]
        
        j = i + 1
        while j < len(segments):
            next_seg = segments[j]
            
            similarity = calculate_title_similarity(current["title"], next_seg["title"])
            
            if similarity >= similarity_threshold and (j - i) <= 3:
                print(f"  Merging '{current['title']}' with '{next_seg['title']}'")
                current["text"] += " " + next_seg["text"]
                current["end"] = next_seg["end"]
                current["keywords"] = list(set(current["keywords"] + next_seg["keywords"]))[:5]
                j += 1
            else:
                break
        
        merged.append(current)
        i = j if j > i + 1 else i + 1
    
    print(f"✓ Reduced from {len(segments)} to {len(merged)} segments")
    return merged

# ============================================================================
# FORMAT OUTPUT
# ============================================================================

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"

def format_output(segments):
    output = []
    
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg["start"])
        duration = seg["end"] - seg["start"]
        
        output.append({
            "index": i,
            "title": seg["title"],
            "timestamp": start,
            "duration_minutes": round(duration / 60, 1),
            "keywords": seg.get("keywords", [])[:5]
        })
    
    return output

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_topic_pipeline(
    transcript_file="transcript.json",
    output_file="topics.json",
    window_seconds=45,
    overlap_seconds=15,
    min_topic_size=5,
    min_segment_duration=90,
    llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Fast and lightweight
):
    
    print("="*70)
    print("TOPIC SEGMENTATION WITH LLM-GENERATED TITLES")
    print("="*70)
    
    print("\n[1/7] Loading transcript...")
    transcript_words = load_transcript(transcript_file)
    total_duration = transcript_words[-1]["end"] if transcript_words else 0
    print(f"✓ Loaded {len(transcript_words)} words ({format_timestamp(total_duration)})")
    
    print(f"\n[2/7] Creating {window_seconds}s windows...")
    windows = create_time_windows(transcript_words, window_seconds, overlap_seconds)
    print(f"✓ Created {len(windows)} time windows")
    
    print("\n[3/7] Clustering topics with BERTopic...")
    windows, topic_model = cluster_topics_with_bertopic(windows, min_topic_size)
    unique_topics = len(set(w["topic"] for w in windows if w["topic"] != -1))
    print(f"✓ Found {unique_topics} distinct topics")
    
    print(f"\n[4/7] Merging into segments (min {min_segment_duration}s)...")
    segments = merge_into_segments(windows, min_segment_duration)
    print(f"✓ Created {len(segments)} initial segments")
    
    segments = merge_adjacent_similar(segments)
    print(f"✓ Merged to {len(segments)} final segments")
    
    print("\n[5/7] Initializing LLM for title generation...")
    llm_generator = LLMTitleGenerator(model_name=llm_model)
    
    print("\n[6/7] Generating topic titles with LLM...")
    segments = generate_titles_with_llm(segments, llm_generator)
    
    segments = merge_similar_topics(segments, similarity_threshold=0.5)
    
    print("\n[7/7] Formatting output...")
    output = format_output(segments)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved to {output_file}")
    
    print("\n" + "="*70)
    print("TOPIC TIMELINE")
    print("="*70)
    
    for item in output:
        print(f"\n{item['index']}) {item['title']}")
        print(f"   {item['timestamp']} ({item['duration_minutes']} min)")
        if item['keywords']:
            print(f"   Keywords: {', '.join(item['keywords'][:3])}")
    
    print("\n" + "="*70)
    print(f"✓ Generated {len(output)} topic segments")
    print("="*70)
    
    return output

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    """
    Recommended LLM models (ordered by speed):
    1. HuggingFaceTB/SmolLM-360M-Instruct - Fastest (360M params)
    2. TinyLlama/TinyLlama-1.1B-Chat-v1.0 - Fast (1.1B params)
    3. microsoft/Phi-3-mini-4k-instruct - Best quality (3.8B params)
    """
    
    topics = run_topic_pipeline(
        transcript_file="transcript.json",
        output_file="topics.json",
        window_seconds=45,
        overlap_seconds=15,
        min_topic_size=5,
        min_segment_duration=90,
        llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Change this to try different models
    )
    
    print("\n✅ Done! Check 'topics.json' for results.")
    print("\nModel options:")
    print("  - SmolLM-360M: Fastest, decent quality")
    print("  - TinyLlama-1.1B: Balanced speed/quality (recommended)")
    print("  - Phi-3-mini: Best quality, slower")