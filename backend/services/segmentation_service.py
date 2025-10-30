from models.llm_models import LLMTitleGenerator, LLMTranscriptCleaner
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class SegmentationService:
    """Manages transcript cleaning, topic modeling, segment merging, and contextual title generation."""

    def __init__(self):
        self.llm_cleaner = LLMTranscriptCleaner()
        self.llm_generator = LLMTitleGenerator()
        
    def _get_full_cleaned_text(self, transcript_words):
        """Reconstructs the full cleaned text from the list of word objects."""
        return " ".join([w["word"] for w in transcript_words])

    def _generate_contextual_title(self, full_text, segment_text):
        """Calls the LLMTitleGenerator with the full transcript for global context."""
        return self.llm_generator.generate_contextual_title(full_text, segment_text)

    def _create_time_windows(self, transcript_words, window_seconds=45, overlap_seconds=25): # Boundary Fix 1: Increased overlap
        if not transcript_words: return []
        
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

    def _cluster_topics(self, windows, min_topic_size=10): # Boundary Fix 2: Increased min_topic_size
        texts = [w["text"] for w in windows]
        
        # Dynamic min_df fix for short documents
        num_documents = len(texts)
        vectorizer_min_df = 1 if num_documents < 2 else 2

        # Boundary Fix 3: Stronger Sentence Transformer Model
        embedding_model = SentenceTransformer('all-mpnet-base-v2') 
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2), stop_words='english', max_features=500, min_df=vectorizer_min_df
        )
        
        topic_model = BERTopic(
            embedding_model=embedding_model, vectorizer_model=vectorizer_model, 
            min_topic_size=min_topic_size, verbose=False, calculate_probabilities=False, nr_topics="auto"
        )
        
        topics, _ = topic_model.fit_transform(texts)
        
        for i, window in enumerate(windows): window["topic"] = topics[i]
        
        topic_model.reduce_topics(texts, nr_topics="auto")
        reduced_topics = topic_model.topics_
        
        for i, window in enumerate(windows): window["topic"] = reduced_topics[i]
        
        return windows

    def _merge_segments(self, windows, min_duration=90):
        if not windows: return []
        segments = []
        current = {"topic": windows[0]["topic"], "texts": [windows[0]["text"]], "start": windows[0]["start"], "end": windows[0]["end"]}
        
        for window in windows[1:]:
            if window["topic"] == current["topic"]:
                current["texts"].append(window["text"]); current["end"] = window["end"]
            else:
                if current["end"] - current["start"] >= min_duration: segments.append(current)
                current = {"topic": window["topic"], "texts": [window["text"]], "start": window["start"], "end": window["end"]}
        
        if current["texts"] and current["end"] - current["start"] >= min_duration: segments.append(current)
        
        for seg in segments: seg["text"] = " ".join(seg["texts"]); del seg["texts"]
        
        merged = [segments[0]] if segments else []
        for seg in segments[1:]:
            if seg["topic"] == merged[-1]["topic"]:
                merged[-1]["text"] += " " + seg["text"]; merged[-1]["end"] = seg["end"]
            else: merged.append(seg)
        return merged

    def _get_segment_transcript(self, segment, transcript_words):
        words = [w for w in transcript_words if w["start"] >= segment["start"] and w["end"] <= segment["end"]]
        return words


    def segment_topics(self, job_id, jobs_dict):
        """Main segmentation pipeline (updates jobs_dict directly)."""
        jobs_dict[job_id]['status'] = 'segmenting'

        try:
            # ───────────────────────────────
            # Initial Setup and Cleaning
            # ───────────────────────────────
            transcript_words = jobs_dict[job_id]['transcript_words']
            if not transcript_words: raise Exception("No transcript words available")

            jobs_dict[job_id]['message'] = 'Cleaning transcript with LLM...'
            jobs_dict[job_id]['progress'] = 52
            
            raw_text = " ".join([w["word"] for w in transcript_words])
            cleaned_text = self.llm_cleaner.clean_text(raw_text)

            # Reconstruct pseudo-timed words from cleaned text
            cleaned_words = [{"word": word, "start": i * 0.5, "end": i * 0.5 + 0.5} 
                             for i, word in enumerate(cleaned_text.split())]
            transcript_words = cleaned_words 
            
            full_cleaned_transcript = self._get_full_cleaned_text(transcript_words)

            # ───────────────────────────────
            # Segmentation and Merging
            # ───────────────────────────────
            jobs_dict[job_id]['message'] = 'Creating time windows...'
            jobs_dict[job_id]['progress'] = 55
            # Passed the adjusted overlap_seconds=25 (Boundary Fix 1)
            windows = self._create_time_windows(transcript_words, 45, 25) 
            if not windows: raise Exception("Could not create time windows")

            jobs_dict[job_id]['message'] = 'Clustering topics...'
            jobs_dict[job_id]['progress'] = 65
            # Passed the adjusted min_topic_size=10 (Boundary Fix 2)
            windows = self._cluster_topics(windows, min_topic_size=10) 

            jobs_dict[job_id]['message'] = 'Merging segments...'
            jobs_dict[job_id]['progress'] = 75
            segments = self._merge_segments(windows, min_duration=90)
            if not segments: raise Exception("Could not create segments")

            # ───────────────────────────────
            # Global Context Title Generation
            # ───────────────────────────────
            jobs_dict[job_id]['message'] = 'Generating highly contextual titles...'
            
            for i, seg in enumerate(segments):
                jobs_dict[job_id]['progress'] = 75 + (i / len(segments)) * 20

                # Use the new helper method that includes global context
                title, keywords = self._generate_contextual_title(
                    full_cleaned_transcript, 
                    seg["text"]             
                )
                
                seg["title"] = title
                seg["keywords"] = keywords
                seg["transcript"] = self._get_segment_transcript(seg, transcript_words)

            # ───────────────────────────────
            # Final Output Packaging
            # ───────────────────────────────
            output = [{"index": i, "title": seg["title"], "start_time": seg["start"], "end_time": seg["end"],
                       "duration_seconds": seg["end"] - seg["start"], "keywords": seg["keywords"][:5],
                       "transcript": seg["transcript"]} for i, seg in enumerate(segments, 1)]

            jobs_dict[job_id]['topics'] = output
            jobs_dict[job_id]['status'] = 'completed'
            jobs_dict[job_id]['progress'] = 100
            jobs_dict[job_id]['message'] = f'Generated {len(output)} topic segments'
            
        except Exception as e:
            jobs_dict[job_id]['status'] = 'error'
            jobs_dict[job_id]['error'] = str(e)
            print(f"Job {job_id}: Segmentation error - {e}")