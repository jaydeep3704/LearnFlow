import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer

class LLMTitleGenerator:
    """Generates concise chapter titles using global video context and GPU acceleration."""
    
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        print(f"Loading LLM Title Generator ({model_name}) onto GPU...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto", 
                trust_remote_code=True
            )
            self.model.eval()
        except Exception as e:
            print(f"Error loading LLMTitleGenerator: {e}. Falling back to keyword titles.")
            self.model = None

    def extract_keywords(self, text, top_n=5):
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), stop_words='english', max_features=top_n * 2
            )
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            top_indices = scores.argsort()[-top_n:][::-1]
            keywords = [feature_names[idx] for idx in top_indices if scores[idx] > 0]
            return keywords[:top_n]
        except:
            return []

    def _create_keyword_title(self, keywords):
        if not keywords: return "Discussion Topic"
        multi_word = [kw for kw in keywords if len(kw.split()) > 1]
        if multi_word: return multi_word[0].title()
        if len(keywords) >= 2: return f"{keywords[0].title()} and {keywords[1].title()}"
        return keywords[0].title()

    def generate_contextual_title(self, full_text, segment_text):
        """Generates a title using the full transcript for global context."""
        
        MAX_CONTEXT_LENGTH = 4000
        full_text_truncated = full_text[:MAX_CONTEXT_LENGTH] 

        prompt = f"""You are an expert chapter generator for educational videos. 
Here is the full cleaned transcript of the video for GLOBAL CONTEXT:
--- FULL VIDEO CONTEXT START ---
{full_text_truncated}
--- FULL VIDEO CONTEXT END ---

The following is the SPECIFIC SEGMENT you must title:
--- SEGMENT TO TITLE START ---
{segment_text}
--- SEGMENT TO TITLE END ---

Analyze the specific segment within the context of the whole video. Generate a concise, specific, descriptive title (3-7 words) that captures the **main action or key concept** discussed ONLY in the segment to title. Do NOT use generic phrases like 'Introduction' or 'Conclusion'. Output ONLY the title.

Title:"""
        
        try:
            if self.model is not None:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=20, temperature=0.3, do_sample=True, top_p=0.9,
                        repetition_penalty=1.2, pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if "Title:" in generated_text:
                    title = generated_text.split("Title:")[-1].strip()
                else:
                    lines = [l.strip() for l in generated_text.split("\n") if l.strip()]
                    title = lines[-1] if lines else ""
                
                title = title.split("\n")[0].strip().strip('"\'.,;:!?')
                
                if len(title) < 3: 
                    raise Exception("LLM output poor, falling back.")
            else:
                raise Exception("LLM not loaded, falling back.")

        except Exception as e:
            keywords = self.extract_keywords(segment_text, top_n=5)
            title = self._create_keyword_title(keywords)
        
        keywords = self.extract_keywords(segment_text, top_n=5)
        return title, keywords

class LLMTranscriptCleaner:
    """Cleans a transcript by removing filler words and disfluencies using an LLM (Optimized for GPU)."""

    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        print(f"Loading LLM for transcript cleaning: {model_name} onto GPU...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto", 
                trust_remote_code=True
            )
            self.model.eval()
        except Exception as e:
            print(f"Error loading cleaner model: {e}")
            self.model = None

    def clean_text(self, text, max_length=1500):
        if self.model is None: return text
        if len(text.split()) > max_length: text = " ".join(text.split()[:max_length])

        prompt = f"""You are an expert transcription cleaner.
Rewrite the following transcript by removing filler words, repetitions, and false starts.
Do NOT change the meaning or important words.
Transcript: {text}
Cleaned Transcript:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=300, temperature=0.4, top_p=0.9, repetition_penalty=1.15, pad_token_id=self.tokenizer.eos_token_id
                )

            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            cleaned = result.split("Cleaned Transcript:")[-1].strip() if "Cleaned Transcript:" in result else result.strip()
            return cleaned
        except Exception as e:
            return text