import os
import json
import re
from typing import List
import openai

from .base_model import BaseModel

class LLMModelOpenAI(BaseModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.model_id = model_id
        self.api_key = "YourApiKey"
        self.client = openai.OpenAI(api_key=self.api_key)
        self.progress_file = f"progress{model_id}.json"
        self.processed_sentences = self._load_progress()
        
        print(f"Initialized LLMModel with {model_id}. Loaded {len(self.processed_sentences)} cached results.")

    def _load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Couldn't load progress file - {e}")
                return {}
        return {}

    def _save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.processed_sentences, f)
        print(f"\tProgress saved to {self.progress_file}")

    def _parse_score(self, response: str) -> float:
        match = re.search(r'\d+', response)
        if not match:
            print(f"\tWarning: No number found in response '{response}'. Using neutral score 0.5")
            return 0.5
            
        score = int(match.group())
        normalized = max(0.0, min(1.0, score / 100.0))
        print(f"\tRaw score: {score} → Normalized: {normalized}")
        return normalized

    def predict(self, sentences: List[str]) -> List[float]:
        results = []
        
        for i, sentence in enumerate(sentences):
            print(f"Processing sentence {i+1}/{len(sentences)}")
            
            if sentence in self.processed_sentences:
                print(f"\tUsing cached score: {self.processed_sentences[sentence]}")
                results.append(self.processed_sentences[sentence])
                continue
                
            try:
                prompt = f"""Analyze the formality of the following sentence on a scale of 0 to 100, where 0 is extremely informal (e.g., slang, casual chat) and 100 is highly formal (e.g., academic/professional writing). Consider:   - Vocabulary (slang vs. technical terms)   - Grammar (contractions, sentence structure)   - Tone (casual vs. respectful/professional)   Return **only** the numerical score (0–100) with no explanation.Sentence: {sentence} """

              
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                
                content = response.choices[0].message.content.strip()
                score = self._parse_score(content)
                
                self.processed_sentences[sentence] = score
                self._save_progress()
                
                results.append(score)   

            except Exception as e:
                print(f"\tError processing sentence: {e}")
                results.append(0.5)
                
        return results