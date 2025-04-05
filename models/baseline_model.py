from typing import List
from .base_model import BaseModel

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity


class BaselineModel(BaseModel):
    def __init__(self, model_id: str = "bert-formality"):
        super().__init__(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        
        self.formal_terms = [
            "acquire", "reside", "commence", "terminate", "request",
            "ascertain", "apologize", "inquire", "construct", "depart"
        ]
        self.informal_terms = [
            "get", "live", "start", "end", "ask",
            "find", "sorry", "check", "build", "leave"
        ]
        
        self.formal_emb = self._get_average_embedding(self.formal_terms)
        self.informal_emb = self._get_average_embedding(self.informal_terms)

    def _get_embeddings(self, words: List[str]) -> np.ndarray:
        inputs = self.tokenizer(words, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.mean(outputs.last_hidden_state, dim=1).numpy()

    def _get_average_embedding(self, words: List[str]) -> np.ndarray:
        embeddings = self._get_embeddings(words)
        return np.mean(embeddings, axis=0)

    def _word_similarity(self, word_emb: np.ndarray) -> float:
        formal_sim = cosine_similarity(word_emb, self.formal_emb.reshape(1, -1))[0][0]
        informal_sim = cosine_similarity(word_emb, self.informal_emb.reshape(1, -1))[0][0]
        return formal_sim - informal_sim

    def predict(self, sentences: List[str]) -> List[float]:
        scores = []
        for sentence in sentences:
            tokens = self.tokenizer(sentence.split(), return_tensors="pt", 
                                  padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**tokens)
            
            word_embeddings = torch.mean(outputs.last_hidden_state, dim=1).numpy()
            word_scores = [self._word_similarity(emb.reshape(1, -1)) for emb in word_embeddings]
            
            # Normalize final score to 0-1 range
            avg_score = np.mean(word_scores)
            norm_score = 1 / (1 + np.exp(-avg_score))
            scores.append(float(norm_score))
        
        return scores