import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

from .base_model import BaseModel
from config import DEVICE

class HuggingFaceModel(BaseModel):
    """Simplified wrapper for Hugging Face transformer models."""

    def __init__(self, model_id: str):
        super().__init__(model_id)
        print(f"Loading Hugging Face model: {model_id} onto device: {DEVICE}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
      

        print(f"Successfully loaded model: {model_id}")

    def predict(self, sentences: List[str]) -> List[float]:
        sentences = [text.strip('"') for text in sentences]
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).numpy().tolist()
        
        return [prob[0] for prob in probs]