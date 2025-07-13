import sys
from transformers import AutoTokenizer, AutoModel
from loguru import logger
import torch
import numpy as np
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class EmbeddingModel:
    def __init__(self, model_name='BAAI/bge-m3', device=None, use_half=False, offline=False):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half = use_half
        self.offline = offline
        
        # Auto-disable half precision on CPU
        if self.device == "cpu":
            self.use_half = False
            
        logger.info(f"Loading embedding model: {model_name} on {self.device} (half={self.use_half}, offline={offline})")
        
        # Simplified loading with offline support
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            local_files_only=offline
        )
        self.model = AutoModel.from_pretrained(
            model_name, 
            local_files_only=offline
        ).to(self.device)
        
        if self.use_half:
            self.model = self.model.half()
            
        self.model.eval()
        torch.set_grad_enabled(False)

    def embed(self, texts, batch_size=16):
        if isinstance(texts, str):
            texts = [texts]
        all_embeddings = []
        
        # Optimize batch size for CPU
        if self.device == "cpu":
            batch_size = min(batch_size, 4)
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=256  # Reduced from 512 for faster processing
            ).to(self.device)
            
            if self.use_half:
                inputs = {k: v.half() for k, v in inputs.items()}
                
            with torch.inference_mode():
                outputs = self.model(**inputs)
                # Use mean pooling instead of CLS token for better semantic representation
                embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                if self.use_half:
                    embeddings = embeddings.half()
                all_embeddings.append(embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for better semantic representation"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_one(self, text):
        return self.embed([text])[0]