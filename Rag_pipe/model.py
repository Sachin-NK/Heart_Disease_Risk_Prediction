from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import pipeline
from loguru import logger
import torch
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class QwenLLM:
    def __init__(self, model_name='Qwen/Qwen3-0.6B', device=None, 
                 max_new_tokens=30, load_in_4bit=True, offline=False):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.offline = offline
        
        # Auto-disable quantization on CPU
        if self.device == "cpu":
            load_in_4bit = False
            
        logger.info(f"Loading Qwen LLM: {model_name} on {self.device} (4bit={load_in_4bit}, offline={offline})")
        
        # Optimized loading with offline support
        kwargs = {
            "trust_remote_code": True,
            "local_files_only": offline
        }
        
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
        elif self.device == "cuda":
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.float16
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            local_files_only=offline
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            **kwargs
        )
        
        # Manual CPU handling with optimizations
        if self.device == "cpu":
            self.model = self.model.float()
            self.model = self.model.to(self.device)
            # Enable optimizations for CPU inference
            torch.set_num_threads(1)  # Single thread for deterministic results
        # Create optimized pipeline for fast inference
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            max_new_tokens=max_new_tokens,
            batch_size=1,  # Process one at a time for consistency
            framework="pt"
        )

    def generate(self, prompt, **kwargs):
        # Optimized generation parameters for speed and accuracy
        gen_args = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", 0.1),  # Low temperature for deterministic output
            "do_sample": kwargs.get("do_sample", False),  # Greedy decoding for speed
            "repetition_penalty": 1.1,  # Minimal repetition penalty
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,  # Enable caching for speed
            "return_full_text": False  # Only return generated part
        }
        
        # Override with any provided kwargs
        for key in ["top_p", "top_k"]:
            if key in kwargs:
                gen_args[key] = kwargs[key]
        
        output = self.generator(prompt, **gen_args)
        generated_text = output[0]["generated_text"]
        
        return generated_text.strip()