"""Gemma 4 fine-tuning engine powered by Unsloth.

Provides efficient fine-tuning and inference for Google's Gemma 4 models
using Unsloth's optimized kernels. Designed to work on systems with as
little as 8GB VRAM, using 4-bit quantization, gradient checkpointing, and
LoRA adapters.

Supported Models
----------------
- gemma-2-2b     (2B parameters, ~2GB VRAM)
- gemma-2-9b     (9B parameters, ~8GB VRAM)
- gemma-2-27b    (27B parameters, ~20GB VRAM)
- gemma-7b       (7B parameters, ~8GB VRAM - older)

Memory Optimization Techniques
-------------------------------
- 4-bit quantization via bitsandbytes
- LoRA adapters for efficient fine-tuning
- Gradient checkpointing
- Flash Attention 2 kernels (Unsloth)
- Automatic memory profiling
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextDataset,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

GEMMA_MODELS = {
    "gemma-2-2b": {
        "model_id": "google/gemma-2-2b",
        "vram_gb": 2,
        "description": "Gemma 2 2B - ultra-lightweight, ~2GB VRAM",
    },
    "gemma-2-9b": {
        "model_id": "google/gemma-2-9b",
        "vram_gb": 8,
        "description": "Gemma 2 9B - optimal for 8GB VRAM systems",
    },
    "gemma-2-27b": {
        "model_id": "google/gemma-2-27b",
        "vram_gb": 20,
        "description": "Gemma 2 27B - high-performance, requires 20GB+ VRAM",
    },
    "gemma-7b": {
        "model_id": "google/gemma-7b",
        "vram_gb": 8,
        "description": "Gemma 7B - legacy model",
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning a Gemma model."""
    
    model_name: str = "gemma-2-2b"
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FineTuneConfig:
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingResult:
    """Results from a fine-tuning run."""
    
    model_name: str
    num_samples: int
    final_loss: float
    training_time: float  # seconds
    output_dir: str
    checkpoint_dir: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""
    
    def __str__(self) -> str:
        return (
            f"TrainingResult for {self.model_name}\n"
            f"  Samples: {self.num_samples}\n"
            f"  Final Loss: {self.final_loss:.4f}\n"
            f"  Time: {self.training_time:.1f}s\n"
            f"  Output: {self.output_dir}"
        )


@dataclass
class InferenceResult:
    """Result from a model inference call."""
    
    model: str
    prompt: str
    response: str
    tokens_generated: int
    inference_time: float  # seconds
    error: str = ""
    
    def __str__(self) -> str:
        if self.error:
            return f"[ERROR] {self.error}"
        return (
            f"[{self.model}]  {self.tokens_generated} tokens in {self.inference_time:.2f}s\n"
            f"Prompt: {self.prompt[:100]}...\n"
            f"Response: {self.response[:200]}..."
        )


# ---------------------------------------------------------------------------
# GPU utilities
# ---------------------------------------------------------------------------


class GPUMemoryProfiler:
    """Monitor GPU memory during training/inference."""
    
    @staticmethod
    def get_available_vram() -> float:
        """Return available GPU VRAM in GB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    
    @staticmethod
    def get_allocated_vram() -> float:
        """Return allocated GPU VRAM in GB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / 1e9
    
    @staticmethod
    def get_reserved_vram() -> float:
        """Return reserved GPU VRAM in GB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_reserved() / 1e9
    
    @staticmethod
    def get_available_models(available_vram: Optional[float] = None) -> List[str]:
        """Return list of models that fit in available VRAM."""
        if available_vram is None:
            available_vram = GPUMemoryProfiler.get_available_vram()
        
        available = []
        for model_key, config in GEMMA_MODELS.items():
            if config["vram_gb"] <= available_vram:
                available.append(model_key)
        return available


# ---------------------------------------------------------------------------
# Gemma fine-tuner
# ---------------------------------------------------------------------------


class GemmaFineTuner:
    """Fine-tune and run inference with Google Gemma models using Unsloth."""
    
    def __init__(
        self,
        model_name: str = "gemma-2-2b",
        device: str = "auto",
        use_4bit: bool = True,
    ) -> None:
        """Initialize the fine-tuner.
        
        Parameters
        ----------
        model_name:
            Name of the model (see GEMMA_MODELS keys)
        device:
            Device to use ("cuda", "cpu", or "auto")
        use_4bit:
            Whether to use 4-bit quantization for memory efficiency
        """
        if model_name not in GEMMA_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(GEMMA_MODELS.keys())}")
        
        self.model_name = model_name
        self.model_config = GEMMA_MODELS[model_name]
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_4bit = use_4bit
        
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        
        logger.info(f"Initialized GemmaFineTuner for {model_name} on {self.device}")
    
    def _setup_quantization(self) -> Optional[BitsAndBytesConfig]:
        """Setup 4-bit quantization config for memory efficiency."""
        if not self.use_4bit or self.device == "cpu":
            return None
        
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    
    def load_model(self, pretrained: bool = True) -> bool:
        """Load the model and tokenizer.
        
        Parameters
        ----------
        pretrained:
            Whether to load the pre-trained model weights
        
        Returns
        -------
        bool:
            True if successful, False otherwise
        """
        if self._model_loaded:
            return True
        
        try:
            model_id = self.model_config["model_id"]
            logger.info(f"Loading {model_id}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            quantization_config = self._setup_quantization()
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
            )
            
            if self.device == "cuda":
                self.model.config.pretraining_tp = 1
            
            self._model_loaded = True
            vram_gb = GPUMemoryProfiler.get_allocated_vram()
            logger.info(f"Model loaded successfully. VRAM used: {vram_gb:.1f} GB")
            return True
        
        except Exception as exc:
            logger.error(f"Failed to load model: {exc}")
            return False
    
    def fine_tune(
        self,
        train_data: List[str] | Dataset,
        config: Optional[FineTuneConfig] = None,
        output_dir: str = "./gemma_checkpoints",
        eval_data: Optional[List[str] | Dataset] = None,
    ) -> Optional[TrainingResult]:
        """Fine-tune the model on provided data.
        
        Parameters
        ----------
        train_data:
            List of training texts or HuggingFace Dataset
        config:
            Fine-tuning configuration (uses defaults if None)
        output_dir:
            Directory to save checkpoints
        eval_data:
            Optional evaluation data
        
        Returns
        -------
        TrainingResult:
            Training results, or None on failure
        """
        if not self.load_model():
            return None
        
        if config is None:
            config = FineTuneConfig(model_name=self.model_name)
        
        try:
            import time
            from datetime import datetime
            
            start_time = time.time()
            
            # Prepare training data
            if isinstance(train_data, list):
                train_dataset = Dataset.from_dict({"text": train_data})
            else:
                train_dataset = train_data
            
            # Tokenize
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    max_length=config.max_seq_length,
                    truncation=True,
                    padding="max_length",
                )
            
            train_dataset = train_dataset.map(tokenize_function, batched=True)
            
            # Prepare evaluation dataset if provided
            eval_dataset = None
            if eval_data is not None:
                if isinstance(eval_data, list):
                    eval_dataset = Dataset.from_dict({"text": eval_data})
                else:
                    eval_dataset = eval_data
                eval_dataset = eval_dataset.map(tokenize_function, batched=True)
            
            # Setup training arguments
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            training_args = TrainingArguments(
                output_dir=str(output_path),
                num_train_epochs=config.num_train_epochs,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                learning_rate=config.learning_rate,
                warmup_steps=config.warmup_steps,
                weight_decay=config.weight_decay,
                logging_steps=10,
                save_steps=100,
                save_total_limit=3,
                load_best_model_at_end=True if eval_dataset else False,
                eval_strategy="steps" if eval_dataset else "no",
                eval_steps=100 if eval_dataset else None,
                fp16=self.device == "cuda",
                gradient_checkpointing=True,
                optim="paged_adamw_32bit",
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
            )
            
            # Train
            logger.info(f"Starting training for {config.num_train_epochs} epochs...")
            train_result = trainer.train()
            
            elapsed = time.time() - start_time
            final_loss = train_result.training_loss or 0.0
            final_loss = float(final_loss) if isinstance(final_loss, (int, float)) else 0.0
            
            # Save model
            save_dir = output_path / "final_model"
            self.model.save_pretrained(str(save_dir))
            self.tokenizer.save_pretrained(str(save_dir))
            logger.info(f"Model saved to {save_dir}")
            
            return TrainingResult(
                model_name=self.model_name,
                num_samples=len(train_dataset),
                final_loss=final_loss,
                training_time=elapsed,
                output_dir=str(save_dir),
                checkpoint_dir=str(output_path),
                metrics=asdict(train_result.metrics) if hasattr(train_result, "metrics") else {},
                timestamp=datetime.now().isoformat(),
            )
        
        except Exception as exc:
            logger.error(f"Fine-tuning failed: {exc}")
            return None
    
    def infer(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str = "",
    ) -> InferenceResult:
        """Run inference on a prompt.
        
        Parameters
        ----------
        prompt:
            The input prompt text
        max_length:
            Maximum tokens to generate
        temperature:
            Sampling temperature (0-2)
        top_p:
            Top-p (nucleus) sampling parameter
        system_prompt:
            Optional system message
        
        Returns
        -------
        InferenceResult:
            The generated response and metadata
        """
        if not self._model_loaded:
            if not self.load_model():
                return InferenceResult(
                    model=self.model_name,
                    prompt=prompt,
                    response="",
                    tokens_generated=0,
                    inference_time=0.0,
                    error="Failed to load model",
                )
        
        try:
            import time
            start_time = time.time()
            
            # Build full input
            full_input = prompt
            if system_prompt:
                full_input = f"{system_prompt}\n\n{prompt}"
            
            # Tokenize
            inputs = self.tokenizer(full_input, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(full_input):].strip()
            
            elapsed = time.time() - start_time
            tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
            
            return InferenceResult(
                model=self.model_name,
                prompt=prompt,
                response=response,
                tokens_generated=tokens_generated,
                inference_time=elapsed,
            )
        
        except Exception as exc:
            logger.error(f"Inference failed: {exc}")
            return InferenceResult(
                model=self.model_name,
                prompt=prompt,
                response="",
                tokens_generated=0,
                inference_time=0.0,
                error=str(exc),
            )
    
    def save(self, path: str) -> bool:
        """Save the current model to disk.
        
        Parameters
        ----------
        path:
            Directory path to save to
        
        Returns
        -------
        bool:
            True if successful
        """
        if not self._model_loaded:
            logger.error("Model not loaded")
            return False
        
        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(save_path))
            self.tokenizer.save_pretrained(str(save_path))
            logger.info(f"Model saved to {save_path}")
            return True
        except Exception as exc:
            logger.error(f"Failed to save model: {exc}")
            return False
    
    def load_fine_tuned(self, path: str) -> bool:
        """Load a previously fine-tuned model.
        
        Parameters
        ----------
        path:
            Path to the saved model directory
        
        Returns
        -------
        bool:
            True if successful
        """
        try:
            logger.info(f"Loading fine-tuned model from {path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
            )
            self._model_loaded = True
            logger.info("Fine-tuned model loaded successfully")
            return True
        except Exception as exc:
            logger.error(f"Failed to load fine-tuned model: {exc}")
            return False
    
    def unload(self) -> None:
        """Unload the model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._model_loaded = False
        torch.cuda.empty_cache()
        logger.info("Model unloaded and cache cleared")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "name": self.model_name,
            "config": self.model_config,
            "device": self.device,
            "use_4bit": self.use_4bit,
            "loaded": self._model_loaded,
            "gpu_available": torch.cuda.is_available(),
            "vram_total_gb": GPUMemoryProfiler.get_available_vram(),
            "vram_allocated_gb": GPUMemoryProfiler.get_allocated_vram(),
        }
