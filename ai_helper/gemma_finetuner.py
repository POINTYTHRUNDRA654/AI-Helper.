"""Local LLM fine-tuning engine powered by Unsloth.

Provides efficient fine-tuning and inference for a wide range of free,
open-source models using Unsloth's optimised kernels.  Works on systems
with as little as 4 GB VRAM via 4-bit quantisation, LoRA adapters, and
gradient checkpointing.

Supported model families
------------------------
- **Gemma 4** (2B / 9B / 27B) — Google's latest; best free quality
- **Gemma 2** (2B / 9B / 27B) — previous gen, very stable
- **Phi-4 / Phi-4-mini** — Microsoft; outstanding reasoning per VRAM
- **LLaMA 3 / 3.1 / 3.2** — Meta; versatile, well-supported
- **Mistral 7B / Nemo 12B** — fast, efficient instruction models
- **Qwen 2.5 / Qwen 2.5-Coder** — strong coding & reasoning

All models are free and downloaded from HuggingFace on first use.

Memory optimisation techniques
-------------------------------
- 4-bit quantisation via ``bitsandbytes``
- LoRA adapters for parameter-efficient fine-tuning
- Gradient checkpointing
- Flash Attention 2 kernels (Unsloth ``unsloth_id`` variants)
- Automatic VRAM profiling and model selection

Quick-start: fine-tune Mossy on modding conversations
------------------------------------------------------
::

    from ai_helper.gemma_finetuner import UnslothFineTuner, prepare_mossy_dataset

    # Prepare training data from Q&A pairs
    pairs = [
        ("How many polygons for a Fallout 4 weapon?",
         "A standard weapon should stay under 5,000 triangles ..."),
        ("What NIF block type should I use for a static prop?",
         "Use BSTriShape as the geometry block and BSFadeNode as the root ..."),
    ]
    dataset = prepare_mossy_dataset(pairs)

    # Fine-tune (uses Unsloth for 2x speed + 60% less VRAM)
    tuner = UnslothFineTuner(model_name="gemma-4-9b")
    result = tuner.fine_tune(dataset, output_dir="./mossy_checkpoints")
    print(result)
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
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

GEMMA_MODELS = {
    # ── Gemma 4 (latest, 2025) ──────────────────────────────────────────────
    "gemma-4-2b": {
        "model_id": "google/gemma-4-2b",
        "vram_gb": 3,
        "description": "Gemma 4 2B — ultra-lightweight latest gen, ~3 GB VRAM",
        "unsloth_id": "unsloth/gemma-4-2b-unsloth-bnb-4bit",
    },
    "gemma-4-9b": {
        "model_id": "google/gemma-4-9b",
        "vram_gb": 8,
        "description": "Gemma 4 9B — best free quality for 8 GB VRAM (recommended)",
        "unsloth_id": "unsloth/gemma-4-9b-unsloth-bnb-4bit",
    },
    "gemma-4-27b": {
        "model_id": "google/gemma-4-27b",
        "vram_gb": 20,
        "description": "Gemma 4 27B — high-performance, 20 GB+ VRAM",
        "unsloth_id": "unsloth/gemma-4-27b-unsloth-bnb-4bit",
    },
    # ── Gemma 2 (previous gen, very stable) ─────────────────────────────────
    "gemma-2-2b": {
        "model_id": "google/gemma-2-2b",
        "vram_gb": 2,
        "description": "Gemma 2 2B — ultra-lightweight, ~2 GB VRAM",
        "unsloth_id": "unsloth/gemma-2-2b-unsloth-bnb-4bit",
    },
    "gemma-2-9b": {
        "model_id": "google/gemma-2-9b",
        "vram_gb": 8,
        "description": "Gemma 2 9B — optimal for 8 GB VRAM systems",
        "unsloth_id": "unsloth/gemma-2-9b-unsloth-bnb-4bit",
    },
    "gemma-2-27b": {
        "model_id": "google/gemma-2-27b",
        "vram_gb": 20,
        "description": "Gemma 2 27B — high-performance, requires 20 GB+ VRAM",
        "unsloth_id": "unsloth/gemma-2-27b-unsloth-bnb-4bit",
    },
    # ── Microsoft Phi-4 (excellent reasoning, very small) ───────────────────
    "phi-4": {
        "model_id": "microsoft/phi-4",
        "vram_gb": 8,
        "description": "Microsoft Phi-4 14B — state-of-the-art small model, strong reasoning",
        "unsloth_id": "unsloth/phi-4-unsloth-bnb-4bit",
    },
    "phi-4-mini": {
        "model_id": "microsoft/Phi-4-mini-instruct",
        "vram_gb": 4,
        "description": "Microsoft Phi-4 Mini 3.8B — runs on 4 GB VRAM, fast inference",
        "unsloth_id": "unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit",
    },
    # ── Meta LLaMA 3.x ──────────────────────────────────────────────────────
    "llama-3-8b": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "vram_gb": 8,
        "description": "LLaMA 3 8B Instruct — strong open-source baseline, 8 GB VRAM",
        "unsloth_id": "unsloth/Meta-Llama-3-8B-Instruct-bnb-4bit",
    },
    "llama-3.1-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "vram_gb": 8,
        "description": "LLaMA 3.1 8B Instruct — improved version with longer context",
        "unsloth_id": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    },
    "llama-3.2-3b": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "vram_gb": 4,
        "description": "LLaMA 3.2 3B Instruct — tiny powerhouse, fits on 4 GB VRAM",
        "unsloth_id": "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
    },
    # ── Mistral ─────────────────────────────────────────────────────────────
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "vram_gb": 6,
        "description": "Mistral 7B v0.3 Instruct — fast, efficient, 6 GB VRAM",
        "unsloth_id": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    },
    "mistral-nemo-12b": {
        "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
        "vram_gb": 12,
        "description": "Mistral Nemo 12B — excellent long-context model, 12 GB VRAM",
        "unsloth_id": "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    },
    # ── Qwen 2.5 (strong coder, good for modding tasks) ─────────────────────
    "qwen2.5-7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "vram_gb": 6,
        "description": "Qwen 2.5 7B Instruct — excellent coding + reasoning, 6 GB VRAM",
        "unsloth_id": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    },
    "qwen2.5-coder-7b": {
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "vram_gb": 6,
        "description": "Qwen 2.5 Coder 7B — best free coding model at 6 GB VRAM",
        "unsloth_id": "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
    },
    # ── Legacy ──────────────────────────────────────────────────────────────
    "gemma-7b": {
        "model_id": "google/gemma-7b",
        "vram_gb": 8,
        "description": "Gemma 7B — legacy model (prefer gemma-4-9b for new projects)",
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
                metrics=dict(train_result.metrics) if hasattr(train_result, "metrics") else {},
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


# ---------------------------------------------------------------------------
# Unsloth-powered fine-tuner (2× faster, 60% less VRAM)
# ---------------------------------------------------------------------------


class UnslothFineTuner:
    """Fine-tune any model in :data:`GEMMA_MODELS` using Unsloth's optimised
    kernels.

    Unsloth provides:

    * **2× faster training** than vanilla HuggingFace on the same hardware.
    * **60% less VRAM** via its custom 4-bit + Flash-Attention-2 kernels.
    * Seamless LoRA and QLoRA adapters.
    * GGUF export for use with llama.cpp / Ollama / LM Studio.

    Requires: ``pip install unsloth[cu121]`` (or cu118, cu122, cu124 for your
    CUDA version).  All weights are downloaded from HuggingFace for free.

    Parameters
    ----------
    model_name:
        Key in :data:`GEMMA_MODELS` (e.g. ``"gemma-4-9b"``).
    max_seq_length:
        Maximum sequence length in tokens.
    load_in_4bit:
        Use 4-bit quantisation (strongly recommended).

    Example
    -------
    ::

        from ai_helper.gemma_finetuner import UnslothFineTuner, prepare_mossy_dataset

        pairs = [("What's the weapon poly budget?", "5,000 tris for a standard weapon.")]
        dataset = prepare_mossy_dataset(pairs)

        tuner = UnslothFineTuner("gemma-4-9b")
        result = tuner.fine_tune(dataset, output_dir="./mossy_checkpoints")
        tuner.save_gguf("./mossy_gguf", quantisation="q4_k_m")  # ready for Ollama
    """

    def __init__(
        self,
        model_name: str = "gemma-4-9b",
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
    ) -> None:
        if model_name not in GEMMA_MODELS:
            raise ValueError(
                f"Unknown model: {model_name!r}. "
                f"Available: {', '.join(GEMMA_MODELS)}"
            )
        self.model_name = model_name
        self.model_config = GEMMA_MODELS[model_name]
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        """Load the model with Unsloth optimisations.

        Falls back to plain HuggingFace loading if Unsloth is not installed.
        Returns ``True`` on success.
        """
        if self._loaded:
            return True
        try:
            from unsloth import FastLanguageModel  # type: ignore[import-untyped]  # noqa: PLC0415
            unsloth_id = self.model_config.get("unsloth_id") or self.model_config["model_id"]
            logger.info("Loading %r with Unsloth (max_seq=%d)…", unsloth_id, self.max_seq_length)
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=unsloth_id,
                max_seq_length=self.max_seq_length,
                load_in_4bit=self.load_in_4bit,
                dtype=None,  # auto-detect
            )
            # Wrap with LoRA adapters
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            self._loaded = True
            logger.info("Unsloth model loaded successfully.")
            return True
        except ImportError:
            logger.warning("Unsloth not installed — falling back to standard HuggingFace loading.")
            # Fall back to GemmaFineTuner
            tuner = GemmaFineTuner(
                model_name=self.model_name,
                use_4bit=self.load_in_4bit,
            )
            if tuner.load_model():
                self.model = tuner.model
                self.tokenizer = tuner.tokenizer
                self._loaded = True
                return True
            return False
        except Exception as exc:
            logger.error("Failed to load model with Unsloth: %s", exc)
            return False

    def fine_tune(
        self,
        train_data: "List[str] | Dataset",
        config: Optional[FineTuneConfig] = None,
        output_dir: str = "./unsloth_checkpoints",
    ) -> Optional[TrainingResult]:
        """Fine-tune using Unsloth + HuggingFace SFTTrainer.

        Parameters
        ----------
        train_data:
            List of formatted text examples or a HuggingFace Dataset with
            a ``"text"`` column.  Use :func:`prepare_mossy_dataset` to build
            one from Q&A conversation pairs.
        config:
            Fine-tuning hyperparameters.  Defaults are tuned for Unsloth.
        output_dir:
            Directory to save checkpoints and the final model.
        """
        if not self.load_model():
            return None

        if config is None:
            config = FineTuneConfig(model_name=self.model_name)

        try:
            import time  # noqa: PLC0415
            from datetime import datetime  # noqa: PLC0415
            from datasets import Dataset  # type: ignore[import-untyped]  # noqa: PLC0415

            start = time.time()

            if isinstance(train_data, list):
                train_dataset = Dataset.from_dict({"text": train_data})
            else:
                train_dataset = train_data

            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # Try SFTTrainer (recommended for instruction tuning)
            try:
                from trl import SFTTrainer  # type: ignore[import-untyped]  # noqa: PLC0415
                training_args = TrainingArguments(
                    output_dir=str(out_path),
                    num_train_epochs=config.num_train_epochs,
                    per_device_train_batch_size=config.batch_size,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                    learning_rate=config.learning_rate,
                    warmup_steps=config.warmup_steps,
                    weight_decay=config.weight_decay,
                    fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
                    bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
                    optim="adamw_8bit",
                    logging_steps=10,
                    save_steps=100,
                    save_total_limit=3,
                    report_to="none",
                )
                trainer = SFTTrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    train_dataset=train_dataset,
                    dataset_text_field="text",
                    max_seq_length=self.max_seq_length,
                    args=training_args,
                )
            except ImportError:
                # Fallback to vanilla Trainer
                training_args = TrainingArguments(
                    output_dir=str(out_path),
                    num_train_epochs=config.num_train_epochs,
                    per_device_train_batch_size=config.batch_size,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                    learning_rate=config.learning_rate,
                    fp16=torch.cuda.is_available(),
                    gradient_checkpointing=True,
                    optim="paged_adamw_32bit",
                    logging_steps=10,
                    save_total_limit=3,
                    report_to="none",
                )
                trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=train_dataset,
                    tokenizer=self.tokenizer,
                )

            logger.info("Starting Unsloth fine-tune: %d examples…", len(train_dataset))
            train_result = trainer.train()
            elapsed = time.time() - start

            final_loss = float(getattr(train_result, "training_loss", 0.0) or 0.0)
            save_dir = out_path / "final_model"
            self.model.save_pretrained(str(save_dir))
            self.tokenizer.save_pretrained(str(save_dir))
            logger.info("Unsloth model saved → %s", save_dir)

            return TrainingResult(
                model_name=self.model_name,
                num_samples=len(train_dataset),
                final_loss=final_loss,
                training_time=elapsed,
                output_dir=str(save_dir),
                checkpoint_dir=str(out_path),
                metrics=dict(train_result.metrics) if hasattr(train_result, "metrics") else {},
                timestamp=datetime.now().isoformat(),
            )

        except Exception as exc:
            logger.error("Unsloth fine-tune failed: %s", exc)
            return None

    def save_gguf(
        self,
        output_dir: str,
        quantisation: str = "q4_k_m",
    ) -> bool:
        """Export the fine-tuned model as GGUF for use with Ollama / LM Studio.

        Requires Unsloth to be installed.

        Parameters
        ----------
        output_dir:
            Directory to write the ``.gguf`` file.
        quantisation:
            GGUF quantisation type.  Common choices:

            * ``"q4_k_m"`` — 4-bit, best balance of quality and size (default)
            * ``"q8_0"`` — 8-bit, higher quality, larger file
            * ``"f16"`` — full float16, largest, best quality
            * ``"q2_k"`` — 2-bit, smallest, lower quality
        """
        if not self._loaded or self.model is None:
            logger.error("Model not loaded; call fine_tune() first.")
            return False
        try:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained_gguf(
                str(out_path),
                self.tokenizer,
                quantization_method=quantisation,
            )
            logger.info("GGUF saved → %s  (quantisation: %s)", out_path, quantisation)
            return True
        except AttributeError:
            logger.error("save_pretrained_gguf is only available with Unsloth installed.")
            return False
        except Exception as exc:
            logger.error("GGUF export failed: %s", exc)
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the loaded model."""
        return {
            "name": self.model_name,
            "config": self.model_config,
            "max_seq_length": self.max_seq_length,
            "load_in_4bit": self.load_in_4bit,
            "loaded": self._loaded,
            "gpu_available": torch.cuda.is_available(),
            "vram_total_gb": GPUMemoryProfiler.get_available_vram(),
            "vram_allocated_gb": GPUMemoryProfiler.get_allocated_vram(),
            "unsloth_id": self.model_config.get("unsloth_id", "N/A"),
        }


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def prepare_mossy_dataset(
    conversations: "List[Tuple[str, str]]",
    system_prompt: str = (
        "You are Mossy, an expert AI assistant specialised in Fallout 4 modding, "
        "3D mesh creation, NIF format, Blender, and game asset pipelines.  "
        "You give accurate, detailed, actionable answers."
    ),
    format: str = "chatml",
) -> "Dataset":
    """Convert a list of (question, answer) pairs into a fine-tuning dataset.

    The resulting dataset has a single ``"text"`` column containing each
    conversation formatted in the chosen chat template.

    Parameters
    ----------
    conversations:
        List of ``(question, answer)`` tuples.  Any topic works — mesh
        questions, Python code, general help, etc.
    system_prompt:
        System message prepended to every conversation.
    format:
        Chat format to use:

        * ``"chatml"`` *(default)* — ``<|im_start|>`` tags; compatible with
          most Unsloth and GGUF models.
        * ``"llama3"`` — ``<|begin_of_text|>`` / ``<|eot_id|>`` tags for
          LLaMA 3.x models.
        * ``"gemma"`` — ``<start_of_turn>`` / ``<end_of_turn>`` tags for
          Gemma models.

    Returns
    -------
    datasets.Dataset
        A HuggingFace Dataset with a ``"text"`` column ready for
        :class:`UnslothFineTuner` or any HuggingFace ``Trainer``.

    Example
    -------
    ::

        pairs = [
            ("What is the polygon budget for a Fallout 4 weapon?",
             "A standard weapon should stay under 5,000 triangles for LOD0. "
             "High-detail hero weapons may go up to 10,000."),
            ("Which NIF block type should I use for a static prop?",
             "Use BSTriShape for the geometry and BSFadeNode as the root node. "
             "Attach BSLightingShaderProperty for PBR-like shading."),
        ]
        dataset = prepare_mossy_dataset(pairs)
    """
    from datasets import Dataset as HFDataset  # type: ignore[import-untyped]  # noqa: PLC0415

    texts: List[str] = []

    for question, answer in conversations:
        if format == "chatml":
            text = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n{answer}<|im_end|>"
            )
        elif format == "llama3":
            text = (
                "<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n{answer}<|eot_id|>"
            )
        elif format == "gemma":
            text = (
                f"<start_of_turn>user\n{system_prompt}\n\n{question}<end_of_turn>\n"
                f"<start_of_turn>model\n{answer}<end_of_turn>"
            )
        else:
            raise ValueError(
                f"Unknown format {format!r}. "
                "Choose 'chatml', 'llama3', or 'gemma'."
            )
        texts.append(text)

    return HFDataset.from_dict({"text": texts})


def list_recommended_models(available_vram_gb: Optional[float] = None) -> str:
    """Return a formatted table of all models, sorted by VRAM requirement.

    Parameters
    ----------
    available_vram_gb:
        If supplied, only show models that fit in this VRAM budget.
    """
    rows = sorted(GEMMA_MODELS.items(), key=lambda kv: kv[1]["vram_gb"])
    lines = [
        "Available free local models for fine-tuning / inference:",
        "",
        f"  {'Key':<20} {'VRAM':>5}  Description",
        "  " + "-" * 80,
    ]
    for key, cfg in rows:
        vram = cfg["vram_gb"]
        if available_vram_gb is not None and vram > available_vram_gb:
            continue
        unsloth = "⚡ Unsloth" if "unsloth_id" in cfg else "          "
        lines.append(f"  {key:<20} {vram:>4}GB  {cfg['description']}  {unsloth}")
    lines += [
        "",
        "⚡ = Unsloth-optimised variant available (2× faster, 60% less VRAM).",
        "    Install Unsloth: pip install 'unsloth[cu121]'",
        "    All models are free and downloaded from HuggingFace.",
    ]
    return "\n".join(lines)

