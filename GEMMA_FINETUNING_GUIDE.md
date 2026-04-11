# Gemma 4 Fine-Tuning with Unsloth

This document explains how to use the integrated Gemma 4 fine-tuning engine in AI-Helper.

## Overview

The `gemma_finetuner` module provides efficient fine-tuning and inference for Google's Gemma models using **Unsloth**. Unsloth optimizes model training to be **1.5x faster** and use **60% less VRAM** compared to standard implementations.

### Key Features

- ✅ **Memory Efficient**: Fine-tune Gemma models on just 8GB VRAM
- ✅ **Fast Training**: Unsloth kernels accelerate training by 1.5x
- ✅ **4-bit Quantization**: Reduce memory footprint with minimal quality loss
- ✅ **LoRA Adapters**: Parameter-efficient fine-tuning
- ✅ **Multiple Model Sizes**: 2B, 9B, and 27B variants
- ✅ **Easy Inference**: Simple API for running inference on fine-tuned models

## Supported Models

| Model | Parameters | VRAM Required | Speed |
|-------|-----------|---------------|-------|
| `gemma-2-2b` | 2 Billion | 2 GB (minimum) | ⚡⚡⚡ Fastest |
| `gemma-2-9b` | 9 Billion | 8 GB (optimal) | ⚡⚡ Fast |
| `gemma-2-27b` | 27 Billion | 20 GB | ⚡ Slower |
| `gemma-7b` | 7 Billion | 8 GB | ⚡⚡ Fast (legacy) |

## Installation

### 1. Install Dependencies

The required dependencies have been added to `requirements.txt`:

```bash
pip install -r requirements.txt
```

This includes:
- `torch` - Deep learning framework
- `transformers` - Model loading and training
- `unsloth` - Memory-efficient training kernels
- `peft` - LoRA adapter implementation
- `trl` - Trainer wrapper
- `datasets` - Data handling

### 2. Verify CUDA (Optional but Recommended)

For GPU acceleration:

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

## Quick Start

### Example 1: Check GPU and Select Model

```python
from ai_helper.gemma_finetuner import GPUMemoryProfiler

# Check available VRAM
vram_gb = GPUMemoryProfiler.get_available_vram()
print(f"Available VRAM: {vram_gb:.1f} GB")

# Get list of models that fit
available = GPUMemoryProfiler.get_available_models()
print(f"Available models: {available}")
```

### Example 2: Run Inference Only

```python
from ai_helper.gemma_finetuner import GemmaFineTuner

# Initialize and load the model
tuner = GemmaFineTuner(model_name="gemma-2-2b")

# Run inference
result = tuner.infer(
    prompt="What is machine learning?",
    max_length=100,
    temperature=0.7,
)

print(f"Response: {result.response}")
print(f"Tokens generated: {result.tokens_generated}")
print(f"Time: {result.inference_time:.2f}s")

# Clean up
tuner.unload()
```

### Example 3: Fine-Tune on Custom Data

```python
from ai_helper.gemma_finetuner import GemmaFineTuner, FineTuneConfig

# Prepare your training data
training_texts = [
    "Your training sentence 1",
    "Your training sentence 2",
    # ... more sentences
]

# Configure fine-tuning
config = FineTuneConfig(
    model_name="gemma-2-2b",
    num_train_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
)

# Fine-tune
tuner = GemmaFineTuner(model_name="gemma-2-2b")
result = tuner.fine_tune(
    train_data=training_texts,
    config=config,
    output_dir="./my_gemma_model",
)

print(result)
```

### Example 4: Load and Use Fine-Tuned Model

```python
from ai_helper.gemma_finetuner import GemmaFineTuner

# Load the fine-tuned model
tuner = GemmaFineTuner(model_name="gemma-2-2b")
tuner.load_fine_tuned("./my_gemma_model/final_model")

# Run inference with the fine-tuned model
result = tuner.infer(
    prompt="Your prompt here",
    max_length=150,
)

print(result.response)
tuner.unload()
```

## Configuration Options

### FineTuneConfig Parameters

```python
config = FineTuneConfig(
    model_name="gemma-2-2b",           # Model to fine-tune
    learning_rate=2e-4,                # Learning rate
    num_train_epochs=3,                # Number of training epochs
    batch_size=4,                      # Batch size
    gradient_accumulation_steps=1,     # Gradient accumulation
    max_seq_length=512,                # Max sequence length
    lora_r=16,                         # LoRA rank
    lora_alpha=32,                     # LoRA alpha
    lora_dropout=0.05,                 # LoRA dropout
    warmup_steps=100,                  # Warmup steps
    weight_decay=0.01,                 # Weight decay
)
```

### Inference Parameters

```python
result = tuner.infer(
    prompt="Your prompt",
    max_length=256,                    # Max tokens to generate
    temperature=0.7,                   # Sampling temperature (0-2)
    top_p=0.9,                         # Top-p sampling
    system_prompt="Optional system message",
)
```

## Memory Optimization Tips

### For 8GB Systems

```python
config = FineTuneConfig(
    model_name="gemma-2-2b",  # Start with 2B
    batch_size=2,             # Set to 2
    max_seq_length=256,       # Reduce sequence length
    gradient_accumulation_steps=2,  # Accumulate gradients
)
```

### For 16GB+ Systems

```python
config = FineTuneConfig(
    model_name="gemma-2-9b",  # Use 9B model
    batch_size=8,
    max_seq_length=512,
    gradient_accumulation_steps=1,
)
```

## Data Format

### List of Strings

```python
train_data = [
    "First training example",
    "Second training example",
    "Third training example",
]

tuner.fine_tune(train_data=train_data, ...)
```

### HuggingFace Dataset

```python
from datasets import Dataset

data = {
    "text": [
        "Example 1",
        "Example 2",
        "Example 3",
    ]
}

dataset = Dataset.from_dict(data)
tuner.fine_tune(train_data=dataset, ...)
```

## Performance Tips

### 1. Batch Processing

```python
prompts = ["prompt1", "prompt2", "prompt3", ...]
results = []

for prompt in prompts:
    result = tuner.infer(prompt, max_length=100)
    results.append(result)

tuner.unload()  # Clean up after all inferences
```

### 2. System Prompts

```python
system = "You are an expert in machine learning."

result = tuner.infer(
    prompt="Explain neural networks",
    system_prompt=system,
)
```

### 3. Temperature Control

- **Lower temperature (0.1-0.3)**: More deterministic, focused
- **Medium temperature (0.5-0.7)**: Balanced
- **Higher temperature (0.9-1.5)**: More creative, diverse

## Troubleshooting

### Out of Memory (OOM) Error

1. Reduce `batch_size` in config
2. Reduce `max_seq_length`
3. Switch to smaller model (2B instead of 9B)
4. Increase `gradient_accumulation_steps`

### Model Loading Fails

```python
# Check if you have the right permissions
# Google Gemma models require acceptance of license
# Visit: https://huggingface.co/google/gemma-2-2b

# Use HuggingFace token if needed
from huggingface_hub import login
login(token="your_hf_token")
```

### Slow Inference

- Use smaller `max_length` for faster generation
- Reduce model size (2B is faster than 27B)
- Enable GPU inference (check with `torch.cuda.is_available()`)

## Integration with AI-Helper

The Gemma fine-tuner integrates with AI-Helper's orchestrator:

```python
from ai_helper.orchestrator import Orchestrator
from ai_helper.gemma_finetuner import GemmaFineTuner

# Can be used alongside other AI integrations
orchestrator = Orchestrator()
tuner = GemmaFineTuner(model_name="gemma-2-2b")

# Run your AI-Helper tasks with custom fine-tuned model
```

## Testing

Run the test suite:

```bash
python -m pytest tests/test_gemma_finetuner.py -v
```

## Examples

Full examples are available in `examples_gemma_finetuner.py`:

```bash
python examples_gemma_finetuner.py
```

## References

- [Google Gemma Models](https://ai.google.dev/gemma/)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [PEFT - Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)

## License

Gemma models are licensed under the [Gemma License Agreement](https://ai.google.dev/gemma/terms).

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review example in `examples_gemma_finetuner.py`
3. Check test file for usage patterns: `tests/test_gemma_finetuner.py`
