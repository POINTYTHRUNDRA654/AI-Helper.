"""Example usage of Gemma fine-tuning module.

This script demonstrates how to use the GemmaFineTuner class to fine-tune
and run inference with Google's Gemma models using Unsloth.
"""

import logging
from ai_helper.gemma_finetuner import (
    GemmaFineTuner,
    FineTuneConfig,
    GPUMemoryProfiler,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_check_gpu():
    """Example: Check GPU availability and select appropriate model."""
    logger.info("=== GPU Health Check ===")
    vram_gb = GPUMemoryProfiler.get_available_vram()
    logger.info(f"Total VRAM: {vram_gb:.1f} GB")
    
    available_models = GPUMemoryProfiler.get_available_models()
    logger.info(f"Available models: {available_models}")
    
    if vram_gb >= 20:
        suggested = "gemma-2-27b"
    elif vram_gb >= 8:
        suggested = "gemma-2-9b"
    else:
        suggested = "gemma-2-2b"
    
    logger.info(f"Suggested model for your system: {suggested}")


def example_inference_only():
    """Example: Load and run inference (no fine-tuning)."""
    logger.info("\n=== Inference Only Example ===")
    
    # Initialize the fine-tuner (loads model on first use)
    tuner = GemmaFineTuner(model_name="gemma-2-2b", device="cuda")
    
    logger.info(f"Model info: {tuner.get_model_info()}")
    
    # Run inference
    prompts = [
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence.",
    ]
    
    for prompt in prompts:
        result = tuner.infer(
            prompt=prompt,
            max_length=100,
            temperature=0.7,
        )
        logger.info(f"\nPrompt: {result.prompt}")
        logger.info(f"Response: {result.response}")
        logger.info(f"Tokens: {result.tokens_generated}, Time: {result.inference_time:.2f}s")
    
    # Clean up
    tuner.unload()


def example_fine_tune():
    """Example: Fine-tune a model on custom data."""
    logger.info("\n=== Fine-Tuning Example ===")
    
    # Sample training data
    training_texts = [
        "AI is transforming the world of technology and business.",
        "Machine learning models can learn from data without explicit programming.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret visual information.",
        "Reinforcement learning teaches agents to make decisions through rewards.",
        "Transfer learning helps models leverage knowledge from one task to another.",
        "Data augmentation improves model robustness by creating variations.",
    ] * 10  # Repeat for more samples
    
    # Configure fine-tuning
    config = FineTuneConfig(
        model_name="gemma-2-2b",
        learning_rate=2e-4,
        num_train_epochs=2,
        batch_size=4,
        max_seq_length=256,
    )
    
    # Initialize tuner
    tuner = GemmaFineTuner(model_name="gemma-2-2b")
    
    # Fine-tune
    result = tuner.fine_tune(
        train_data=training_texts,
        config=config,
        output_dir="./my_gemma_model",
    )
    
    if result:
        logger.info(f"Fine-tuning completed: {result}")
    else:
        logger.error("Fine-tuning failed")
        return
    
    # Test the fine-tuned model
    logger.info("\n=== Testing Fine-Tuned Model ===")
    if tuner.load_fine_tuned(result.output_dir):
        test_prompt = "What are the benefits of machine learning?"
        result = tuner.infer(
            prompt=test_prompt,
            max_length=150,
        )
        logger.info(f"Fine-tuned model response:\n{result.response}")
    
    tuner.unload()


def example_system_prompt():
    """Example: Use system prompt with inference."""
    logger.info("\n=== System Prompt Example ===")
    
    tuner = GemmaFineTuner(model_name="gemma-2-2b")
    
    result = tuner.infer(
        prompt="What is your expertise?",
        system_prompt="You are an expert AI assistant specialized in machine learning and data science.",
        max_length=100,
    )
    
    logger.info(f"Response with system prompt:\n{result.response}")
    tuner.unload()


def example_batch_inference():
    """Example: Run inference on multiple prompts efficiently."""
    logger.info("\n=== Batch Inference Example ===")
    
    tuner = GemmaFineTuner(model_name="gemma-2-2b")
    
    prompts = [
        "List five benefits of AI.",
        "Explain neural networks.",
        "What is deep learning?",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        result = tuner.infer(
            prompt=prompt,
            max_length=100,
            temperature=0.5,
        )
        logger.info(f"\n[{i}] {prompt}")
        logger.info(f"    {result.response[:100]}...")
    
    tuner.unload()


if __name__ == "__main__":
    # Run examples
    print("Choose an example to run:")
    print("1. Check GPU and select model")
    print("2. Inference only (no fine-tuning)")
    print("3. Fine-tune on custom data")
    print("4. System prompt with inference")
    print("5. Batch inference")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    try:
        if choice == "1":
            example_check_gpu()
        elif choice == "2":
            example_inference_only()
        elif choice == "3":
            example_fine_tune()
        elif choice == "4":
            example_system_prompt()
        elif choice == "5":
            example_batch_inference()
        else:
            print("Invalid choice")
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
