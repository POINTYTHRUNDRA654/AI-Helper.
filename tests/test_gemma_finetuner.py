"""Tests for Gemma fine-tuning engine."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from ai_helper.gemma_finetuner import (
    FineTuneConfig,
    GEMMA_MODELS,
    GemmaFineTuner,
    GPUMemoryProfiler,
    InferenceResult,
    TrainingResult,
)


class TestGPUMemoryProfiler(unittest.TestCase):
    """Test GPU memory profiling utilities."""
    
    def test_get_available_models_no_gpu(self):
        """Test getting available models with no GPU."""
        with patch("torch.cuda.is_available", return_value=False):
            models = GPUMemoryProfiler.get_available_models(available_vram=16.0)
            # Should return empty list or basic models when GPU not checked properly
            self.assertIsInstance(models, list)
    
    def test_get_available_models_with_8gb(self):
        """Test that models fitting in 8GB VRAM are returned."""
        models = GPUMemoryProfiler.get_available_models(available_vram=8.0)
        self.assertIn("gemma-2-2b", models)
        self.assertIsInstance(models, list)
    
    def test_get_available_models_with_2gb(self):
        """Test limited models with 2GB VRAM."""
        models = GPUMemoryProfiler.get_available_models(available_vram=2.0)
        self.assertIn("gemma-2-2b", models)
    
    def test_get_available_models_insufficient_vram(self):
        """Test with insufficient VRAM."""
        models = GPUMemoryProfiler.get_available_models(available_vram=1.0)
        self.assertEqual(len(models), 0)


class TestFineTuneConfig(unittest.TestCase):
    """Test fine-tuning configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FineTuneConfig()
        self.assertEqual(config.model_name, "gemma-2-2b")
        self.assertEqual(config.num_train_epochs, 3)
        self.assertEqual(config.batch_size, 4)
        self.assertGreater(config.learning_rate, 0)
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = FineTuneConfig(num_train_epochs=5)
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["num_train_epochs"], 5)
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {"model_name": "gemma-2-9b", "num_train_epochs": 10}
        config = FineTuneConfig.from_dict(data)
        self.assertEqual(config.model_name, "gemma-2-9b")
        self.assertEqual(config.num_train_epochs, 10)


class TestGemmaFineTunerInit(unittest.TestCase):
    """Test GemmaFineTuner initialization."""
    
    def test_init_valid_model(self):
        """Test initialization with valid model name."""
        tuner = GemmaFineTuner(model_name="gemma-2-2b")
        self.assertEqual(tuner.model_name, "gemma-2-2b")
        self.assertFalse(tuner._model_loaded)
    
    def test_init_invalid_model(self):
        """Test initialization with invalid model name."""
        with self.assertRaises(ValueError) as context:
            GemmaFineTuner(model_name="invalid-model")
        self.assertIn("Unknown model", str(context.exception))
    
    def test_init_device_auto_cuda(self):
        """Test device auto-detection with CUDA."""
        with patch("torch.cuda.is_available", return_value=True):
            tuner = GemmaFineTuner(device="auto")
            self.assertEqual(tuner.device, "cuda")
    
    def test_init_device_auto_cpu(self):
        """Test device auto-detection with CPU only."""
        with patch("torch.cuda.is_available", return_value=False):
            tuner = GemmaFineTuner(device="auto")
            self.assertEqual(tuner.device, "cpu")
    
    def test_init_device_explicit(self):
        """Test explicit device specification."""
        tuner = GemmaFineTuner(device="cpu")
        self.assertEqual(tuner.device, "cpu")
    
    def test_get_model_info(self):
        """Test getting model information."""
        tuner = GemmaFineTuner(model_name="gemma-2-2b")
        info = tuner.get_model_info()
        self.assertIn("name", info)
        self.assertIn("device", info)
        self.assertIn("loaded", info)
        self.assertEqual(info["name"], "gemma-2-2b")
        self.assertFalse(info["loaded"])


class TestTrainingResult(unittest.TestCase):
    """Test training result data class."""
    
    def test_training_result_creation(self):
        """Test creating a training result."""
        result = TrainingResult(
            model_name="gemma-2-2b",
            num_samples=100,
            final_loss=1.5,
            training_time=3600.0,
            output_dir="/path/to/model",
        )
        self.assertEqual(result.model_name, "gemma-2-2b")
        self.assertEqual(result.num_samples, 100)
        self.assertEqual(result.final_loss, 1.5)
    
    def test_training_result_str(self):
        """Test string representation of training result."""
        result = TrainingResult(
            model_name="gemma-2-2b",
            num_samples=50,
            final_loss=2.0,
            training_time=1000.0,
            output_dir="/path/to/model",
        )
        result_str = str(result)
        self.assertIn("gemma-2-2b", result_str)
        self.assertIn("50", result_str)


class TestInferenceResult(unittest.TestCase):
    """Test inference result data class."""
    
    def test_inference_result_success(self):
        """Test creating a successful inference result."""
        result = InferenceResult(
            model="gemma-2-2b",
            prompt="Hello",
            response="Hello, world!",
            tokens_generated=3,
            inference_time=0.5,
        )
        self.assertEqual(result.model, "gemma-2-2b")
        self.assertEqual(result.response, "Hello, world!")
        self.assertEqual(result.tokens_generated, 3)
        self.assertEqual(result.error, "")
    
    def test_inference_result_error(self):
        """Test inference result with error."""
        result = InferenceResult(
            model="gemma-2-2b",
            prompt="Hello",
            response="",
            tokens_generated=0,
            inference_time=0.0,
            error="Model not loaded",
        )
        self.assertEqual(result.error, "Model not loaded")
        result_str = str(result)
        self.assertIn("ERROR", result_str)


class TestModelConfiguration(unittest.TestCase):
    """Test model configuration constants."""
    
    def test_gemma_models_defined(self):
        """Test that Gemma models are properly defined."""
        self.assertGreater(len(GEMMA_MODELS), 0)
        for model_name, config in GEMMA_MODELS.items():
            self.assertIn("model_id", config)
            self.assertIn("vram_gb", config)
            self.assertIn("description", config)
    
    def test_all_models_have_reasonable_vram(self):
        """Test that all models have reasonable VRAM requirements."""
        for model_name, config in GEMMA_MODELS.items():
            vram = config["vram_gb"]
            self.assertGreater(vram, 0)
            self.assertLess(vram, 100)  # Sanity check


if __name__ == "__main__":
    unittest.main()
