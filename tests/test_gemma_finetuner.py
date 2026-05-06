"""Tests for the fine-tuning engine (GemmaFineTuner, UnslothFineTuner, helpers)."""

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
    UnslothFineTuner,
    list_recommended_models,
    prepare_mossy_dataset,
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
        """GEMMA_MODELS contains entries with required keys."""
        self.assertGreater(len(GEMMA_MODELS), 0)
        for model_name, config in GEMMA_MODELS.items():
            self.assertIn("model_id", config, model_name)
            self.assertIn("vram_gb", config, model_name)
            self.assertIn("description", config, model_name)

    def test_all_models_have_reasonable_vram(self):
        """All models report a sane VRAM requirement."""
        for model_name, config in GEMMA_MODELS.items():
            vram = config["vram_gb"]
            self.assertGreater(vram, 0, model_name)
            self.assertLess(vram, 100, model_name)

    def test_new_model_families_present(self):
        """Gemma 4, Phi-4, LLaMA-3, Mistral, and Qwen models are all registered."""
        keys = set(GEMMA_MODELS.keys())
        self.assertTrue(any("gemma-4" in k for k in keys), "No Gemma 4 entries")
        self.assertTrue(any("phi" in k for k in keys), "No Phi entries")
        self.assertTrue(any("llama" in k.lower() for k in keys), "No LLaMA entries")
        self.assertTrue(any("mistral" in k.lower() for k in keys), "No Mistral entries")
        self.assertTrue(any("qwen" in k.lower() for k in keys), "No Qwen entries")

    def test_unsloth_ids_are_strings(self):
        """Models with an unsloth_id store it as a non-empty string."""
        for model_name, config in GEMMA_MODELS.items():
            if "unsloth_id" in config:
                self.assertIsInstance(config["unsloth_id"], str, model_name)
                self.assertGreater(len(config["unsloth_id"]), 0, model_name)


class TestUnslothFineTuner(unittest.TestCase):
    """Tests for UnslothFineTuner initialisation and metadata."""

    def test_init_valid_model(self):
        """UnslothFineTuner initialises cleanly for any registered model."""
        tuner = UnslothFineTuner(model_name="gemma-4-9b")
        self.assertEqual(tuner.model_name, "gemma-4-9b")
        self.assertFalse(tuner._loaded)

    def test_init_invalid_model(self):
        """Unknown model name raises ValueError."""
        with self.assertRaises(ValueError):
            UnslothFineTuner(model_name="nonexistent-model-xyz")

    def test_init_defaults(self):
        """Default constructor uses gemma-4-9b with 4-bit quantisation."""
        tuner = UnslothFineTuner()
        self.assertEqual(tuner.model_name, "gemma-4-9b")
        self.assertTrue(tuner.load_in_4bit)
        self.assertEqual(tuner.max_seq_length, 2048)

    def test_get_model_info_not_loaded(self):
        """get_model_info works before the model is loaded."""
        tuner = UnslothFineTuner(model_name="phi-4-mini")
        info = tuner.get_model_info()
        self.assertEqual(info["name"], "phi-4-mini")
        self.assertFalse(info["loaded"])
        self.assertIn("unsloth_id", info)

    def test_all_registered_models_instantiate(self):
        """Every key in GEMMA_MODELS can be used to construct UnslothFineTuner."""
        for key in GEMMA_MODELS:
            tuner = UnslothFineTuner(model_name=key)
            self.assertEqual(tuner.model_name, key)

    def test_save_gguf_without_loaded_model(self):
        """save_gguf returns False when model is not loaded."""
        tuner = UnslothFineTuner(model_name="gemma-4-9b")
        result = tuner.save_gguf("/tmp/test_gguf")
        self.assertFalse(result)


class TestPrepareMossyDataset(unittest.TestCase):
    """Tests for prepare_mossy_dataset helper."""

    def _pairs(self):
        return [
            ("What is the weapon poly budget?",
             "A standard weapon should stay under 5,000 triangles."),
            ("What NIF block is used for static props?",
             "BSTriShape is the geometry block; BSFadeNode is the root."),
        ]

    def test_chatml_format(self):
        """chatml format produces <|im_start|> tagged text."""
        ds = prepare_mossy_dataset(self._pairs(), format="chatml")
        self.assertEqual(len(ds), 2)
        text = ds[0]["text"]
        self.assertIn("<|im_start|>system", text)
        self.assertIn("<|im_start|>user", text)
        self.assertIn("<|im_start|>assistant", text)
        self.assertIn("weapon poly budget", text)

    def test_llama3_format(self):
        """llama3 format produces <|begin_of_text|> tagged text."""
        ds = prepare_mossy_dataset(self._pairs(), format="llama3")
        self.assertEqual(len(ds), 2)
        text = ds[0]["text"]
        self.assertIn("<|begin_of_text|>", text)
        self.assertIn("<|start_header_id|>user<|end_header_id|>", text)

    def test_gemma_format(self):
        """gemma format produces <start_of_turn> tagged text."""
        ds = prepare_mossy_dataset(self._pairs(), format="gemma")
        self.assertEqual(len(ds), 2)
        text = ds[0]["text"]
        self.assertIn("<start_of_turn>user", text)
        self.assertIn("<start_of_turn>model", text)

    def test_invalid_format_raises(self):
        """Unknown format raises ValueError."""
        with self.assertRaises(ValueError):
            prepare_mossy_dataset(self._pairs(), format="unknown_format")

    def test_custom_system_prompt(self):
        """Custom system prompt is embedded in the output."""
        custom = "You are a master blacksmith."
        ds = prepare_mossy_dataset(self._pairs(), system_prompt=custom, format="chatml")
        self.assertIn(custom, ds[0]["text"])

    def test_dataset_has_text_column(self):
        """Output dataset always has a 'text' column."""
        ds = prepare_mossy_dataset(self._pairs())
        self.assertIn("text", ds.column_names)

    def test_empty_dataset(self):
        """Empty input produces an empty dataset."""
        ds = prepare_mossy_dataset([])
        self.assertEqual(len(ds), 0)


class TestListRecommendedModels(unittest.TestCase):
    """Tests for list_recommended_models helper."""

    def test_returns_string(self):
        """list_recommended_models returns a non-empty string."""
        result = list_recommended_models()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 100)

    def test_contains_model_keys(self):
        """Output mentions known model keys."""
        result = list_recommended_models()
        self.assertIn("gemma-4-9b", result)
        self.assertIn("phi-4", result)

    def test_vram_filter(self):
        """With a 4 GB VRAM filter only small models appear."""
        result = list_recommended_models(available_vram_gb=4.0)
        # gemma-4-27b needs 20 GB — must not appear
        self.assertNotIn("gemma-4-27b", result)
        # gemma-2-2b needs 2 GB — must appear
        self.assertIn("gemma-2-2b", result)

    def test_unsloth_marker_present(self):
        """Output includes the Unsloth marker for supported models."""
        result = list_recommended_models()
        self.assertIn("Unsloth", result)


if __name__ == "__main__":
    unittest.main()
