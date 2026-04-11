"""AI Helper – desktop monitoring and orchestration package."""

__version__ = "0.1.0"

# Make key classes easily importable
try:
    from .gemma_finetuner import (
        GemmaFineTuner,
        FineTuneConfig,
        GPUMemoryProfiler,
        GEMMA_MODELS,
        TrainingResult,
        InferenceResult,
    )
    __all__ = [
        "GemmaFineTuner",
        "FineTuneConfig",
        "GPUMemoryProfiler",
        "GEMMA_MODELS",
        "TrainingResult",
        "InferenceResult",
    ]
except ImportError:
    # Gracefully handle if dependencies not installed
    pass
