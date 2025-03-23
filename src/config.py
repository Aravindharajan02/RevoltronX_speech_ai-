# File: config.py
import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Data configuration
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Model configuration
DEFAULT_MODEL = {
    "whisper": "openai/whisper-small",
    "wav2vec2": "facebook/wav2vec2-base-960h"
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 8,
    "learning_rate": 1e-5,
    "max_steps": 5000,
    "warmup_steps": 500,
    "eval_steps": 500,
    "save_steps": 500
}

# Dialect weights (for training focus)
DIALECT_WEIGHTS = {
    "american": 1.0,
    "british": 1.2,
    "australian": 1.2,
    "indian": 1.5,
    "nigerian": 1.5,
    "unknown": 1.0
}

# ASR model parameters
ASR_CONFIG = {
    "whisper": {
        "max_length": 225,
        "language": "en",
        "task": "transcribe"
    },
    "wav2vec2": {
        "ctc_loss_reduction": "mean"
    }
}

# Error correction parameters
ERROR_CORRECTION = {
    "min_occurrences": 3,  # Minimum occurrences to include in confusion matrix
    "context_window": 5,   # Words before/after for context-aware correction
    "confidence_threshold": 0.3  # Minimum confidence for language model corrections
}