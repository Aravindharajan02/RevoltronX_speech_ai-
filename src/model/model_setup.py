
# File: src/model/model_setup.py
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    TrainingArguments
)
import torch

def load_whisper_model(model_name="openai/whisper-small", device="cuda"):
    """
    Load Whisper model and processor
    
    Args:
        model_name: Name of pretrained model
        device: Device to load model to
        
    Returns:
        processor: WhisperProcessor
        model: WhisperForConditionalGeneration
    """
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device)
    
    return processor, model

def load_wav2vec2_model(model_name="facebook/wav2vec2-base-960h", device="cuda"):
    """
    Load Wav2Vec2 model and processor
    
    Args:
        model_name: Name of pretrained model
        device: Device to load model to
        
    Returns:
        processor: Wav2Vec2Processor
        model: Wav2Vec2ForCTC
    """
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device)
    
    return processor, model

def get_training_args(output_dir="./results", model_type="whisper"):
    """
    Create training arguments
    
    Args:
        output_dir: Directory to save outputs
        model_type: Type of model (whisper or wav2vec2)
        
    Returns:
        training_args: TrainingArguments
    """
    # Common arguments
    args = {
        "output_dir": output_dir,
        "evaluation_strategy": "steps",
        "eval_steps": 500,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-5,
        "warmup_steps": 500,
        "max_steps": 5000,
        "gradient_checkpointing": True,
        "fp16": True,
        "save_steps": 500,
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "greater_is_better": False,
    }
    
    # Model-specific adjustments
    if model_type.lower() == "whisper":
        args["predict_with_generate"] = True
        args["generation_max_length"] = 225
        args["metric_for_best_model"] = "wer"
    elif model_type.lower() == "wav2vec2":
        args["metric_for_best_model"] = "wer"
    
    return TrainingArguments(**args)
