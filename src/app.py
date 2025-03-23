# File: app.py
import os
import torch
import gradio as gr
import numpy as np
import json
from src.model.model_setup import load_whisper_model, load_wav2vec2_model
from src.error_correction.correction import ErrorCorrector

# Load model
def load_model(model_path, model_type="whisper"):
    """Load model and processor"""
    if model_type == "whisper":
        processor, model = load_whisper_model(model_path)
    else:
        processor, model = load_wav2vec2_model(model_path)
    
    return processor, model

# Load error corrector
def load_corrector(confusion_path):
    """Load error correction system"""
    with open(confusion_path, 'r') as f:
        confusion_dict = json.load(f)
    
    corrector = ErrorCorrector(
        confusion_dict=confusion_dict,
        language_model="bert-base-uncased"
    )
    
    return corrector

# Transcribe audio
def transcribe_audio(audio_path, model, processor, corrector, dialect=None, apply_correction=True):
    """Transcribe audio and apply correction if needed"""
    # Load and preprocess audio
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Get model prediction
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        if hasattr(model, "generate"):
            # For Whisper
            outputs = model.generate(**inputs)
            transcript = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            # For Wav2Vec2
            outputs = model(**inputs)
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            transcript = processor.batch_decode(predicted_ids)[0]
    
    # Apply correction if requested
    if apply_correction and corrector:
        corrected_transcript = corrector.correct_transcript(transcript, dialect)
        return transcript, corrected_transcript
    
    return transcript, transcript

# Main application
def create_demo(model_path="./results/final_model", 
                model_type="whisper",
                confusion_path="./results/confusion_matrix.json"):
    # Load model and corrector
    processor, model = load_model(model_path, model_type)
    corrector = load_corrector(confusion_path) if os.path.exists(confusion_path) else None
    
    # Define dialects
    dialects = ["american", "british", "australian", "indian", "nigerian", "unknown"]
    
    # Create Gradio interface
    demo = gr.Interface(
        fn=lambda audio, dialect, apply_correction: transcribe_audio(
            audio, model, processor, corrector, dialect, apply_correction
        ),
        inputs=[
            gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio Input"),
            gr.Dropdown(choices=dialects, label="Dialect (Optional)", value="unknown"),
            gr.Checkbox(label="Apply Error Correction", value=True)
        ],
        outputs=[
            gr.Textbox(label="Original Transcript"),
            gr.Textbox(label="Corrected Transcript")
        ],
        title="Dialect-Aware Speech Recognition",
        description="Upload audio or record your voice to transcribe with dialect-specific adaptation and error correction.",
        examples=[
            ["./examples/american_sample.wav", "american", True],
            ["./examples/british_sample.wav", "british", True],
            ["./examples/indian_sample.wav", "indian", True]
        ]
    )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)