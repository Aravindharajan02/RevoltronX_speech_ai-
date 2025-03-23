import librosa
import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm

def preprocess_audio(audio_path, sample_rate=16000):
    """
    Load and preprocess audio file
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        processed_audio: Preprocessed audio as numpy array
        sample_rate: Sample rate of processed audio
    """
    # Load audio file and resample if needed
    audio, sr = librosa.load(audio_path, sr=None)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    
    # Normalize audio
    audio = audio / (np.max(np.abs(audio)) + 1e-10)
    
    # Optional: Apply noise reduction
    # Using simple high-pass filter to remove background noise
    # audio = librosa.effects.preemphasis(audio)
    
    return audio, sample_rate

def process_dataset(data_csv, output_dir, processor=None):
    """
    Process all audio files in a dataset and save preprocessed versions
    
    Args:
        data_csv: Path to CSV with audio paths and metadata
        output_dir: Directory to save processed files
        processor: Optional processor for feature extraction
        
    Returns:
        processed_df: DataFrame with updated paths and metadata
    """
    data_df = pd.read_csv(data_csv)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    processed_paths = []
    durations = []
    
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Processing audio files"):
        audio_path = row['audio_path']
        file_id = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Process audio
        audio, sr = preprocess_audio(audio_path)
        
        # Save processed audio
        output_path = os.path.join(output_dir, f"{file_id}.npy")
        np.save(output_path, audio)
        
        # Extract features if processor is provided
        if processor:
            features = processor(audio, sampling_rate=sr, return_tensors="pt")
            features_path = os.path.join(output_dir, f"{file_id}_features.pt")
            torch.save(features, features_path)
        
        processed_paths.append(output_path)
        durations.append(len(audio) / sr)
    
    # Update DataFrame
    data_df['processed_path'] = processed_paths
    data_df['duration'] = durations
    
    # Save updated DataFrame
    output_csv = os.path.join(output_dir, "processed_metadata.csv")
    data_df.to_csv(output_csv, index=False)
    
    return data_df
