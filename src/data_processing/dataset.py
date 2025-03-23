import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from data_processing.audio_preprocessing import preprocess_audio

class DialectDataset(Dataset):
    """
    Dataset for dialect-specific speech recognition
    """
    def __init__(self, data_df, processor, max_length=None, load_from_processed=True):
        """
        Initialize dataset
        
        Args:
            data_df: DataFrame with audio paths and transcripts
            processor: Processor for model (WhisperProcessor or Wav2Vec2Processor)
            max_length: Maximum length of audio (in samples)
            load_from_processed: Whether to load preprocessed files
        """
        self.data_df = data_df
        self.processor = processor
        self.max_length = max_length
        self.load_from_processed = load_from_processed
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        
        # Load audio
        if self.load_from_processed and 'processed_path' in row:
            # Load preprocessed audio
            audio = np.load(row['processed_path'])
            sr = 16000  # Assuming all preprocessed files have this sample rate
        else:
            # Process audio on-the-fly
            audio_path = row['audio_path']
            audio, sr = preprocess_audio(audio_path)
        
        # Truncate or pad if necessary
        if self.max_length is not None:
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            elif len(audio) < self.max_length:
                padding = np.zeros(self.max_length - len(audio))
                audio = np.concatenate([audio, padding])
        
        # Get transcript and dialect
        text = row['text']
        dialect = row.get('dialect', 'unknown')
        
        # Process audio for model input
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Process text for labels
        with self.processor.as_target_processor():
            labels = self.processor(text, return_tensors="pt").input_ids.squeeze(0)
        
        return {
            "input_values": inputs["input_values"],
            "attention_mask": inputs.get("attention_mask", None),
            "labels": labels,
            "dialect": dialect,
            "text": text,
            "audio_path": row.get('audio_path', row.get('processed_path', ''))
        }
