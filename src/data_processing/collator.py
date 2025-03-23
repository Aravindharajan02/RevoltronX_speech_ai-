# File: src/data_processing/collator.py
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
import torch

@dataclass
class DataCollatorWithPadding:
    """
    Data collator for batching ASR samples
    """
    processor: any
    padding: Union[bool, str] = True
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract audio inputs
        input_values = [feature["input_values"] for feature in features]
        
        # Extract attention masks if they exist
        attention_mask = None
        if "attention_mask" in features[0] and features[0]["attention_mask"] is not None:
            attention_mask = [feature["attention_mask"] for feature in features]
        
        # Get labels
        label_features = [feature["labels"] for feature in features]
        
        # Pad inputs to max length in batch
        batch = self.processor.pad(
            {"input_values": input_values}, 
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Pad labels
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                {"input_ids": label_features},
                padding=self.padding,
                return_tensors="pt",
            )
        
        # Replace padding with -100 to ignore these in loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        
        # Add other metadata
        batch["dialect"] = [feature["dialect"] for feature in features]
        batch["text"] = [feature["text"] for feature in features]
        batch["audio_path"] = [feature["audio_path"] for feature in features]
        
        return batch
