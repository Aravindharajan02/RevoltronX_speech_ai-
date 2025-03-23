# File: src/model/trainer.py
from transformers import Trainer
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple

class DialectTrainer(Trainer):
    """
    Custom trainer with dialect-specific handling
    """
    def __init__(self, *args, **kwargs):
        # Extract dialect weights if provided
        self.dialect_weights = kwargs.pop("dialect_weights", None)
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation with dialect weighting
        """
        # Extract dialect information
        dialect = inputs.pop("dialect", None)
        
        # Keep references to original text and audio for debugging
        text = inputs.pop("text", None)
        audio_path = inputs.pop("audio_path", None)
        
        # Compute standard loss
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Apply dialect-specific weighting if provided
        if dialect is not None and self.dialect_weights is not None:
            # Create weight tensor based on dialects in batch
            weights = torch.ones_like(loss)
            for i, d in enumerate(dialect):
                if d in self.dialect_weights:
                    weights[i] = self.dialect_weights[d]
            
            # Apply weights to loss
            loss = loss * weights
            loss = loss.mean()
            
            # Replace the output loss for proper gradient computation
            outputs.loss = loss
        
        return (loss, outputs) if return_outputs else loss