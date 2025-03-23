
# File: src/error_correction/correction.py
import numpy as np
import torch
from transformers import pipeline
from nltk.tokenize import word_tokenize
import nltk
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('words')
from nltk.corpus import wordnet, words

class ErrorCorrector:
    """
    Error correction system for ASR transcripts
    """
    def __init__(self, confusion_dict=None, language_model=None, dialect_specific=None):
        """
        Initialize error corrector
        
        Args:
            confusion_dict: Dictionary of common errors from error analysis
            language_model: Language model for context-aware correction
            dialect_specific: Dictionary of dialect-specific corrections
        """
        self.confusion_dict = confusion_dict if confusion_dict else {}
        self.dialect_specific = dialect_specific if dialect_specific else {}
        
        # Initialize language model for context-aware correction
        if language_model:
            self.nlp = pipeline("fill-mask", model=language_model)
        else:
            self.nlp = None
        
        # Initialize word list for spell checking
        self.word_set = set(words.words())   
        
def rule_based_correction(self, text, dialect=None):
    """
    Apply rule-based corrections using confusion matrix
    
    Args:
        text: ASR transcript to correct
        dialect: Optional dialect for dialect-specific corrections
        
    Returns:
        corrected_text: Corrected transcript
    """
    # Tokenize text
    tokens = word_tokenize(text.lower())
    corrected_tokens = []
    
    # Apply corrections
    for token in tokens:
        # Check in confusion dictionary
        if token in self.confusion_dict:
            corrections = self.confusion_dict[token]
            # Use most frequent correction
            if corrections:
                best_correction = max(corrections.items(), key=lambda x: x[1])[0]
                corrected_tokens.append(best_correction)
                continue
        
        # Apply dialect-specific corrections if available
        if dialect and dialect in self.dialect_specific and token in self.dialect_specific[dialect]:
            corrected_tokens.append(self.dialect_specific[dialect][token])
            continue
        
        # Keep original token if no corrections found
        corrected_tokens.append(token)
    
    # Join tokens back into text
    corrected_text = ' '.join(corrected_tokens)
    
    # Fix capitalization and spacing
    corrected_text = self._fix_formatting(corrected_text)
    
    return corrected_text

def _fix_formatting(self, text):
    """
    Fix capitalization and spacing in corrected text
    
    Args:
        text: Text to fix
        
    Returns:
        formatted_text: Properly formatted text
    """
    # Capitalize first letter of sentences
    formatted_text = '. '.join(s.capitalize() for s in text.split('. '))
    
    # Fix spacing around punctuation
    formatted_text = re.sub(r'\s+([.,;:!?])', r'\1', formatted_text)
    
    # Fix spacing after punctuation
    formatted_text = re.sub(r'([.,;:!?])(\w)', r'\1 \2', formatted_text)
    
    return formatted_text

def context_aware_correction(self, text):
    """
    Use language model for context-aware correction
    
    Args:
        text: ASR transcript to correct
        
    Returns:
        corrected_text: Corrected transcript
    """
    if not self.nlp:
        return text
    
    # Tokenize text
    tokens = word_tokenize(text.lower())
    corrected_tokens = list(tokens)
    
    # Analyze each token in context
    for i in range(len(tokens)):
        token = tokens[i]
        
        # Skip punctuation and common words
        if len(token) <= 2 or token in {"the", "and", "or", "but", "in", "on", "at", "to", "a", "an"}:
            continue
        
        # Check if token is in word list (spell check)
        if token not in self.word_set:
            # Create context window (mask the current token)
            context = tokens.copy()
            context[i] = self.nlp.tokenizer.mask_token
            context_text = ' '.join(context)
            
            try:
                # Get predictions from language model
                predictions = self.nlp(context_text, top_k=5)
                
                # Find best prediction
                best_prediction = None
                best_score = 0
                
                for pred in predictions:
                    pred_token = pred["token_str"].lower().strip()
                    score = pred["score"]
                    
                    # Prefer predictions that are close to original word
                    edit_dist = edit_distance(token, pred_token)
                    similarity_factor = 1.0 / (1.0 + edit_dist)
                    adjusted_score = score * similarity_factor
                    
                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_prediction = pred_token
                
                # Apply correction if confident enough
                if best_prediction and best_score > 0.3:
                    corrected_tokens[i] = best_prediction
            except Exception as e:
                # Fall back to original token on error
                pass
    
    # Join tokens back into text
    corrected_text = ' '.join(corrected_tokens)
    
    # Fix formatting
    corrected_text = self._fix_formatting(corrected_text)
    
    return corrected_text

def correct_transcript(self, text, dialect=None):
    """
    Apply multiple correction strategies to improve ASR transcript
    
    Args:
        text: ASR transcript to correct
        dialect: Optional dialect for dialect-specific corrections
        
    Returns:
        corrected_text: Corrected transcript
    """
    # First apply rule-based corrections
    corrected_text = self.rule_based_correction(text, dialect)
    
    # Then apply context-aware corrections if language model is available
    if self.nlp:
        corrected_text = self.context_aware_correction(corrected_text)
    
    return corrected_text        