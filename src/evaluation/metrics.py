# File: src/evaluation/metrics.py
import evaluate
import numpy as np
from jiwer import wer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch

def compute_metrics(pred, processor):
    """
    Compute WER and other metrics from model predictions
    
    Args:
        pred: EvalPrediction object with predictions and label_ids
        processor: Processor for decoding predictions
        
    Returns:
        metrics: Dictionary of metrics
    """
    # For Whisper models
    if hasattr(pred, "predictions") and isinstance(pred.predictions, tuple):
        pred_ids = pred.predictions[0]
    # For Wav2Vec2 models
    elif hasattr(pred, "predictions"):
        pred_ids = np.argmax(pred.predictions, axis=-1)
    else:
        raise ValueError("Unsupported prediction format")
    
    # Replace -100 with pad token ID
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(pred.label_ids, skip_special_tokens=True)
    
    # Compute WER
    wer_metric = evaluate.load("wer")
    wer_score = wer_metric.compute(predictions=pred_str, references=label_str)
    
    # Compute Character Error Rate
    cer_metric = evaluate.load("cer")
    cer_score = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {
        "wer": wer_score,
        "cer": cer_score
    }

def evaluate_by_dialect(model, test_dataset, processor, data_collator, device="cuda", batch_size=8):
    """
    Evaluate model performance by dialect
    
    Args:
        model: ASR model
        test_dataset: Test dataset
        processor: Processor for decoding
        data_collator: Data collator for batching
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        
    Returns:
        results: Dictionary with dialect-specific metrics
    """
    # Group samples by dialect
    dialect_groups = {}
    for idx in range(len(test_dataset)):
        item = test_dataset[idx]
        dialect = item["dialect"]
        if dialect not in dialect_groups:
            dialect_groups[dialect] = []
        dialect_groups[dialect].append(idx)
    
    results = {}
    all_predictions = []
    all_references = []
    all_dialects = []
    
    # Set model to evaluation mode
    model.eval()
    
    # Evaluate each dialect separately
    for dialect, indices in tqdm(dialect_groups.items(), desc="Evaluating dialects"):
        dialect_dataset = torch.utils.data.Subset(test_dataset, indices)
        dialect_loader = torch.utils.data.DataLoader(
            dialect_dataset, 
            batch_size=batch_size, 
            collate_fn=data_collator
        )
        
        dialect_predictions = []
        dialect_references = []
        
        for batch in tqdm(dialect_loader, desc=f"Evaluating {dialect}", leave=False):
            # Move inputs to device
            input_values = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Get model predictions
            with torch.no_grad():
                if hasattr(model, "generate"):
                    # For sequence-to-sequence models like Whisper
                    outputs = model.generate(
                        input_values, 
                        attention_mask=attention_mask
                    )
                    predicted_ids = outputs
                else:
                    # For CTC models like Wav2Vec2
                    outputs = model(
                        input_values=input_values,
                        attention_mask=attention_mask
                    )
                    predicted_ids = torch.argmax(outputs.logits, dim=-1)
            
            # Decode predictions and references
            decoded_preds = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            labels = batch["labels"].to(device)
            labels[labels == -100] = processor.tokenizer.pad_token_id
            decoded_refs = processor.batch_decode(labels, skip_special_tokens=True)
            
            # Collect results
            dialect_predictions.extend(decoded_preds)
            dialect_references.extend(decoded_refs)
            
            # Also collect for overall evaluation
            all_predictions.extend(decoded_preds)
            all_references.extend(decoded_refs)
            all_dialects.extend([dialect] * len(decoded_preds))
        
        # Calculate WER for this dialect
        wer_score = wer(dialect_references, dialect_predictions)
        cer_metric = evaluate.load("cer")
        cer_score = cer_metric.compute(predictions=dialect_predictions, references=dialect_references)
        
        # Store results
        results[dialect] = {
            "wer": wer_score,
            "cer": cer_score,
            "samples": len(indices),
            "predictions": dialect_predictions,
            "references": dialect_references
        }
    
    # Calculate overall metrics
    overall_wer = wer(all_references, all_predictions)
    overall_cer = evaluate.load("cer").compute(predictions=all_predictions, references=all_references)
    
    results["overall"] = {
        "wer": overall_wer,
        "cer": overall_cer,
        "samples": len(test_dataset),
        "predictions": all_predictions,
        "references": all_references,
        "dialects": all_dialects
    }
    
    return results

def plot_dialect_performance(results, output_path=None):
    """
    Visualize model performance by dialect
    
    Args:
        results: Dictionary with dialect-specific metrics
        output_path: Path to save plot
        
    Returns:
        fig: Matplotlib figure
    """
    dialects = [d for d in results.keys() if d != "overall"]
    wer_scores = [results[d]["wer"] * 100 for d in dialects]  # Convert to percentage
    sample_sizes = [results[d]["samples"] for d in dialects]
    
    # Sort by WER
    sorted_indices = np.argsort(wer_scores)
    sorted_dialects = [dialects[i] for i in sorted_indices]
    sorted_wer = [wer_scores[i] for i in sorted_indices]
    sorted_samples = [sample_sizes[i] for i in sorted_indices]
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot WER bars
    bars = ax1.bar(sorted_dialects, sorted_wer, alpha=0.7, color='skyblue')
    ax1.set_ylabel('Word Error Rate (%)', fontsize=12)
    ax1.set_xlabel('Dialect', fontsize=12)
    ax1.axhline(y=results["overall"]["wer"] * 100, color='r', linestyle='-', 
                label=f'Overall WER: {results["overall"]["wer"]*100:.2f}%')
    
    # Add sample size as text on bars
    for i, (bar, sample) in enumerate(zip(bars, sorted_samples)):
        ax1.text(i, bar.get_height() + 0.5, f'n={sample}', 
                ha='center', va='bottom', fontsize=10)
    
    # Create second y-axis for CER
    ax2 = ax1.twinx()
    ax2.plot(sorted_dialects, [results[d]["cer"] * 100 for d in sorted_dialects], 
             'o-', color='darkred', label=f'CER (Overall: {results["overall"]["cer"]*100:.2f}%)')
    ax2.set_ylabel('Character Error Rate (%)', fontsize=12)
    
    # Set title and legend
    ax1.set_title('ASR Performance by Dialect', fontsize=14, pad=20)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig
