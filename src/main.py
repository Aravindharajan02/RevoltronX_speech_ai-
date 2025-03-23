# File: src/main.py
import os
import argparse
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from data_processing.audio_preprocessing import process_dataset
from data_processing.dataset import DialectDataset
from data_processing.collator import DataCollatorWithPadding
from model.model_setup import load_whisper_model, load_wav2vec2_model, get_training_args
from model.trainer import DialectTrainer
from evaluation.metrics import evaluate_by_dialect, plot_dialect_performance
from error_correction.error_analysis import analyze_error_patterns, build_confusion_matrix
from error_correction.correction import ErrorCorrector

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Speech Recognition with Dialect Adaptation")
    
    parser.add_argument("--mode", type=str, required=True, choices=["train", "evaluate", "analyze", "correct"],
                        help="Mode to run the script in")
    parser.add_argument("--data_csv", type=str, help="Path to dataset CSV file")
    parser.add_argument("--output_dir", type=str, default="./results", 
                        help="Directory to save outputs")
    parser.add_argument("--model_type", type=str, default="whisper",
                        choices=["whisper", "wav2vec2"], help="Type of model to use")
    parser.add_argument("--model_name", type=str, 
                        help="Pretrained model name or path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for training/evaluation")
    parser.add_argument("--dialect_weights", type=str, 
                        help="Path to JSON file with dialect weights for training")
    
    return parser.parse_args()

def main():
    """Main function to run the application"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set default model names if not provided
    if not args.model_name:
        if args.model_type == "whisper":
            args.model_name = "openai/whisper-small"
        else:
            args.model_name = "facebook/wav2vec2-base-960h"
    
    print(f"Using {args.model_type} model: {args.model_name}")
    
    # Load model and processor
    if args.model_type == "whisper":
        processor, model = load_whisper_model(args.model_name, args.device)
    else:
        processor, model = load_wav2vec2_model(args.model_name, args.device)
    
    # Load data
    if args.data_csv:
        data_df = pd.read_csv(args.data_csv)
        print(f"Loaded {len(data_df)} samples from {args.data_csv}")
    
    # Process based on selected mode
    if args.mode == "train":
        train_model(args, processor, model, data_df)
    elif args.mode == "evaluate":
        evaluate_model(args, processor, model, data_df)
    elif args.mode == "analyze":
        analyze_errors(args, processor, model, data_df)
    elif args.mode == "correct":
        apply_corrections(args, processor, model, data_df)

def train_model(args, processor, model, data_df):
    """Train the ASR model"""
    # Split data into train/validation
    if "split" in data_df.columns:
        train_df = data_df[data_df.split == "train"]
        eval_df = data_df[data_df.split == "validation"]
    else:
        # Randomly split if no split column
        train_size = int(0.8 * len(data_df))
        train_df = data_df.iloc[:train_size]
        eval_df = data_df.iloc[train_size:]
    
    print(f"Training on {len(train_df)} samples, validating on {len(eval_df)} samples")
    
    # Create datasets
    train_dataset = DialectDataset(train_df, processor)
    eval_dataset = DialectDataset(eval_df, processor)
    
    # Create data collator
    data_collator = DataCollatorWithPadding(processor=processor)
    
    # Load dialect weights if provided
    dialect_weights = None
    if args.dialect_weights:
        with open(args.dialect_weights, 'r') as f:
            dialect_weights = json.load(f)
        print(f"Loaded dialect weights: {dialect_weights}")
    
    # Get training arguments
    training_args = get_training_args(args.output_dir, args.model_type)
    
    # Initialize trainer
    trainer = DialectTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor if hasattr(processor, "feature_extractor") else processor,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        dialect_weights=dialect_weights
    )
    
    # Train model
    trainer.train()
    
    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    
    # Evaluate on test set
    metrics = trainer.evaluate()
    print(f"Final evaluation metrics: {metrics}")
    
    with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

def evaluate_model(args, processor, model, data_df):
    """Evaluate the ASR model by dialect"""
    # Use test split if available
    if "split" in data_df.columns:
        test_df = data_df[data_df.split == "test"]
    else:
        test_df = data_df
    
    print(f"Evaluating on {len(test_df)} samples")
    
    # Create dataset and data collator
    test_dataset = DialectDataset(test_df, processor)
    data_collator = DataCollatorWithPadding(processor=processor)
    
    # Evaluate by dialect
    results = evaluate_by_dialect(
        model=model,
        test_dataset=test_dataset,
        processor=processor,
        data_collator=data_collator,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Print summary
    print("\nResults by dialect:")
    for dialect, metrics in results.items():
        if dialect != "overall":
            print(f"{dialect}: WER={metrics['wer']:.4f}, CER={metrics['cer']:.4f}, Samples={metrics['samples']}")
    
    print(f"\nOverall: WER={results['overall']['wer']:.4f}, CER={results['overall']['cer']:.4f}")
    
    # Plot results
    fig = plot_dialect_performance(results, os.path.join(args.output_dir, "dialect_performance.png"))
    
    # Save detailed results
    with open(os.path.join(args.output_dir, "evaluation_results.json"), 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        clean_results = {}
        for dialect, metrics in results.items():
            clean_metrics = {k: v for k, v in metrics.items() if k not in ["predictions", "references", "dialects"]}
            clean_results[dialect] = clean_metrics
        
        json.dump(clean_results, f, indent=2)
    
    # Save predictions for error analysis
    with open(os.path.join(args.output_dir, "predictions.json"), 'w') as f:
        json.dump({
            "predictions": results["overall"]["predictions"],
            "references": results["overall"]["references"],
            "dialects": results["overall"]["dialects"]
        }, f, indent=2)

def analyze_errors(args, processor, model, data_df):
    """Analyze error patterns in ASR output"""
    # Load predictions if available
    predictions_path = os.path.join(args.output_dir, "predictions.json")
    if os.path.exists(predictions_path):
        with open(predictions_path, 'r') as f:
            data = json.load(f)
            predictions = data["predictions"]
            references = data["references"]
            dialects = data["dialects"]
        
        print(f"Loaded {len(predictions)} predictions for error analysis")
    else:
        # Evaluate model to get predictions
        print("No predictions found. Running evaluation first...")
        evaluate_model(args, processor, model, data_df)
        
        # Load newly created predictions
        with open(predictions_path, 'r') as f:
            data = json.load(f)
            predictions = data["predictions"]
            references = data["references"]
            dialects = data["dialects"]
    
    # Analyze error patterns
    error_analysis = analyze_error_patterns(predictions, references, dialects)
    
    # Build confusion matrix
    confusion_matrix = build_confusion_matrix(error_analysis, min_occurrences=3)
    
    # Visualize common errors
    visualize_common_errors(
        error_analysis,
        top_n=20,
        output_path=os.path.join(args.output_dir, "common_errors.png")
    )
    
    # Save error analysis results
    with open(os.path.join(args.output_dir, "error_analysis.json"), 'w') as f:
        json.dump(error_analysis, f, indent=2)
    
    # Save confusion matrix
    with open(os.path.join(args.output_dir, "confusion_matrix.json"), 'w') as f:
        json.dump(confusion_matrix, f, indent=2)
    
    # Print summary of top errors
    print("\nTop substitution errors:")
    all_subs = {}
    for dialect, subs in error_analysis["substitutions"].items():
        for pair, count in subs.items():
            from_word, to_word = pair
            error_key = f"{from_word}->{to_word}"
            if error_key not in all_subs:
                all_subs[error_key] = 0
            all_subs[error_key] += count
    
    top_errors = sorted(all_subs.items(), key=lambda x: x[1], reverse=True)[:10]
    for error, count in top_errors:
        print(f"{error}: {count} occurrences")

def apply_corrections(args, processor, model, data_df):
    """Apply error correction to ASR output"""
    # Load confusion matrix if available
    confusion_path = os.path.join(args.output_dir, "confusion_matrix.json")
    if os.path.exists(confusion_path):
        with open(confusion_path, 'r') as f:
            confusion_dict = json.load(f)
        print(f"Loaded confusion matrix with {len(confusion_dict)} entries")
    else:
        # Run error analysis first
        print("No confusion matrix found. Running error analysis first...")
        analyze_errors(args, processor, model, data_df)
        
        # Load newly created confusion matrix
        with open(confusion_path, 'r') as f:
            confusion_dict = json.load(f)
    
    # Load predictions
    predictions_path = os.path.join(args.output_dir, "predictions.json")
    with open(predictions_path, 'r') as f:
        data = json.load(f)
        predictions = data["predictions"]
        references = data["references"]
        dialects = data["dialects"]
    
    # Initialize error corrector
    corrector = ErrorCorrector(
        confusion_dict=confusion_dict,
        language_model="bert-base-uncased"  # Can be customized
    )
    
    # Apply corrections to predictions
    print("Applying error correction...")
    corrected_predictions = []
    for i, (pred, dialect) in tqdm(enumerate(zip(predictions, dialects)), total=len(predictions)):
        corrected_pred = corrector.correct_transcript(pred, dialect)
        corrected_predictions.append(corrected_pred)
    
    # Evaluate original vs corrected performance
    orig_wer = wer(references, predictions)
    corrected_wer = wer(references, corrected_predictions)
    
    print(f"\nOriginal WER: {orig_wer:.4f}")
    print(f"Corrected WER: {corrected_wer:.4f}")
    print(f"Improvement: {(orig_wer - corrected_wer) * 100:.2f}% ({(1 - corrected_wer/orig_wer) * 100:.2f}% reduction)")
    
    # Save corrected predictions
    with open(os.path.join(args.output_dir, "corrected_predictions.json"), 'w') as f:
        json.dump({
            "original_predictions": predictions,
            "corrected_predictions": corrected_predictions,
            "references": references,
            "dialects": dialects,
            "original_wer": orig_wer,
            "corrected_wer": corrected_wer
        }, f, indent=2)
    
    # Print example corrections
    print("\nExample corrections:")
    for i in range(min(5, len(predictions))):
        if predictions[i] != corrected_predictions[i]:
            print(f"\nDialect: {dialects[i]}")
            print(f"Original: {predictions[i]}")
            print(f"Corrected: {corrected_predictions[i]}")
            print(f"Reference: {references[i]}")

if __name__ == "__main__":
    main()