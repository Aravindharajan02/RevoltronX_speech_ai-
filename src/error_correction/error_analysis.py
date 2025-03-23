# File: src/error_correction/error_analysis.py
import numpy as np
import pandas as pd
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.metrics import edit_distance
from tqdm import tqdm

# Download NLTK resources
nltk.download('punkt')

def align_sequences(pred, ref):
    """
    Align predicted and reference sequences using dynamic programming
    
    Args:
        pred: Predicted sequence (string or list of tokens)
        ref: Reference sequence (string or list of tokens)
        
    Returns:
        aligned_pred: Aligned predicted sequence with insertions/deletions marked
        aligned_ref: Aligned reference sequence with insertions/deletions marked
        operations: List of operations (match, substitute, insert, delete)
    """
    # Tokenize if inputs are strings
    if isinstance(pred, str):
        pred = word_tokenize(pred.lower())
    if isinstance(ref, str):
        ref = word_tokenize(ref.lower())
    
    # Initialize cost matrix
    n, m = len(pred), len(ref)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    
    # Fill first row and column
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    
    # Fill cost matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]  # Match
            else:
                dp[i, j] = min(
                    dp[i - 1, j] + 1,  # Deletion
                    dp[i, j - 1] + 1,  # Insertion
                    dp[i - 1, j - 1] + 1  # Substitution
                )
    
    # Traceback to find alignment
    aligned_pred = []
    aligned_ref = []
    operations = []
    i, j = n, m
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and pred[i - 1] == ref[j - 1]:
            # Match
            aligned_pred.insert(0, pred[i - 1])
            aligned_ref.insert(0, ref[j - 1])
            operations.insert(0, "match")
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            # Substitution
            aligned_pred.insert(0, pred[i - 1])
            aligned_ref.insert(0, ref[j - 1])
            operations.insert(0, "substitute")
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            # Deletion
            aligned_pred.insert(0, pred[i - 1])
            aligned_ref.insert(0, "<del>")
            operations.insert(0, "delete")
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            # Insertion
            aligned_pred.insert(0, "<ins>")
            aligned_ref.insert(0, ref[j - 1])
            operations.insert(0, "insert")
            j -= 1
    
    return aligned_pred, aligned_ref, operations

def analyze_error_patterns(predictions, references, dialects=None):
    """
    Analyze error patterns in ASR output
    
    Args:
        predictions: List of predicted transcripts
        references: List of reference transcripts
        dialects: List of dialect labels (optional)
        
    Returns:
        error_patterns: Dictionary with error statistics
    """
    # Initialize error patterns dictionary
    error_patterns = defaultdict(lambda: defaultdict(int))
    substitutions = defaultdict(lambda: defaultdict(int))
    deletions = defaultdict(int)
    insertions = defaultdict(int)
    
    # Process each sample
    for i, (pred, ref) in tqdm(enumerate(zip(predictions, references)), 
                               total=len(predictions), 
                               desc="Analyzing errors"):
        # Get dialect
        dialect = dialects[i] if dialects is not None else "unknown"
        
        # Align sequences
        aligned_pred, aligned_ref, operations = align_sequences(pred, ref)
        
        # Analyze operations
        for j, op in enumerate(operations):
            if op == "substitute":
                pred_word = aligned_pred[j]
                ref_word = aligned_ref[j]
                error_key = f"{pred_word}->{ref_word}"
                error_patterns[dialect][error_key] += 1
                substitutions[dialect][(pred_word, ref_word)] += 1
            elif op == "delete":
                error_key = f"{aligned_pred[j]}->∅"
                error_patterns[dialect][error_key] += 1
                deletions[dialect][aligned_pred[j]] += 1
            elif op == "insert":
                error_key = f"∅->{aligned_ref[j]}"
                error_patterns[dialect][error_key] += 1
                insertions[dialect][aligned_ref[j]] += 1
    
    # Convert defaultdicts to regular dicts for easier handling
    error_patterns_dict = {}
    for dialect, errors in error_patterns.items():
        error_patterns_dict[dialect] = dict(errors)
    
    substitutions_dict = {}
    for dialect, subs in substitutions.items():
        substitutions_dict[dialect] = dict(subs)
    
    deletions_dict = {}
    for dialect, dels in deletions.items():
        deletions_dict[dialect] = dict(dels)
    
    insertions_dict = {}
    for dialect, ins in insertions.items():
        insertions_dict[dialect] = dict(ins)
    
    return {
        "error_patterns": error_patterns_dict,
        "substitutions": substitutions_dict,
        "deletions": deletions_dict,
        "insertions": insertions_dict
    }

def build_confusion_matrix(error_analysis, min_occurrences=5):
    """
    Build confusion matrix for common errors
    
    Args:
        error_analysis: Output from analyze_error_patterns
        min_occurrences: Minimum occurrences to include in confusion matrix
        
    Returns:
        confusion_dict: Dictionary mapping misrecognized words to corrections
    """
    confusion_dict = {}
    
    # Process substitutions
    for dialect, substitutions in error_analysis["substitutions"].items():
        for (from_word, to_word), count in substitutions.items():
            if count >= min_occurrences:
                if from_word not in confusion_dict:
                    confusion_dict[from_word] = {}
                
                if to_word not in confusion_dict[from_word]:
                    confusion_dict[from_word][to_word] = 0
                
                confusion_dict[from_word][to_word] += count
    
    return confusion_dict

def visualize_common_errors(error_analysis, top_n=20, output_path=None):
    """
    Visualize most common errors
    
    Args:
        error_analysis: Output from analyze_error_patterns
        top_n: Number of top errors to display
        output_path: Path to save visualization
        
    Returns:
        fig: Matplotlib figure
    """
    # Flatten errors across dialects
    all_substitutions = defaultdict(int)
    for dialect, subs in error_analysis["substitutions"].items():
        for pair, count in subs.items():
            all_substitutions[pair] += count
    
    # Sort by frequency
    sorted_errors = sorted(all_substitutions.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Create dataframe for visualization
    error_df = pd.DataFrame([
        {"From": from_word, "To": to_word, "Count": count}
        for (from_word, to_word), count in sorted_errors
    ])
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Count", y="From", data=error_df, hue="To", dodge=False)
    plt.title(f"Top {top_n} Word Substitution Errors", fontsize=14)
    plt.xlabel("Frequency", fontsize=12)
    plt.ylabel("Misrecognized Word", fontsize=12)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()