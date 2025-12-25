#!/usr/bin/env python3
"""
Zero-Shot Classification for Judol Comment Detection
Uses transformers pipeline with facebook/bart-large-mnli model.

Usage:
    python zeroshot_labeling.py                    # Process all comments
    python zeroshot_labeling.py --sample 100       # Process sample of 100
    python zeroshot_labeling.py --test             # Run test on sample comments
"""

import os
import sys
import pandas as pd
import argparse
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '..', 'datasets')
INPUT_FILE = os.path.join(DATASET_DIR, 'comments_labeled_final.csv')
OUTPUT_FILE = os.path.join(DATASET_DIR, 'comments_zeroshot_labeled.csv')

# Candidate labels for zero-shot classification
CANDIDATE_LABELS = [
    "promosi judi online",      # Gambling promotion
    "komentar biasa"            # Normal comment
]

# Alternative labels (dapat diubah sesuai kebutuhan)
CANDIDATE_LABELS_EN = [
    "gambling promotion",
    "normal comment"
]


def load_classifier(device=0):
    """Load the zero-shot classification pipeline."""
    try:
        from transformers import pipeline
        print("Loading BART model for zero-shot classification...")
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device  # 0 for GPU, -1 for CPU
        )
        print("Model loaded successfully!")
        return classifier
    except ImportError:
        print("Error: transformers library not installed.")
        print("Install with: pip install transformers torch")
        sys.exit(1)


def classify_comment(classifier, text, labels=None, threshold=0.5):
    """
    Classify a single comment using zero-shot classification.
    
    Args:
        classifier: The zero-shot classification pipeline
        text: Comment text to classify
        labels: Candidate labels (default: CANDIDATE_LABELS)
        threshold: Confidence threshold for judol classification
        
    Returns:
        tuple: (label, confidence_judol, confidence_normal)
    """
    if labels is None:
        labels = CANDIDATE_LABELS
    
    if pd.isna(text) or str(text).strip() == "":
        return 0, 0.0, 1.0
    
    text = str(text)[:512]  # Truncate to avoid token limit
    
    try:
        result = classifier(text, labels, multi_label=True)
        scores = dict(zip(result['labels'], result['scores']))
        
        judol_score = scores.get(labels[0], 0.0)
        normal_score = scores.get(labels[1], 0.0)
        
        # Label as judol (1) if confidence > threshold
        label = 1 if judol_score > threshold else 0
        
        return label, judol_score, normal_score
    except Exception as e:
        print(f"Error classifying: {e}")
        return 0, 0.0, 1.0


def run_test(classifier):
    """Run test on sample comments."""
    print("\n=== TEST ZERO-SHOT CLASSIFICATION ===\n")
    
    test_comments = [
        ("Main di PULAU777 gacor banget!", 1),
        ("Video ini sangat bermanfaat, terima kasih!", 0),
        ("【LAZADATOTO】 WD lancar modal receh!", 1),
        ("Lagu band TOTO memang keren", 0),
        ("Jangan main judi, rugi!", 0),
        ("Coba aja di ARWANATOTO, pasti jepe!", 1),
        ("Film ini bagus banget, recommended!", 0),
        ("Modal 50k langsung JP 5jt di MONA4D", 1),
    ]
    
    correct = 0
    for text, expected in test_comments:
        label, judol_conf, normal_conf = classify_comment(classifier, text)
        status = "✓" if label == expected else "✗"
        if label == expected:
            correct += 1
        print(f"{status} [{label}] judol={judol_conf:.2f} normal={normal_conf:.2f}")
        print(f"   {text[:50]}...")
        print()
    
    print(f"\nAccuracy: {correct}/{len(test_comments)} ({correct/len(test_comments)*100:.1f}%)")


def process_dataset(classifier, sample_size=None, threshold=0.5):
    """Process the entire dataset with zero-shot classification."""
    print(f"\n=== PROCESSING DATASET ===")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
    
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df):,} comments")
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f"Using sample of {len(df):,} comments")
    
    # Process with progress bar
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        text = row['comment_text']
        label, judol_conf, normal_conf = classify_comment(
            classifier, text, threshold=threshold
        )
        results.append({
            'comment_text': text,
            'zeroshot_label': label,
            'zeroshot_judol_conf': judol_conf,
            'zeroshot_normal_conf': normal_conf,
            'original_label': row.get('label', None),
            'ai_prob': row.get('ai_prob', None)
        })
    
    result_df = pd.DataFrame(results)
    
    # Stats
    print(f"\n=== RESULTS ===")
    print(f"Total: {len(result_df):,}")
    print(f"Judol (zeroshot=1): {result_df['zeroshot_label'].sum():,}")
    print(f"Safe (zeroshot=0): {(result_df['zeroshot_label']==0).sum():,}")
    
    # Compare with original labels if available
    if 'original_label' in result_df.columns and result_df['original_label'].notna().any():
        agree = (result_df['zeroshot_label'] == result_df['original_label']).sum()
        print(f"\nAgreement with original labels: {agree:,} ({agree/len(result_df)*100:.1f}%)")
    
    # Save
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")


def main():
    parser = argparse.ArgumentParser(description='Zero-Shot Classification for Judol Detection')
    parser.add_argument('--test', action='store_true', help='Run test on sample comments')
    parser.add_argument('--sample', type=int, help='Process only N sample comments')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--labels', choices=['id', 'en'], default='id', 
                        help='Label language: id (Indonesian) or en (English)')
    
    args = parser.parse_args()
    
    # Set device
    device = -1 if args.cpu else 0
    
    # Load classifier
    classifier = load_classifier(device=device)
    
    if args.test:
        run_test(classifier)
    else:
        process_dataset(
            classifier, 
            sample_size=args.sample,
            threshold=args.threshold
        )


if __name__ == "__main__":
    main()
