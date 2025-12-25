import pandas as pd
import numpy as np
import sys

# Configure pandas to show full text
pd.set_option('display.max_colwidth', None)

def print_case(f, idx, row, reason):
    f.write(f"ID: {idx}\n")
    f.write(f"Text: {row['comment_text']}\n")
    f.write(f"Label: {row['label']} | AI Prob: {row['ai_prob']:.4f}\n")
    f.write(f"Reason: {reason}\n")
    f.write("-" * 50 + "\n")

def analyze():
    output_file = 'verification_report.txt'
    with open(output_file, 'w') as f:
        # Load the labeled dataset
        df = pd.read_csv('datasets/comments_labeled_final.csv')
        f.write(f"Total labeled comments: {len(df)}\n")
        f.write("-" * 30 + "\n")

        # --- 1. Potential False Positives (FP) Analysis ---
        f.write("\n\n[POTENTIAL FALSE POSITIVES (Label 1 but suspicious)]\n")
        f.write("====================================================\n")
        
        # A. Low AI Confidence (< 0.3)
        fp_candidates_low_ai = df[(df['label'] == 1) & (df['ai_prob'] < 0.3)]
        f.write(f"\n1. Low AI Confidence (< 0.3) but Labeled 1 (Likely Regex/Expert hallucination): {len(fp_candidates_low_ai)}\n")
        for idx, row in fp_candidates_low_ai.head(50).iterrows():
            print_case(f, idx, row, "Low AI Confidence - Check if Regex is too broad")

        # B. Anti-Gambling Context
        anti_keywords = ['jauhi', 'haram', 'dosa', 'bohong', 'tipu', 'miskin', 'hancur', 'stop', 'jangan']
        mask_anti_suspect = df['label'] == 1
        fp_candidates_anti = df[mask_anti_suspect & df['comment_text'].fillna('').astype(str).str.lower().apply(lambda x: any(k in x for k in anti_keywords))]
        f.write(f"\n2. Labeled 1 but contains Anti-Gambling keywords: {len(fp_candidates_anti)}\n")
        for idx, row in fp_candidates_anti.head(50).iterrows():
            print_case(f, idx, row, "Contains Anti-Gambling keywords - Check if context is missed")

        # --- 2. Potential False Negatives (FN) Analysis ---
        f.write("\n\n[POTENTIAL FALSE NEGATIVES (Label 0 but suspicious)]\n")
        f.write("====================================================\n")

        # A. High AI Confidence (> 0.6) but Labeled 0 (Must be Anti-Gambling Override)
        fn_candidates_override = df[(df['label'] == 0) & (df['ai_prob'] >= 0.6)]
        f.write(f"\n1. High AI Confidence (>= 0.6) but Labeled 0 (Anti-Gambling Override Triggered): {len(fn_candidates_override)}\n")
        for idx, row in fn_candidates_override.head(50).iterrows():
            print_case(f, idx, row, "Anti-Gambling Override - Verify if it is really anti-gambling or just tactics")

        # B. Near Miss (0.4 < AI < 0.6)
        fn_candidates_near_miss = df[(df['label'] == 0) & (df['ai_prob'] > 0.4) & (df['ai_prob'] < 0.6)]
        f.write(f"\n2. Near Miss (0.4 < AI < 0.6): {len(fn_candidates_near_miss)}\n")
        for idx, row in fn_candidates_near_miss.head(50).iterrows():
            print_case(f, idx, row, "Near Miss - Check if Regex missed a pattern")

        # C. Contains Gambling Keywords
        judol_keywords = ['gacor', 'slot', 'maxwin', 'jp', 'wd', 'depo', 'hoki', 'toto']
        mask_fn_suspect = df['label'] == 0
        fn_candidates_keywords = df[mask_fn_suspect & df['comment_text'].fillna('').astype(str).str.lower().apply(lambda x: any(k in x for k in judol_keywords))]
        # Exclude those we already listed in Near Miss/Override to avoid duplicates?
        # Simply list top ones
        f.write(f"\n3. Labeled 0 but contains Gambling keywords: {len(fn_candidates_keywords)}\n")
        for idx, row in fn_candidates_keywords.head(50).iterrows():
            print_case(f, idx, row, "Keyword Match - Check if valid usage or missed pattern")

    print(f"Report saved to {output_file}")

if __name__ == "__main__":
    analyze()
