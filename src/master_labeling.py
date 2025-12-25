
import os
import sys
# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import tensorflow as tf
import re
import os
import unicodedata
import sys
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION
# ==========================================
# Resolve paths relative to this script file to allow running from anywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to root, then into datasets
DATASET_DIR = os.path.join(BASE_DIR, '..', 'datasets')

INPUT_FILE = os.path.join(DATASET_DIR, 'comments_from_scraping_new.csv')
OUTPUT_FILE = os.path.join(DATASET_DIR, 'comments_labeled_final.csv')

# Import classify from original script to ensure identical baseline
try:
    # Add current dir to path for local imports
    sys.path.append(BASE_DIR)
    from utils.auto_labeling import classify, normalize_text
except ImportError as e:
    print(f"Error: auto_labeling.py not found or failed to import. {e}")
    sys.exit(1)

# Helper for Leetspeak (used in expert patch if needed, but auto_labeling handles extensive stuff)
# We will trust auto_labeling.classify for the weak label.

# ... (Expert Patterns remain same)

# ... (Main Pipeline updates)

# EXPERT PATCH PATTERNS (Checking Obfuscation & Specific Sites)
EXPERT_SITE_PATTERNS = [
    r'MINI\d{3,}',   # MINI1221
    r'MBAK[A-Z0-9]*\d+', # Catch MBAK4D, MBAKD2, etc
    r'LIGAMANSION\d*',
    r'DORA\d{2,}',   # DORA77
    r'KYT\d+',       # KYT4D
    r'DOGRA\d+',
    r'PASTE(L?)\d+',
    r'KURIRSLOT',
    r'ARWANA\w+',
    r'PLAZABOLA',
    r'PROBET',
    r'PLAY\d+',      # PLAY777
    r'VIP\d+',       # VIP88
    r'WAYANG\d+',
    r'TOTAL\d+', 
    r'JOKER\d+',
    r'CROWN\d+',     # Require digits (Crown is common)
    r'ROYAL\d+',     # Require digits (Royal is common)
    r'WOKEBET',
    r'MAJOR\s*\d+',
    r'PESIAR\d+',    # Require digits (Pesiar is common meaning Cruise)
    r'SERU\d+',      # SERU69
    r'DAYAK\s*\d+',  # DAYAK 777
    r'TARGET\d+',    # Require digits (Target is common)
    r'KANGJP\d*',
    r'DOYAN\d+',     # Require digits
    r'DUO\s*GAMING',
    r'4RABET',
    r'AMBIL\d+',     # Require digits (Ambil is common)
    r'WE\s*TOGEL',
    r'SUPER\s*MONEY\d*',
    r'HOKI\d+',
    r'CIUM\d+',
    r'KOBE\d+',      # Require digits
    r'ALEXIS\d+',    # Require digits
    r'XUXU\d+',
    r'PULAU\d+',     # PULAU777, PULAUWIN (Ensure digits/context)
    r'PRIMBON\d+',   # PRIMBON178 (New)
    r'TIMO\d+',      # TIMO4D (New)
    r'GELORA\d+',    # Require digits
    r'LAUTAN\w+',    # Lautan = Ocean. "Lautan api". Risk. User exp: "LAUTANSL0T". 
                     # Change to LAUTAN\d+ or LAUTANSL.
    r'JOS\d+',
    r'KOPI\d+',      # Kopi = Coffee. "Kopi susu". User exp: "Kopi77". Require digits. OK.
    r'PSTOTO\d*',
    r'SEKALI\d+',
    r'MANUT\d+',
    r'CIDUK\d+|CIDUK[-]?JP', # User exp: Ciduk-JP.
    r'CUKONG\d+',
    r'GROK\d+',      # Require digits
    r'SAMBAR\d+|SAMBARJP',
    r'MONET\d+|MONET[-]?\d+', 
    r'OJOL\d+',
    r'GLOBAL\d+',    # Global is common. Require digits. OK.
    r'JEPOR\d+|JEPOR[-]?\d+',
    r'REKOR\d+|REKOR[-]?\d+', # Rekor = Record. Require digits.
    r'TOHIR\d+',     # Tohir is name. Require digits.
    r'PLAYTOTO\d*',
    r'AREA\s*MAIN',
    r'KAWASAN\s*TEMPUR',
    r'ARENA\s*TEMPUR',
    r'AUTO\s*TURBO'
]

# URL Handling
URL_PATTERN = r'(https?://\S+|www\.\S+)'
SAFE_DOMAINS = [r'youtube\.com', r'youtu\.be', r'google\.com', r'facebook\.com', r'instagram\.com']

# Anti-gambling keywords for heuristic correction of TRAINING data
ANTI_KEYWORDS = [
    'berhenti', 'stop', 'jijik', 'tobat', 'hancur', 'rugi', 'penipuan', 'tipu', 
    'bohong', 'haram', 'dosa', 'setan', 'iblis', 'jauhi', 'jangan', 'korban',
    'miskin', 'melarat', 'habis', 'kalah', ' rungkad', 'gembel'
]

def is_likely_anti_gambling(text):
    text = str(text).lower()
    if any(k in text for k in ANTI_KEYWORDS):
        if "jangan ragu" in text or "jangan takut" in text or "jangan lupa" in text:
            return False
        return True
    return False

def check_expert_pattern(text):
    norm = normalize_text(text)
    
    # 1. Standard Check (NFKC)
    if any(re.search(p, norm, re.IGNORECASE) for p in EXPERT_SITE_PATTERNS):
        return True
        
    # 2. Aggressive Check (Remove ALL non-alphanumeric)
    # Handles "D U O", "[H][o]", "A_M_B_I_L"
    norm_aggressive = re.sub(r'[^a-z0-9]', '', norm)
    if any(re.search(p, norm_aggressive, re.IGNORECASE) for p in EXPERT_SITE_PATTERNS):
        return True
        
    # 3. Suspicious URLs
    # Label 1 if contains URL AND NOT in Safe List
    urls = re.findall(URL_PATTERN, str(text).lower()) # Use raw text for URLs
    for url in urls:
        is_safe = any(re.search(safe, url) for safe in SAFE_DOMAINS)
        if not is_safe:
            return True # Found a suspicious URL
            
    return False

def run_pipeline():
    print("--- 1. LOADING DATA ---")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
    df = pd.read_csv(INPUT_FILE)
    print(f"Total rows: {len(df)}")
    
    # Text Column Check
    if 'comment_text' not in df.columns and 'cleaned_comment_text' in df.columns:
        df['comment_text'] = df['cleaned_comment_text']
        
    # --- DEDUPLICATION STEP ---
    initial_count = len(df)
    print(f"Initial count: {initial_count}")
    df.drop_duplicates(subset=['comment_text'], keep='first', inplace=True)
    dedup_count = len(df)
    print(f"Removed {initial_count - dedup_count} duplicates.")
    print(f"Count after deduplication: {dedup_count}")
    # --------------------------

    print("\n--- 2. INITIAL REGEX LABELING (WEAK) ---")
    df['weak_label'] = df['comment_text'].apply(classify)
    print(f"Weak Judol Count: {df['weak_label'].sum()}")
    
    print("\n--- 3. HEURISTIC CLEANING (ANTI-JUDOL) ---")
    # Prepare training labels: exact copy of weak labels, but flipped to 0 if Anti-Judol
    df['training_label'] = df['weak_label']
    
    # Preprocess for checking
    # Note: df['clean_text'] is used for AI input
    df['clean_text'] = df['comment_text'].apply(normalize_text)
    
    mask_anti = df['clean_text'].apply(is_likely_anti_gambling)
    corrected_count = df.loc[mask_anti & (df['weak_label'] == 1)].shape[0]
    df.loc[mask_anti & (df['weak_label'] == 1), 'training_label'] = 0
    print(f"Corrected {corrected_count} likely false positives (Anti-Judol) for training.")
    
    print("\n--- 4. AI MODEL TRAINING (TENSORFLOW) ---")
    # Features & Targets
    X_text = df['clean_text'].values
    y = df['training_label'].values # Train on CLEANED labels
    
    # Parameters
    max_features = 20000
    sequence_length = 50
    embedding_dim = 64
    
    # Vectorization
    vectorize_layer = TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)
    vectorize_layer.adapt(X_text)
    
    # Model
    model = Sequential([
        vectorize_layer,
        Embedding(max_features + 1, embedding_dim),
        GlobalAveragePooling1D(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X_text, y, test_size=0.2, random_state=42)
    
    # Train
    validation_split = 0.2
    epochs = 8 # Sufficient for this task
    batch_size = 128
    
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    
    print("Training started...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    print("\n--- 5. AI PREDICTION ---")
    y_pred_proba = model.predict(X_text, batch_size=256).flatten()
    df['ai_prob'] = y_pred_proba
    df['ai_label'] = (y_pred_proba >= 0.5).astype(int)
    print(f"AI Judol Count: {df['ai_label'].sum()}")
    
    print("\n--- 6. FINAL LABELING (COMBINED) ---")
    # Apply expert pattern check
    mask_expert = df['comment_text'].apply(check_expert_pattern)
    
    # Final label logic (combining Regex + AI + Expert):
    # - Regex (weak_label) sudah di-tune dengan baik
    # - AI memberikan confidence score
    # - Expert pattern untuk catch yang missed
    #
    # Rules:
    # 1. Regex=1 AND AI>=0.3 -> label 1 (high confidence judol)
    # 2. Regex=1 AND AI<0.3 -> label 1 (trust regex, AI might be wrong)
    # 3. Regex=0 AND AI>=0.6 -> label 1 (AI confident, regex might miss)
    # 4. Expert pattern -> label 1
    # 5. Anti-gambling -> label 0 (override all)
    
    df['final_label'] = 0
    
    # Rule 1 & 2: Regex says 1 -> final = 1 (trust regex)
    df.loc[df['weak_label'] == 1, 'final_label'] = 1
    
    # Rule 3: AI >= 0.6 (confident) -> final = 1
    df.loc[df['ai_prob'] >= 0.6, 'final_label'] = 1
    
    # Rule 4: Expert pattern -> final = 1
    df.loc[mask_expert, 'final_label'] = 1
    
    # Rule 5: Anti-gambling override -> final = 0
    mask_anti = df['clean_text'].apply(is_likely_anti_gambling)
    df.loc[mask_anti, 'final_label'] = 0
    
    # Summary
    print(f"Final Label Summary:")
    print(f"  Regex (weak_label)=1: {df['weak_label'].sum()}")
    print(f"  AI >= 0.6: {(df['ai_prob'] >= 0.6).sum()}")
    print(f"  Expert Pattern: {mask_expert.sum()}")
    print(f"  Anti-gambling (override): {mask_anti.sum()}")
    print(f"FINAL JUDOL COUNT: {df['final_label'].sum()}")
    
    # Save
    df['label'] = df['final_label']
    cols_to_drop = ['weak_label', 'training_label', 'clean_text', 'ai_label', 'final_label']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved final labeled dataset to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_pipeline()
