import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import unicodedata
import argparse
from tqdm import tqdm
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Initialize tqdm for pandas
tqdm.pandas()

# ==========================================
# CONFIGURATION
# ==========================================
# Resolve paths relative to this script file to allow running from anywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to root, then into datasets
DATASET_DIR = os.path.join(BASE_DIR, '..', 'datasets')

DEFAULT_INPUT_FILE = os.path.join(DATASET_DIR, 'comments_from_scraping_new.csv')
DEFAULT_OUTPUT_FILE = os.path.join(DATASET_DIR, 'comments_labeled_final.csv')

# ==========================================
# PART 1: AUTO LABELING LOGIC (MERGED)
# ==========================================
# This section contains logic previously in utils/auto_labeling.py
# ==========================================

# Kata kunci judol (lebih spesifik)
JUDOL_KEYWORDS = [
    # Istilah slot/judi - SPESIFIK
    'slot', 'gacor', 'maxwin', 'scatter', 'sceter', 'scater', 'jackpot', 
    'pragmatic', 'pgsoft', 'pg soft', 'habanero', 'spadegaming', 'joker123', 
    'rtp live', 'bocoran slot', 'freespin', 'puteran',
    # Istilah transaksi judol
    'withdraw', 'depo ', 'modal receh', 'cuan',
    'new member', 'newmember',
    # Nama situs judol (lebih spesifik) - xxxTOTO akan dideteksi by has_site_pattern
    'sbobet',
    # Slang JP/jackpot
    'jepi', 'jpnya', 'jepee', 'jekpot', 'jekpod',
    # Note: 'toto','togel','hoki' dihapus karena bisa berarti hal lain (band, nama orang, gaming)
]

# Frasa judol
JUDOL_PHRASES = [
    'cari di google', 'search di google', 'ketik di google',
    'wd ', 'wd gede', 'wd lancar', 'modal receh', 'modal kecil',
    'modal 10k', 'modal 20k', 'modal 25k', 'modal 30k', 'modal 35k', 
    'modal 50k', 'modal 100k', 'x500', 'x1000', 'x5000',
    'situs terpercaya', 'situs resmi', 'link resmi',
    'auto wd', 'auto jp', 'auto cuan', 'auto maxwin',
    'dijamin wd', 'pasti bayar', 'pasti wd', 'hoki hari ini', 
    'gacor hari ini', 'jp hari ini', 'mabar slot',  # 'main di' dihapus - terlalu umum
    'gacor parah', 'jackpot besar', 'main togel', 'suka togel', 'main slot',
    'salam hoki', 'salam jp', 'salam cuan', 'salam gacor',
    # Frasa promosi
    'baru join', 'baru daftar', 'baru gabung', 'member baru',
    'langsung menang', 'langsung jp', 'langsung wd', 'langsung dapat',
    'scatter turun', 'scatter hitam', 'scatter bertubi', 'dikasih maxwin',
    'nikmati bonus', 'bonus pertama', 'bonus new member',
    # Pola uang (klaim menang)
    '1jt', '2jt', '3jt', '5jt', '10jt', '15jt', '20jt', '25jt', '30jt', 
    '35jt', '50jt', '100jt', '200jt', '500jt',
    '1 juta', '2 juta', '5 juta', '10 juta', '20 juta', '50 juta', '100 juta',
    'ratusan juta', 'puluhan juta', 'jutaan', 'jt modal',
]

JUDOL_EMOJIS = ['🎰', '💰', '💵', '💸', '🎲', '💎']

# Mapping unicode fancy letters ke ASCII
def build_unicode_map():
    """Build mapping from fancy unicode to ASCII letters."""
    mapping = {}
    
    # Mathematical Bold (𝐀-𝐙, 𝐚-𝐳)
    for i, c in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        mapping[chr(0x1D400 + i)] = c.lower()
    for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
        mapping[chr(0x1D41A + i)] = c
    
    # Mathematical Italic
    for i, c in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        mapping[chr(0x1D434 + i)] = c.lower()
    for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
        mapping[chr(0x1D44E + i)] = c
    
    # Mathematical Bold Italic
    for i, c in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        mapping[chr(0x1D468 + i)] = c.lower()
    for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
        mapping[chr(0x1D482 + i)] = c
    
    # Mathematical Script
    for i, c in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        mapping[chr(0x1D49C + i)] = c.lower()
    for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
        mapping[chr(0x1D4B6 + i)] = c
    
    # Mathematical Sans-Serif Bold
    for i, c in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        mapping[chr(0x1D5D4 + i)] = c.lower()
    for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
        mapping[chr(0x1D5EE + i)] = c
    
    # Mathematical Sans-Serif Bold Digits (𝟬-𝟵)
    for i, c in enumerate('0123456789'):
        mapping[chr(0x1D7EC + i)] = c
    
    # Mathematical Bold Digits (𝟎-𝟗)
    for i, c in enumerate('0123456789'):
        mapping[chr(0x1D7CE + i)] = c
    
    # Mathematical Monospace
    for i, c in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        mapping[chr(0x1D670 + i)] = c.lower()
    for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
        mapping[chr(0x1D68A + i)] = c
    
    # Mathematical Fraktur (U+1D504-U+1D537) - often used in spam
    fraktur_upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    fraktur_start = 0x1D504
    for i, c in enumerate(fraktur_upper):
        mapping[chr(fraktur_start + i)] = c.lower()
    # Mathematical Fraktur lowercase (U+1D51E-U+1D537)
    for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
        mapping[chr(0x1D51E + i)] = c
    
    # Mathematical Double-Struck (U+1D538 - U+1D56B)
    # Exceptions: C, H, N, P, Q, R, Z are in BMP (U+2102 etc) - handled by NFKC usually, but adding for completeness
    # The range 1D538 is 'A' (Double-Struck Capital A)
    # This range has gaps where characters are in BMP, but Python unicodedata handles gaps or we blindly map?
    # Better to rely on NFKC for standard ones, but let's check the range.
    # A=1D538, B=1D539, C=2102, D=1D53B...
    # Simple loop over A-Z using NFKC on the code point might be robust, but let's just add the contiguous blocks if any
    # Or just rely on NFKC which is called in normalize_text.
    # However, has_obfuscated_site_name uses UNICODE_MAP *before* NFKC?
    # Yes: normalize_text calls UNICODE_MAP then NFKC.
    # So if we want has_obfuscated_site_name to see clean ASCII, we should map them.
    # Although line 385 does upper() then line 216 does NFKC...
    # Wait, has_obfuscated_site_name logic:
    #   if char in UNICODE_MAP: append(mapped)...
    #   then ''.join()
    #   then NFKC? No, explicit has_obfuscated_site_name logic does NOT call NFKC on the `text_normalized` (line 385).
    #   It says `text_clean = re.sub(..., text_normalized)`.
    #   Wait, `text_normalized` in line 385 is just `join(normalized).upper()`.
    #   If Double-Struck chars are NOT in UNICODE_MAP, they remain as is.
    #   Then `text_clean = re.sub(r'[^A-Z0-9]', '', text_clean)` removes them!
    #   So yes, we MUST map Double-Struck explicitly.
    
    doublestruck_upper_start = 0x1D538
    for i, c in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        # Some are reserved/in BMP, checking validity isn't strictly needed if we just map the code point
        # But if the char is invalid/reserved it won't appear anyway.
        mapping[chr(doublestruck_upper_start + i)] = c.lower()
        
    doublestruck_lower_start = 0x1D552
    for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
        mapping[chr(doublestruck_lower_start + i)] = c
    
    # Regional Indicator Symbols (🇦-🇿)
    for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
        mapping[chr(0x1F1E6 + i)] = c
        
    # Circled Latin Capital Letters (Ⓐ-Ⓩ: U+24B6 - U+24CF)
    for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
        mapping[chr(0x24B6 + i)] = c
    # Circled Latin Small Letters (ⓐ-ⓩ: U+24D0 - U+24E9)
    for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
        mapping[chr(0x24D0 + i)] = c
        
    # Enclosed Alphanumeric Supplement
    # Circled A-Z (U+1F150 - U+1F169)
    for i, c in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        mapping[chr(0x1F150 + i)] = c.lower()
    # Squared A-Z (U+1F170 - U+1F189)
    for i, c in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        mapping[chr(0x1F170 + i)] = c.lower()
    
    # Greek letters often used
    # Updated omega to 'w' for leetspeak usage (pulauwin)
    greek_map = {'α': 'a', 'Α': 'a', 'β': 'b', 'Β': 'b', 'σ': 's', 'Σ': 's', 
                 'τ': 't', 'Τ': 't', 'δ': 'd', 'Δ': 'd', 'ω': 'w', 'Ω': 'w',
                 'ε': 'e', 'η': 'n', 'ι': 'i', 'κ': 'k', 'λ': 'l', 'μ': 'm',
                 'ν': 'n', 'ο': 'o', 'π': 'p', 'ρ': 'r', 'υ': 'u', 'φ': 'f',
                 'χ': 'x', 'ψ': 'ps', 'ζ': 'z', 'ά': 'a', 'έ': 'e', 'ή': 'n',
                 'ί': 'i', 'ό': 'o', 'ύ': 'u', 'ώ': 'o',
                 # Cyrillic homoglyphs (used to bypass filters)
                 'А': 'a', 'а': 'a', 'В': 'b', 'Е': 'e', 'е': 'e', 'К': 'k',
                 'М': 'm', 'Н': 'h', 'О': 'o', 'о': 'o', 'Р': 'p', 'р': 'p',
                 'С': 'c', 'с': 'c', 'Т': 't', 'т': 't', 'У': 'y', 'у': 'y',
                 'Х': 'x', 'х': 'x'}
    mapping.update(greek_map)
    
    # Common obfuscation (exclude . and * - handled by special char substitution)
    # Removing digit-to-letter mapping from here to preserve numbers in site patterns/money
    obfusc = {'@': 'a', '$': 's',
              '†': 't', '҉': '', 'ñ': 'n', 'Ä': 'a', 'Ö': 'o', 'ǟ': 'a',
              'ʀ': 'r', 'ա': 'w', 'ռ': 'n', 'ȶ': 't', 'օ': 'o', 'ή': 'n',
              'Ŵ': 'w', 'Ⓞ': 'o', '丅': 't', 'ᗩ': 'a', 'ᗯ': 'w', '𝓣': 't',
              '𝓽': 't', '𝐍': 'n', '𝒶': 'a', '𝕒': 'a', '𝕨': 'w',
              # CJK/Kana Leetspeak (Updated for BATRE4D)
              '乃': 'b', 'ﾑ': 'a', 'ｲ': 't', '尺': 'r', '乇': 'e', 'り': 'd',
              'ℓ': 'l', 'ɴ': 'n', 'ρ': 'p',
              # Cleanup Keycaps & VS
              '\u20E3': '', '\uFE0F': '',
              # Small Capitals (ᴀ-ᴢ)
              'ᴀ': 'a', 'ʙ': 'b', 'ᴄ': 'c', 'ᴅ': 'd', 'ᴇ': 'e', 'ꜰ': 'f', 'ɢ': 'g', 
              'ʜ': 'h', 'ɪ': 'i', 'ᴊ': 'j', 'ᴋ': 'k', 'ʟ': 'l', 'ᴍ': 'm', 'ɴ': 'n', 
              'ᴏ': 'o', 'ᴘ': 'p', 'ꞯ': 'q', 'ʀ': 'r', 'ꜱ': 's', 'ᴛ': 't', 'ᴜ': 'u', 
              'ᴠ': 'v', 'ᴡ': 'w', 'x': 'x', 'ʏ': 'y', 'ᴢ': 'z'}
    mapping.update(obfusc)
    
    return mapping

UNICODE_MAP = build_unicode_map()

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    
    # Remove zero-width characters and other invisible obfuscation
    # \u200b: Zero width space, \u200c: ZWNJ, \u200d: ZWJ, \u200e/\u200f: LTR/RTL marks, \u2060: Word joiner, \ufeff: BOM
    text = re.sub(r'[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff]', '', text)
    
    # Apply unicode mapping
    result = []
    for char in text:
        if char in UNICODE_MAP:
            result.append(UNICODE_MAP[char])
        else:
            result.append(char)
    text = ''.join(result)
    
    # Standard normalization (NFKD to separate base chars from combining marks)
    text = unicodedata.normalize('NFKD', text)
    # Remove combining marks (diacritics)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    
    # Standard normalization (NFKC for folding mathematical bold/italic)
    text = unicodedata.normalize('NFKC', text)
    
    # Convert to lowercase and remove non-alphanumeric except spaces
    text = text.lower()
    
    # Replace special chars with space (not remove) to keep words separated
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def normalize_leetspeak(text):
    """Normalize leetspeak (numbers to letters) for keyword matching."""
    t = normalize_text(text)
    # Simple leetspeak mapping
    leetspeak = str.maketrans('0134578@$', 'oiyeastba')
    return t.translate(leetspeak)

def has_keywords(text):
    t = normalize_leetspeak(text)
    return any(k in t for k in JUDOL_KEYWORDS)

def has_phrases(text):
    t = normalize_leetspeak(text)
    # Exclude subscriber/viewer context
    subscriber_context = ['subs', 'subscriber', 'views', 'penonton', 'followers', 'like']
    if any(w in t for w in subscriber_context):
        return False
    return any(p in t for p in JUDOL_PHRASES)

def has_unicode_brackets(text):
    if pd.isna(text):
        return False
    # Only specific unicode brackets used in judol spam (NOT common emojis)
    brackets = ['【', '】', '〖', '〗', '『', '』', '「', '」', '꧁', '꧂']
    text_str = str(text)
    # If ANY of these specific spam brackets exist, it's likely spam
    return any(b in text_str for b in brackets)

# Pre-compile exclusion pattern
EXCLUDED_WORDS = ['totoan', 'gerrard', 'gerard', 'edward', 'forward', 'reward', 
                  'password', 'keyboard', 'record', 'ahmad', 'sad', 'bad',
                  'mad', 'dad', 'had', 'add', 'odd', 'god', 'red', 'bed', 'led',
                  'kid', 'bid', 'rid', 'mid', 'lid', 'old', 'cold', 'gold', 'bold',
                  'sold', 'told', 'hold', 'fold', 'mold', 'card', 'hard', 'yard',
                  'lord', 'word', 'bird', 'third', 'heard', 'world', 'child',
                  'friend', 'behind', 'mind', 'kind', 'find', 'blind', 'wind',
                  'sound', 'ground', 'found', 'bound', 'pound', 'alucard',
                  'legend', 'island', 'hand', 'band', 'land', 'sand', 'brand',
                  'stand', 'grand', 'demand', 'command', 'expand', 'load',
                  'road', 'head', 'dead', 'read', 'lead', 'bread', 'spread', 'thread',
                  'instead', 'ahead', 'overhead', 'upload', 'download', 'period', 'round',
                  'squad', 'end', 'send', 'spend', 'trend', 'friend', 'blend', 'defend',
                  'liquid', 'seagood', 'good', 'food', 'mood', 'blood', 'flood', 'wood', 'hood',
                  'could', 'would', 'should', 'need', 'feed', 'seed', 'speed', 'weed', 'indeed',
                  'build', 'field', 'yield', 'shield', 'wild', 'guild', 'valid', 'solid', 'stupid',
                  'rapid', 'vivid', 'acid', 'avoid', 'void', 'roid', 'roid', 'android', 'paid',
                  'sed', 'red', 'fled', 'bled', 'sped', 'shed', 'wed', 'ted', 'ned', 'led',
                  'c4d', 'c3d', 'r3d', 'b3d', 's3d', 'a4d',
                  'tenaga', 'olahraga', 'sinaga', 'kenanga', 'tetangga', 'mangga',
                  'sitoto', 'dewanya', 'dewata', 'dewi', 'dewa19', 'dewasa',
                  ]
EXCLUSION_PATTERN_COMPILED = re.compile(r'\b(' + '|'.join(EXCLUDED_WORDS) + r')\b')

# Pre-compile site patterns
SITE_PATTERNS = [
    # Nama situs judol yang spesifik (minimal 2 char prefix/suffix, no space before)
    r'\b(?!(?:si|fan|so|ka))[\w]{2,}toto\b', # Exclude 'sitoto' (Si Toto), 'fantoto', etc if needed. 
    # But better to use exclusion words. The exclusion list checks exact match.
    # regex \b\w{2,}toto\b matches 'sitoto'.
    # EXCLUSION_PATTERN_COMPILED.sub(' ', t) happens BEFORE regex.
    # So adding 'sitoto' to EXCLUDED_WORDS is sufficient.
    r'\b\w{2,}slot\b', r'\b\w{2,}togel\b',  # xxxTOTO, xxxSLOT, xxxTOGEL
    r'\btoto\w{2,}\b',  # TOTOxxx (totospin, totocc, dll) - minimal 2 char suffix
    r'\b[a-z]{2,}(?:4d|777|88)\b', r'\b[a-z]+\d+d\b', # xxx4D, xxx777, xxx88 (require 2+ letters prefix, exclude C4D)
    r'\b\w{2,}hoki\b', r'\b\w+naga\b', r'\bgaruda\s*hoki\b',
    r'\bga\s*ruda\s*ho\s*ki\b', r'\bruda\s*ho\s*ki\b',  # GA RUDA HO KI pattern
    # Pola situs dengan angka umum
    r'\b[a-z]{3,}(?:138|303|369|898|123|76|62|77|98)\b', # Require 3+ letters prefix (removed 69 - too common)
    # Situs spesifik dengan angka
    r'\bharta\d+\b', r'\bplaytoto\d+\b', r'\bbonus\w+\b', r'\bdewa\w{2,}\b',
    # Nama situs spesifik yang ditemukan
    r'\barwana', r'\bplazabola\b', r'\bmona\s*4d\b', 
    r'\bkino\w*d\b', r'\blazadatoto\b', r'\bshopetoto\b',
    r'\bpulauwin\b', r'\baero\w*\d+\b', r'\bvisi\s*4d\b',
    r'\bdora\s*\d+\b', r'\bambil\s*4d\b', r'\bxuxu\s*4d\b',
    r'\bgacorwin\w*\b', r'\bpusatwin\b', # Added pusatwin
    # Situs baru ditemukan dari analisis FN
    r'\bipototo\b', r'\bometoto\b', r'\bpstoto\d*\b', r'\btotospin\b',
    r'\bevostoto\b', r'\btotocc\b',
    # Situs dari analisis FN terbaru
    r'\bmini\d{3,}\b', r'\brtpwin\b', r'\bgopek\d+\b', r'\bbibit\d+\b',
    r'\bphoenix\d+\b', r'\bligamansion\d*\b', r'\bmbak\d+[a-z]*\b',
    r'\bdewadora\b', r'\bagustoto\b', r'\bmuraipoker\b', r'\bpaste\d*[a-z]*\b',
    r'\bkurirslot\b', r'\bweton\W*\d+\b', r'\bpp\s*ho\s*ki\b',
    # Situs dari analisis FN round 2
    r'\bzoom\d+\b', r'\bbandargaruda\b', r'\bsukajp\b', r'\bmamajitu\b',
    r'\bvhoki\b', r'\b5unsur\b', r'\bga\s*ru\s*da\s*ho\s*k[i]?\b',
    # Situs dari analisis FN round 3
    r'\bligakembar\b', r'\bfilabola\b', r'\bpulauwin\b',
    # Situs dari analisis FN round 4 (post-AI pipeline)
    r'\bdibet\d+[a-z]*\b', r'\bjuno\d+[a-z]*\b', r'\bpstoto\d+\b',
    r'\bdewa\s*dora\b', r'\bharta\d+\b',
    # Pola jp/jepi (Note: Pola uang dipindah ke has_judol_money)
    r'\bjepi\b', r'\bjepee\b', r'\bjekpot\b',
]
SITE_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in SITE_PATTERNS]

def has_site_pattern(text):
    t = normalize_text(text)
    
    # Exclude kata-kata yang bukan nama situs
    t = EXCLUSION_PATTERN_COMPILED.sub(' ', t)
    
    return any(p.search(t) for p in SITE_PATTERNS_COMPILED)

def has_judol_money(text):
    """Check for money patterns common in gambling promotion (10jt, 500jt)."""
    t = normalize_text(text)
    patterns = [
        r'\d{2,}jt\b', r'\d{2,}\s*juta\b',
    ]
    # Check exclusion for legitimate money context (gaji, tunjangan, harga, dll)
    legit_context = ['gaji', 'tunjangan', 'harga', 'bayar', 'hutang', 'utang', 'dpr', 'pejabat', 'korupsi',
                     # Casual mentions
                     'subscribe', 'subcribe', 'subscriber', 'views', 'penonton', 'followers',
                     # Business/wealth context
                     'kekayaan', 'triliun', 'miliar', 'saham', 'dollar', 'bisnis', 'perusahaan',
                     # Transfer/casual
                     'transfer', 'tf ', 'kirim', 'pinjam', 'nyadar', 'baru']
    if any(w in t for w in legit_context):
        return False
        
    return any(re.search(p, t, re.IGNORECASE) for p in patterns)

def count_judol_emojis(text):
    if pd.isna(text):
        return 0
    return sum(1 for e in JUDOL_EMOJIS if e in str(text))

def has_obfuscated_site_name(text):
    """Check for obfuscated site names with unicode/fancy characters."""
    if pd.isna(text):
        return False
    text_str = str(text)
    text_upper = text_str.upper()
    
    # Cek apakah ada unicode fancy characters (indikator obfuscation)
    # Cek apakah ada unicode fancy characters (indikator obfuscation)
    # > 0x1F00 (Symbols), 0x1D00-0x1DBF (Phonetic Extensions/Small Caps), 0x0300-0x036F (Combining Marks)
    # Removing filters (isalpha/isdigit) to ensure all symbols like Flags/Keycaps are checked
    has_fancy_unicode = any((ord(c) > 0x1F00 or (0x1D00 <= ord(c) <= 0x1DBF) or (0x0300 <= ord(c) <= 0x036F)) 
                            for c in text_str)
    
    # Jika tidak ada unicode fancy, skip pattern matching yang rawan FP
    if not has_fancy_unicode:
        # Hanya cek pattern spesifik yang jarang muncul di teks biasa
        text_clean = re.sub(r'[\s\u200b\u200c\u200d\ufeff]', '', text_upper)
        specific_patterns = [
            r'MINI\d{3,}',        # MINI1221
            r'SERU\d+',           # SERU69
            r'KYT\d+D?',          # KYT4D
            r'GARUDA.?HO.?KI',    # GARUDAHOKI
            r'PLAZA.?BOLA',       # PLAZABOLA
            # Specific site names added (no boundary check for robust match)
            'PULAUWIN', 'ARWANATOTO', 'KURIRSLOT', 'BATRE4D', 'BATRE4Y',
            'SUKU88' # Added SUKU88
        ]
        for pattern in specific_patterns:
            # Jika pattern adalah simple string (bukan regex), pakai 'in'
            if pattern.isalpha() and pattern.isupper(): 
                if pattern in text_clean:
                    return True
            # Jika regex
            elif re.search(pattern, text_clean, re.IGNORECASE):
                return True
        return False
    
    # Ada unicode fancy - normalize ke ASCII untuk pattern matching
    # Apply unicode map untuk normalize fancy chars ke ASCII
    normalized = []
    for char in text_str:
        if char in UNICODE_MAP:
            normalized.append(UNICODE_MAP[char])
        else:
            normalized.append(char)
    text_normalized = ''.join(normalized).upper()
    
    # Check if 'SLOT' appears as a standalone word (not part of a site name like xxxSLOT)
    # If it does, it's likely just the keyword, not obfuscation
    text_normalized_spaced = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text_normalized)  # Keep spaces
    if re.search(r'\bSLOT\b', text_normalized_spaced) and not re.search(r'\w{2,}SLOT', text_normalized_spaced):
        # "slot" is a standalone word, not an obfuscated site name
        # Still check for other patterns, but skip the xxxSLOT pattern
        pass  # Continue to check other patterns, but we'll handle xxxSLOT separately below
    
    # Remove spaces, zero-width chars, and non-alphanumeric
    text_clean = re.sub(r'[\s\u200b\u200c\u200d\ufeff]', '', text_normalized)
    text_clean = re.sub(r'[^A-Z0-9]', '', text_clean)  # Keep only alphanumeric
    
    # Pattern untuk nama situs dengan unicode obfuscation
    site_patterns = [
        r'\w{2,}4D\d*\b',     # xxx4D (BUKIT4D, etc)
        r'\w{2,}TOTO\b',      # xxxTOTO
        r'\w{2,}TOGEL\b',     # xxxTOGEL  
        r'T[O0]GEL\d+',       # T0GEL62, TOGEL99
        r'\w+WIN\b',          # xxxWIN (PULAUWIN, etc) - but check for common words
        r'\w{2,15}SLOT\b',     # xxxSLOT - limit prefix to max 15 chars to avoid matching entire sentences
        r'\w{3,}88\b',        # xxx88
        r'\w{3,}168\b',       # xxx168
        # Removed: xxx69, xxx77 (too common - "than 69", "dota77", etc)
        r'\w{2,}369\b',       # xxx369 (RP369, dll)
        r'\w{3,}898\b',       # xxx898 (GALAXY898, dll)
        r'\w{3,}789\b',       # xxx789
        r'\w{3,}123\b',       # xxx123
        r'\w{3,}138\b',       # xxx138
        r'\w{3,}777\b',       # xxx777
        r'\w{3,}888\b',       # xxx888
        # Specific 69/77 sites (not generic pattern)
        r'SERU69\b', r'DORA77\b', r'LESTI77\b', r'GIAT77[7]?\b',
    ]
    # Check if SLOT is standalone (to avoid matching "mending ... slot" as xxxSLOT)
    slot_is_standalone = re.search(r'\bSLOT\b', text_normalized_spaced) and not re.search(r'\w{2,}SLOT', text_normalized_spaced)
    # Check if TOTO is standalone (to avoid matching "Aah Toto" as xxxTOTO)
    toto_is_standalone = re.search(r'\bTOTO\b', text_normalized_spaced) and not re.search(r'\w{2,}TOTO', text_normalized_spaced)
    
    for pattern in site_patterns:
        # Skip xxxSLOT pattern if SLOT appears as a standalone word
        if slot_is_standalone and 'SLOT' in pattern and r'\w' in pattern:
            continue  # Skip this pattern for standalone SLOT
        # Skip xxxTOTO pattern if TOTO appears as a standalone word
        if toto_is_standalone and 'TOTO' in pattern and r'\w' in pattern:
            continue  # Skip this pattern for standalone TOTO
        if re.search(pattern, text_clean, re.IGNORECASE):
            return True
    
    # Specific site names
    site_names = ['PULAUWIN', 'ARWANA', 'GACORWIN', 'LAZADATOTO', 'SHOPETOTO',
                  'GARUDAHOKI', 'PLAZABOLA', 'ARWANATOTO', 'KYT4D',
                  'SENDAL4D', 'SAJAK4D', 'PELATIH4D', 'VISI4D', 'SOR76', 
                  'DOYANTOTO', 'PREMIERSLOT88', 'TOTOTAROT', 'LOHANSLOT',
                  'GIAT777', 'TOGEL62', 'SABDA4D', 'AERO88', 'BERKAH99',
                  'LESTI77', 'BUKIT4D', 'PUSATWIN']
    for name in site_names:
        if name in text_clean:
            return True
    unicode_patterns = [
        '𝐏𝐒𝐓𝐎𝐓𝐎', '𝙋', '𝐈𝐓', '𝐏𝐄', '𝐆',  # Mathematical Bold
        '𝕊𝔸𝕁𝔸𝕂', '𝔻',  # Double-struck
        '𝐑𝐔𝐃a', '𝐇𝐎 Ki', 'GA 𝐑',  # GARUDA HOKI parts
        'A҉R҉W҉A҉N҉A', 'ÄRWÄñÄ†Ö†Ö', 'aRŴ𝐚ή',  # ARWANA variations
        '𝒜𝑅',  # Script ARWANA
        '🇦​🇷​🇼​🇦​🇳​🇦​🇹​🇴​🇹​🇴',  # Regional indicators
        'P͟U͟L͟A͟U͟W͟I͟N', 'P͓̽U͓̽L͓̽A͓̽U',  # Strikethrough/combining
        '𝐒𝐄𝐍𝐃𝐀𝐋𝟒𝐃', '𝓟𝓤𝓛𝓐 𝓤𝓦𝓘𝓝', '𝐒𝐀𝐁𝐃𝐀𝟒𝐃',
    ]
    for p in unicode_patterns:
        if p in text_str:
            return True
    
    return False
    
def has_spaced_site_name(text):
    """Check for site names with spaces between letters (e.g., P U L A U W I N)."""
    if pd.isna(text):
        return False
    # Use full normalization first (handles maps, diacritics, etc)
    text_norm = normalize_text(text).upper()
    
    # Remove ALL spaces and non-alphanumeric for compressed check
    text_clean = re.sub(r'[^A-Z0-9]', '', text_norm)
    
    # List of site names to check in compressed form
    targets = [
        'PULAUWIN', 'ARWANA', 'GACORWIN', 'LAZADATOTO', 'SHOPETOTO',
        'GARUDAHOKI', 'PLAZABOLA', 'ARWANATOTO', 'KYT4D',
        'SENDAL4D', 'SAJAK4D', 'PELATIH4D', 'VISI4D', 'SOR76', 
        'DOYANTOTO', 'PREMIERSLOT88', 'TOTOTAROT', 'LOHANSLOT',
        'GIAT777', 'TOGEL62', 'SABDA4D', 'AERO88', 'BERKAH99',
        'LESTI77', 'BUKIT4D', 'PUSATWIN', 'PULAU777', 'PUIAU777',
        'PUIAU', 'PULAUTUJUH', 'DYANTOTO', 'DRWANA', 'DRWANATOTO', 'DRWANATSTO',
        # Situs baru dari analisis FN
        'MINI1221', 'RTPWIN', 'GOPEK500', 'BIBIT168', 'PHOENIX638',
        'LIGAMANSION', 'MBAK4D', 'DEWADORA', 'AGUSTOTO', 'MURAIPOKER',
        'PASTE4D', 'KURIRSLOT', 'PSTOTO99',
        # Situs dari analisis FN round 2
        'ZOOM555', 'BANDARGARUDA', 'SUKAJP', 'MAMAJITU', 'VHOKI', '5UNSUR'
    ]
    
    for t in targets:
        if t in text_clean:
            # Check if it's intentionally spaced/obfuscated
            # We look for the characters in sequence with optional garbage in between
            regex = r'.*'.join(re.escape(c) for c in t)
            if re.search(regex, text_norm, re.IGNORECASE):
                return True
    return False

def is_band_or_person_toto(text):
    """Check if 'toto' refers to band TOTO or person name (not gambling)."""
    if pd.isna(text):
        return False
    text_lower = str(text).lower()
    
    # If 'toto' is not in text, no need to check
    if 'toto' not in text_lower:
        return False
    
    # If text has strong gambling site patterns (xxxTOTO, etc), it's NOT person/band
    if has_site_pattern(text):
        return False
    
    # Check if this looks like an obfuscated site name with TOTO pattern
    # We only block if the obfuscation check finds a xxxTOTO pattern, not just any emoji
    text_normalized = normalize_text(text).upper()
    text_clean = re.sub(r'[^A-Z0-9]', '', text_normalized)
    # Check for xxxTOTO pattern (gambling site)
    if re.search(r'[A-Z]{2,}TOTO', text_clean):
        return False  # This is likely a gambling site (e.g., ARWANATOTO)
    # Context for BAND TOTO
    band_words = ['lagu', 'musik', 'album', 'rosanna', 'africa', 'hold you back', 
                  'dewa 19', 'dewa19', 'band', 'personil', 'drummer', 'guitarist',
                  'feat ', 'cover', 'mendengar', 'dengerin', 'listen', 'bermusik',
                  'channel', 'menarik', 'sah good', 'seagood', 'terbaik',
                  # Fan context
                  'penggemar', 'suka banget', 'salam dari', 'salam untuk']
    
    # Context for person name "Pak Toto", "si Toto", "Grazie Toto"
    person_words = ['pak toto', 'bapak toto', 'otto toto', 'toto wolff', 'kekayaan',
                    'data center', 'dci', 'pionir', 'praktisi', 'wawancara', 
                    'undang', 'podcast', 'diundang', 'beliau', 'teknologi',
                    'vendor', 'engineer', 'kerjain', 'motivasi',
                    # Indonesian casual references to person named Toto
                    'si toto', 'dasar si toto', 'ah toto', 'woua toto', 'thanks toto',
                    'grazie toto', 'takurany toto', 'hulu tah', 'ruksak toto',
                    'jebleh ku', 'beban sitoto', 'totoka', 'totoale',
                    # Sundanese casual (hulu ruksak = crazy head)
                    'hulu ruksak', 'tangkurak',
                    # YouTuber/streamer context
                    'ketua',
                    # F1 context
                    'f1', 'formula 1', 'mercedes', 'red bull',
                    # Italian greeting (F1 fans)
                    'grazie',
                    # Foreign language Toto (Moroccan rapper El Grande Toto)
                    'grande', 'el grande', 'morocco', 'rap', 'maroc', 'pablo',
                    # Game context (not gambling)
                    'game', 'minecraft', 'kang sine']
    
    # Esports team context - if these esports teams are mentioned with "toto",
    # it's likely a joke/meme, not gambling (e.g., "SONIC TOTO" = Onic team)
    esports_words = ['onic', 'evos', 'rrq', 'alter ego', 'bigetron', 'aura', 
                     'geek fam', 'nxl', 'aerowolf', 'sonic', 'mpl', 'esports',
                     'mobile legends', 'ml', 'm series', 'playoff']
    
    # Check for contexts
    has_band_context = any(w in text_lower for w in band_words)
    has_person_context = any(w in text_lower for w in person_words)
    has_esports_context = any(w in text_lower for w in esports_words)
    
    # If has toto AND (band/person/esports context), return True
    if 'toto' in text_lower:
        if has_band_context or has_person_context or has_esports_context:
            return True
        # Standalone "toto" in casual comments (not combined with site patterns)
        # Check if toto appears as a standalone word with casual context
        casual_patterns = [r'\bsi toto\b', r'\btoto\s*😂', r'\btoto\s*😭', 
                          r'\btoto\s*🔥', r'\btoto\s*👏', r'ah toto', 
                          r'aah toto', r'thanks toto', r'grazie toto',
                          r'suka.{0,10}toto', r'toto.{0,10}❤']
        for p in casual_patterns:
            if re.search(p, text_lower):
                return True
    
    return has_band_context or has_person_context

def is_anti_gambling_weak(text):
    """Check if comment is warning/criticism about gambling (not promotion).
    RENAMED from is_anti_gambling to avoid conflict with master_labeling strict check.
    """
    if pd.isna(text):
        return False
    text_lower = str(text).lower()
    
    # Jika ada nama situs judol, ini tetap promosi meskipun pakai kata warning
    # (banyak promosi judol yang pakai gaya "Awas ketagihan!" atau "Jangan main di xxx")
    if has_site_pattern(text):
        return False
    
    # Anti-gambling phrases (hanya berlaku jika TIDAK ada site pattern)
    anti_phrases = [
        'berhenti judi', 'stop judi', 'jangan judi', 'bahaya judi',
        'di tipu', 'ditipu', 'tipu', 'penipu', 'penipuan',
        'scam', 'scammer', 'bodong', 'palsu',
        'jangan main', 'jangan percaya', 'hati-hati', 'awas',
        'rugi', 'bangkrut', 'habis', 'korban', 'tobat',
        'belum dibayar', 'tidak dibayar', 'ga dibayar',
    ]
    
    for phrase in anti_phrases:
        if phrase in text_lower:
            return True
    return False

def classify(text):
    if pd.isna(text) or str(text).strip() == "":
        return 0
    
    text = str(text).lower()
    
    # Skip jika ini konteks band TOTO atau nama orang
    if is_band_or_person_toto(text):
        return 0
    
    score = 0
    if has_keywords(text): score += 2
    if has_phrases(text): score += 3
    if has_unicode_brackets(text): score += 3
    if has_site_pattern(text): score += 3  # Nama situs judol adalah indikator kuat
    if has_judol_money(text): score += 2   # Uang besar (+2, bukan +3)
    if has_obfuscated_site_name(text): score += 3  # Increased from 2 to 3 - strong indicator
    if has_spaced_site_name(text): score += 3 # Added for spaced-out names
    if count_judol_emojis(text) >= 2: score += 1
    
    # Special check for "mending ... daripada ... slot" comparison (not promo)
    # If text has "mending" and "slot", it's likely a comparison/joke, unless it has a link/site pattern
    text_normalized = normalize_text(text)  # Use normalized version for cleaner matching
    if 'slot' in text_normalized and 'mending' in text_normalized and not has_site_pattern(text) and not has_obfuscated_site_name(text):
         # check if "daripada" or "drpd" or "timbang" exists
         if any(w in text_normalized for w in ['daripada', 'drpd', 'timbang', 'ketimbang', 'dari pada']):
              score -= 4 # Downgrade heavily

    # Heuristic for game lore (Zeus + Kratos) - Not gambling
    if 'kratos' in text and ('zeus' in text or 'olympus' in text) and not has_site_pattern(text):
         score -= 4

    # Heuristic for "gacor" without gambling context - Not gambling
    # "gacor" in Indonesian can mean: 1) bird singing loudly, 2) slang for "great/performing well"
    # Only flag as gambling if there are other strong gambling indicators
    gacor_bird_context = ['burung', 'kicau', 'suara', 'lagu', 'nyanyian', 'karaoke', 
                          'murai', 'kenari', 'cucak', 'pleci', 'love bird', 'cendet']
    if 'gacor' in text:
        # If gacor appears with bird context, definitely not gambling
        if any(w in text for w in gacor_bird_context):
            score -= 4  # Likely bird/singing context
        # If gacor appears WITHOUT gambling site patterns/obfuscation, likely not gambling
        elif not has_site_pattern(text) and not has_obfuscated_site_name(text) and not has_phrases(text):
            # Check if this is really just casual usage (no slot, no maxwin, no money patterns)
            if 'slot' not in text and 'maxwin' not in text and not has_judol_money(text):
                score -= 2  # Downgrade - likely casual "gacor" usage

    # Reduce score for anti-gambling comments (sudah dicek di is_anti_gambling)
    # Using the renamed function here
    if is_anti_gambling_weak(text): score -= 4
    
    return 1 if score >= 3 else 0

# ==========================================
# PART 2: MASTER LABELING PIPELINE
# ==========================================

# EXPERT PATCH PATTERNS (Checking Obfuscation & Specific Sites)
EXPERT_SITE_PATTERNS = [
    r'MINI\d{3,}',   # MINI1221
    r'MBAK[A-Z0-9]{0,10}\d+', # Catch MBAK4D, MBAKD2, etc - Limit length to avoid greedy match
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
    r'LAUTAN(SLOT|TOTO|POKER|WIN|BET|\d+)',    # Lautan = Ocean. "Lautan api". Risk. User exp: "LAUTANSL0T". 
                     # Change to LAUTAN\d+ or LAUTANSL.
    # r'JOS\d+', # Removed common word
    r'KOPI(SLOT|TOTO|\d+)',      # Kopi = Coffee. "Kopi susu". User exp: "Kopi77". Require digits. OK.
    r'PSTOTO\d*',
    r'SEKALI\d+',
    r'MANUT\d+',
    r'CIDUK\d+|CIDUK[-]?JP', # User exp: Ciduk-JP.
    r'CUKONG\d+',
    # r'GROK\d+',      # Removed - Crypto spam, often not Judol specific.

    # PROBET was ID 1349 (Label 1 correct).
    r'DENYUT\d+',
    r'HOLYWIN',
    r'DOYAN\s*TOTO',
    r'D\s*U\s*O?\s*G\s*A\s*M\s*I\s*N\s*G', # Catch DUGAMING (missing O) matched as DU O? G..
    r'SAMBAR\d+|SAMBARJP',
    r'MONET\d+|MONET[-]?\d+', 
    r'OJOL\d+',
    r'GLOBAL\d+',    # Global is common. Require digits. OK.
    r'JEPOR\d+|JEPOR[-]?\d+',
    r'REKOR\d+|REKOR[-]?\d+', # Rekor = Record. Require digits.
    r'RP\d+',        # RP369
    r'TOHIR\d+',     # Tohir is name. Require digits.
    r'PLAYTOTO\d*',
    r'AREA\s*MAIN',
    r'KAWASAN\s*TEMPUR',
    r'ARENA\s*TEMPUR',
    r'AUTO\s*TURBO',
    r'TIKET\d+',      # TIKET200
]

# URL Handling
URL_PATTERN = r'(https?://\S+|www\.\S+)'
SAFE_DOMAINS = [r'youtube\.com', r'youtu\.be', r'google\.com', r'facebook\.com', r'instagram\.com']

# Anti-gambling keywords for heuristic correction of TRAINING data
# Only strong anti-gambling words - avoid words used in promotion tactics
ANTI_KEYWORDS_STRONG = [
    'berhenti main', 'stop judi', 'jijik', 'tobat', 'penipuan', 'tipu', 
    'bohong', 'haram', 'dosa', 'setan', 'iblis', 'jauhi', 'korban',
    'miskin', 'melarat', 'hancur', 'bangkrut', 'neraka', 'tobat', 'siksa'
]

# Promotion tactics that look like anti-gambling but are actually ads
PROMO_TACTICS = [
    'jangan bilang', 'gak nyuruh', 'awalnya takut', 'takut rungkad',
    'pernah rungkad', 'gak rugi', 'tidak rugi', 'tanpa rugi',
    'cape ditipu', 'cape rungkad', 'tempat jujur', 'situs jujur'
]

def is_likely_anti_gambling(text):
    """Check if comment is genuinely anti-gambling (not a promotion tactic)."""
    text = str(text).lower()
    
    # Check for promotion tactics first - these are NOT anti-gambling
    if any(tactic in text for tactic in PROMO_TACTICS):
        return False
    
    # Check for strong anti-gambling phrases
    if any(k in text for k in ANTI_KEYWORDS_STRONG):
        # But also check it's not combined with site promotion
        # But also check it's not combined with site promotion
        # Or negated (e.g. "bukan slot bohongan")
        if 'jangan ragu' in text or 'jangan takut' in text or 'jangan lupa' in text:
            return False
            
        # Check for negation of anti-keywords
        # "bukan ... bohong", "gak ... tipu"
        negations = ['bukan', 'ga', 'gak', 'tidak', 'gapernah', 'bkn']
        words = text.split()
        # Simple proximity check
        for i, word in enumerate(words):
            # Check if this word contains an anti-keyword
            if any(k in word for k in ANTI_KEYWORDS_STRONG):
                # Check 3 words before
                start = max(0, i-3)
                context = words[start:i]
                if any(neg in context for neg in negations):
                    return False # Negated anti-keyword -> likely Promo (e.g. "bukan tipu")
                    
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
    parser = argparse.ArgumentParser(description='Label gambling comments.')
    parser.add_argument('--input', default=DEFAULT_INPUT_FILE, help='Input CSV file')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_FILE, help='Output CSV file')
    args = parser.parse_args()

    print("--- 1. LOADING DATA ---")
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return
    df = pd.read_csv(args.input)
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
    print("Running regex classification...")
    df['weak_label'] = df['comment_text'].progress_apply(classify)
    print(f"Weak Judol Count: {df['weak_label'].sum()}")
    
    print("\n--- 3. HEURISTIC CLEANING (ANTI-JUDOL) ---")
    # Prepare training labels: exact copy of weak labels, but flipped to 0 if Anti-Judol
    df['training_label'] = df['weak_label']
    
    # Preprocess for checking
    # Note: df['clean_text'] is used for AI input
    print("Normalizing text...")
    df['clean_text'] = df['comment_text'].progress_apply(normalize_text)
    
    print("Checking for anti-gambling context...")
    mask_anti = df['clean_text'].progress_apply(is_likely_anti_gambling)
    corrected_count = df.loc[mask_anti & (df['weak_label'] == 1)].shape[0]
    df.loc[mask_anti & (df['weak_label'] == 1), 'training_label'] = 0
    print(f"Corrected {corrected_count} likely false positives (Anti-Judol) for training.")
    
    # NEW: Incorporate Expert Patterns into Training Labels
    # This matches the "fix everything" request: ensure AI learns patterns caught by expert rules (e.g. PASTE4D)
    print("Checking expert patterns for training data...")
    mask_expert = df['comment_text'].progress_apply(check_expert_pattern)
    expert_added_count = df.loc[mask_expert & (df['training_label'] == 0)].shape[0]
    df.loc[mask_expert, 'training_label'] = 1
    print(f"Added {expert_added_count} expert pattern labels to training data.")
    # Features & Targets
    X_text = df['clean_text'].values
    y = df['training_label'].values # Train on CLEANED labels
    
    # Parameters
    max_features = 20000
    sequence_length = 50
    embedding_dim = 64
    
    # Vectorization
    print("Adapting TextVectorization (this may take a moment)...")
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
    print("Predicting with AI model...")
    y_pred_proba = model.predict(X_text, batch_size=256).flatten()
    df['ai_prob'] = y_pred_proba
    df['ai_label'] = (y_pred_proba >= 0.5).astype(int)
    print(f"AI Judol Count: {df['ai_label'].sum()}")
    
    print("\n--- 6. FINAL LABELING (COMBINED) ---")
    # Apply expert pattern check
    print("Using pre-calculated expert patterns...")
    # mask_expert is already calculated above
    
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
    print("Final anti-gambling check...")
    mask_anti = df['clean_text'].progress_apply(is_likely_anti_gambling)
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
    
    df.to_csv(args.output, index=False)
    print(f"\nSaved final labeled dataset to: {args.output}")

if __name__ == "__main__":
    run_pipeline()
