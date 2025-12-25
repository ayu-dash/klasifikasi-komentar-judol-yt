"""
Script untuk melabeli komentar promosi judol secara otomatis.
Label: 1 = promosi judol, 0 = bukan promosi judol
"""

import pandas as pd
import re
import unicodedata

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
    return any(p in t for p in JUDOL_PHRASES)

def has_unicode_brackets(text):
    if pd.isna(text):
        return False
    # Only specific unicode brackets used in judol spam (NOT common emojis)
    brackets = ['【', '】', '〖', '〗', '『', '』', '「', '」', '꧁', '꧂']
    text_str = str(text)
    # If ANY of these specific spam brackets exist, it's likely spam
    return any(b in text_str for b in brackets)

def has_site_pattern(text):
    t = normalize_text(text)
    
    # Exclude kata-kata yang bukan nama situs (common words ending with d)
    # Only exclude exact word matches, not partial (e.g., 'squad' should not block 'squad78')
    excluded_words = ['totoan', 'gerrard', 'gerard', 'edward', 'forward', 'reward', 
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
                      'c4d', 'c3d', 'r3d', 'b3d', 's3d', 'a4d']  # Cinema 4D, Blender 3D, etc.
    # Use single compiled pattern for better performance
    exclusion_pattern = r'\b(' + '|'.join(excluded_words) + r')\b'
    t = re.sub(exclusion_pattern, ' ', t)
    
    patterns = [
        # Nama situs judol yang spesifik (minimal 2 char prefix/suffix, no space before)
        r'\b\w{2,}toto\b', r'\b\w{2,}slot\b', r'\b\w{2,}togel\b',  # xxxTOTO, xxxSLOT, xxxTOGEL
        r'\btoto\w{2,}\b',  # TOTOxxx (totospin, totocc, dll) - minimal 2 char suffix
        r'\b[a-z]{2,}(?:4d|777|88)\b', r'\b[a-z]+\d+d\b', # xxx4D, xxx777, xxx88 (require 2+ letters prefix, exclude C4D)
        # Nama situs dengan keyword hoki, naga, garuda
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
    return any(re.search(p, t, re.IGNORECASE) for p in patterns)

def has_judol_money(text):
    """Check for money patterns common in gambling promotion (10jt, 500jt)."""
    t = normalize_text(text)
    patterns = [
        r'\d{2,}jt\b', r'\d{2,}\s*juta\b',
    ]
    # Check exclusion for legitimate money context (gaji, tunjangan, harga, dll)
    legit_context = ['gaji', 'tunjangan', 'harga', 'bayar', 'hutang', 'utang', 'dpr', 'pejabat', 'korupsi']
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
            'PULAUWIN', 'ARWANATOTO', 'KURIRSLOT', 'BATRE4D', 'BATRE4Y'
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
        r'\w+SLOT\b',         # xxxSLOT
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
    for pattern in site_patterns:
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
        '𝐏𝐒𝐓𝐎𝐓𝐎', '𝙋����', '���𝐈𝐓', '𝐏𝐄�����', '𝐆�����',  # Mathematical Bold
        '𝕊𝔸𝕁𝔸𝕂', '𝔻',  # Double-struck
        '𝐑𝐔𝐃a', '𝐇𝐎 Ki', 'GA 𝐑',  # GARUDA HOKI parts
        'A҉R҉W҉A҉N҉A', 'ÄRWÄñÄ†Ö†Ö', 'aRŴ𝐚ή�',  # ARWANA variations
        '𝒜𝑅���',  # Script ARWANA
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
    
    # Jika ada pattern situs judol, bukan band/person
    if has_site_pattern(text):
        return False
    
    # Context for BAND TOTO
    band_words = ['lagu', 'musik', 'album', 'rosanna', 'africa', 'hold you back', 
                  'dewa 19', 'dewa19', 'band', 'personil', 'drummer', 'guitarist',
                  'feat ', 'cover', 'mendengar', 'dengerin', 'listen', 'bermusik',
                  'channel', 'menarik', 'sah good', 'seagood']
    
    # Context for person name "Pak Toto" (Otto Toto Sugiri)
    person_words = ['pak toto', 'bapak toto', 'otto toto', 'toto wolff', 'kekayaan',
                    'data center', 'dci', 'pionir', 'praktisi', 'wawancara', 
                    'undang', 'podcast', 'diundang', 'beliau', 'teknologi',
                    'vendor', 'engineer', 'kerjain', 'motivasi']
    
    # Cek apakah ada konteks band atau person
    has_band_context = any(w in text_lower for w in band_words)
    has_person_context = any(w in text_lower for w in person_words)
    
    return has_band_context or has_person_context

def is_anti_gambling(text):
    """Check if comment is warning/criticism about gambling (not promotion)."""
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
    
    # Reduce score for anti-gambling comments (sudah dicek di is_anti_gambling)
    if is_anti_gambling(text): score -= 4
    
    return 1 if score >= 3 else 0

if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv('../datasets/comments_from_scraping_new.csv')
    print(f"Total: {len(df):,} komentar")

    print("Labeling...")
    df['label'] = df['comment_text'].apply(classify)

    judol = df['label'].sum()
    print(f"\nHasil: Judol={judol:,} ({judol/len(df)*100:.2f}%), Non-Judol={len(df)-judol:,}")

    df.to_csv('comments_labeled.csv', index=False)
    print("Saved: comments_labeled.csv")
