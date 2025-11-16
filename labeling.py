import pandas as pd
import re

def improved_label_gambling_comments(csv_file_path, output_file_path=None):
    """
    Melabeli komentar judi dengan algoritma yang lebih akurat
    """
    
    # Baca file
    try:
        df = pd.read_csv(csv_file_path, encoding='latin-1', engine='python', on_bad_lines='skip')
    except:
        df = pd.read_csv(csv_file_path, encoding='latin-1', error_bad_lines=False)
    
    print(f"File berhasil dibaca. Total baris: {len(df)}")
    
    # Kata kunci judi yang lebih spesifik
    gambling_platforms = [
        # Platform/situs judi
        'lazadatoto', 'pstoto99', 'mini1221', 'kyt4d', 'togel62', 'bukit4d', 
        'pelatih4d', 'sajak4d', 'sendal4d', 'gelora4d', 'mona4d', 'sgi88',
        'garudahoki', 'arwanatoto', 'plazabola', 'insan4d', 'berkahslot',
        'pulauwin', 'paste4d', 'kurirslot', 'traxearn', 'biptrade', 'garuda69', 'phoenix638',
        'mbak4d2', 'gaspol 168', 'mega177', 'upahslot', 'sikat88', 'pesiar88', 'grok681h', 'timo4d',
        'bet4d', 'dibet4d', 'denyut69', 'squad777', 'pr0be 855', 'pr0be', 'spin68', 'pulauwin', 'pulau777' 
    ]
    
    gambling_terms = [
        # Istilah teknis perjudian
        'depo', 'deposit', 'wd', 'withdraw', 'modal', 'saldo', 'maxwin', 'scatter',
        'jepe', 'gacor', 'hoki', 'jackpot', 'bet', 'taruhan', 'slot', 'togel',
        'casino', 'poker', 'bandar', 'agen', 'bonus', 'freechip', 'turnover',
        'rollingan', 'cashback', 'rebate', 'situs', 'platform', 'permainan uang',
        'investasi', 'profit', 'cuan', 'pasang', 'wede', 'jp', 'pragmatic', 'rtp'
    ]
    
    def is_gambling_comment(text):
        """
        Fungsi yang lebih akurat untuk mendeteksi komentar judi
        """
        if not isinstance(text, str):
            return 0
        
        text_lower = text.lower()
        
        # 1. Cek platform judi - jika ada, langsung label 1
        for platform in gambling_platforms:
            if platform in text_lower:
                return 1
        
        # 2. Cek kombinasi istilah judi + konteks uang
        gambling_terms_found = []
        for term in gambling_terms:
            if term in text_lower:
                gambling_terms_found.append(term)
        
        # Jika ada istilah judi, cek konteksnya
        if gambling_terms_found:
            # Pattern untuk nominal uang
            money_patterns = [
                r'\d+[km]',  # 100k, 50m
                r'\d+\s*(rb|ribu|jt|juta|k|m)',  # 100 rb, 50 juta
                r'rp\s*\d+',  # Rp 100000
                r'\d+\s*(rupiah|perak)'  # 100 ribu rupiah
            ]
            
            # Cek apakah ada pattern uang
            has_money = any(re.search(pattern, text_lower) for pattern in money_patterns)
            
            # Kata-kata yang bisa menyebabkan false positive
            false_positive_triggers = ['game', 'main', 'mlbb', 'mobile legend', 'turnamen', 'tournament']
            has_false_positive = any(trigger in text_lower for trigger in false_positive_triggers)
            
            # Jika ada uang DAN tidak ada konteks game, label sebagai judi
            if has_money and not has_false_positive:
                return 1
            
            # Jika ada kombinasi istilah judi yang spesifik
            strong_combinations = [
                ['depo', 'wd'], ['modal', 'wd'], ['saldo', 'wd'],
                ['maxwin', 'depo'], ['gacor', 'depo'], ['jepe', 'modal'], ['rtp','slot']
            ]
            
            for combo in strong_combinations:
                if all(term in gambling_terms_found for term in combo):
                    return 1
        
        return 0
    
    # Terapkan labeling
    print("Melabeli komentar dengan algoritma improved...")
    df['target'] = df['cleaned_comment_text'].apply(is_gambling_comment)
    
    # Statistik
    total = len(df)
    judi = df['target'].sum()
    normal = total - judi
    
    print(f"\n=== HASIL LABELING YANG LEBIH AKURAT ===")
    print(f"Total komentar: {total}")
    print(f"Komentar judi (target=1): {judi} ({judi/total*100:.2f}%)")
    print(f"Komentar normal (target=0): {normal} ({normal/total*100:.2f}%)")
    
    # Test case untuk komentar yang bermasalah
    test_cases = [
        "sonic cuma divisi ml divisi laen kalah",
        "lazadatoto barusan wd 3 ikat modal 100 aja",
        "main game mlbb kalah terus",
        "depo 50k wd 500k di garudahoki",
        "turnamen mlbb seru banget"
    ]
    
    print(f"\n=== TEST CASE ===")
    for test in test_cases:
        result = is_gambling_comment(test)
        print(f"'{test}' -> {result}")
    
    # Simpan hasil
    if output_file_path:
        df.to_csv(output_file_path, index=False, encoding='utf-8')
        print(f"\nFile disimpan sebagai: {output_file_path}")
    
    return df

# Analisis false positive
def analyze_false_positives(df):
    """
    Menganalisis komentar yang mungkin salah label
    """
    print(f"\n=== ANALISIS FALSE POSITIVE ===")
    
    # Cari komentar yang mengandung kata "kalah" tapi bukan judi
    kalah_comments = df[df['cleaned_comment_text'].str.contains('kalah', na=False) & (df['target'] == 1)]
    
    print(f"Komentar dengan kata 'kalah' yang dilabeli judi: {len(kalah_comments)}")
    for idx, row in kalah_comments.head(10).iterrows():
        print(f"- {row['cleaned_comment_text']}")
    
    # Cari komentar yang mengandung kata "game" tapi dilabeli judi
    game_comments = df[df['cleaned_comment_text'].str.contains('game', na=False) & (df['target'] == 1)]
    print(f"\nKomentar dengan kata 'game' yang dilabeli judi: {len(game_comments)}")

# Jalankan program
if __name__ == "__main__":
    input_file = "cleaned_comments.csv"
    output_file = "labeled_comments.csv"
    
    df = improved_label_gambling_comments(input_file, output_file)
    
    # Analisis false positive
    analyze_false_positives(df)
    
    # Tampilkan preview
    print(f"\n=== PREVIEW HASIL ===")
    print(df[['cleaned_comment_text', 'target']].head(15))
