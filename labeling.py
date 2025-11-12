import pandas as pd
import re
from tqdm import tqdm
from datetime import datetime
import os

class AdvancedJudiLabelingEngine:
    def __init__(self):
        # 1️⃣ STRONG BRANDS
        self.strong_brands = [
            'pesiar88', 'mbak4d2', 'g3d3', 'sor76', 'squad777', 'inigrok681h',
            'tapidora77', 'cobadora77', 'denyut69', 'gadaob4t', 'major189', 'starstruck',
            'tkp189', 'grokk681h', 'dora77', 'pakcoy', 'derr', 'sgi88','sg188','sgi808',
            'sgi888','sgi','sg','pstoto','pstoto99','pstoto88','pstoto77','psto',
            'arwanatoto','arwana','toto','pulauwin','pulau','win','lazadatoto','lazada4d',
            'lazada88','lazada77','lazada','visi4d','visi','jaya4d','mega4d','super4d',
            'ultra4d','prime4d','royal4d','king4d','queen4d','pro4d','max4d','gold4d',
            'silver4d','bronze4d','new4d','neo4d','alpha4d','beta4d','omega4d','delta4d',
            'city4d','metro4d','urban4d','capital4d','luck4d','fortune4d','rich4d','wealth4d',
            'star4d','moon4d','sun4d','galaxy4d','speed4d','quick4d','fast4d','instant4d',
            'insan4d','pandora4d','naga4d','hoki4d','paste4d','sendal4d','sekali4d',
            'togel62','garudahoki','garuda','hoki','dewapoker','pokermasa','masapoker',
            'karturapi','dominoqq','bandarqq','capsasusun','cemeonline','berkahslot','berkah',
            'slot','mini1221','mini12211','mini','mini88','mini77','mini99','mini55','mini33',
            'mini22','mini11','mini123','mini321','zeus','bibit168','bibit169','cilik168',
            'grok681h','tapigrok681h','samagrok681h','xrpgrok681h','hpgrok681h'
        ]

        # 2️⃣ PATTERN DETECTION
        self.patterns = [
            r'mini\d+', r'maxi\d+', r'mega\d+', r'super\d+', r'pro\d+',
            r'royal\d+', r'king\d+', r'queen\d+', r'\b\d{4,}\b',
            r'[a-z]{3,}\d{2,}', r'\w*slot\w*', r'\w*togel\w*',
            r'\w*judi\w*', r'\w*poker\w*', r'\w*casino\w*',
            r'[a-z]{3,}4d', r'[a-z]{3,}\s*4[dD]',
            r'\b[a-z]\d+[a-z]\d*\b',      # g3d3, s0r76
            r'\b[a-z]+\d+[a-z]+\d*\b',    # inigrok681h
            r'\b[a-z]+\d{3,}\b',          # squad777, denyut69
        ]

        # 3️⃣ DOMAIN KEYWORDS
        self.domain_keywords = [
            'togel','slot','judi','poker','casino','taruhan','betting','bola','scatter',
            'jackpot','menang','rezeki','untung','profit','bonus','main','eth','btc','bnb',
            'portofolio','buy','sell','pump','market'
        ]

    # ----------------- Helper Functions -----------------
    def detect_strong_brands(self, text):
        text_lower = text.lower()
        return [brand for brand in self.strong_brands if brand in text_lower]

    def detect_patterns(self, text):
        text_lower = text.lower()
        matches = []
        for pat in self.patterns:
            matches.extend(re.findall(pat, text_lower))
        return matches

    def calculate_domain_score(self, text):
        text_lower = text.lower()
        score = sum(2 for kw in self.domain_keywords if kw in text_lower)
        return score

    # ----------------- Main Labeling -----------------
    def advanced_labeling(self, text):
        text_lower = text.lower()
        details = {}

        # Strong brand
        brands = self.detect_strong_brands(text)
        if brands:
            details['strong_brands'] = brands
            return 'judol', 0.99, 'strong_brand_detected', details

        # Patterns
        patterns = self.detect_patterns(text)
        details['patterns'] = patterns

        # Domain score
        domain_score = self.calculate_domain_score(text)
        details['domain_score'] = domain_score

        # Decision logic
        if domain_score >= 2 or patterns:
            return 'judol', 0.9, 'pattern_or_domain', details

        return 'bukan', 0.7, 'insufficient_evidence', details

    # ----------------- Dataset Labeling -----------------
    def label_dataset(self, df, text_column='comment_text'):
        df['label_ultimate'] = 'bukan'
        df['label_confidence'] = 0.0
        df['label_category'] = 'unknown'
        df['strong_brands_detected'] = ''
        df['patterns_detected'] = ''
        df['labeling_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Labeling dataset"):
            text = row[text_column]
            label, conf, cat, details = self.advanced_labeling(text)

            df.at[idx, 'label_ultimate'] = label
            df.at[idx, 'label_confidence'] = conf
            df.at[idx, 'label_category'] = cat
            df.at[idx, 'strong_brands_detected'] = ', '.join(details.get('strong_brands', []))
            df.at[idx, 'patterns_detected'] = ', '.join(details.get('patterns', []))

        return df

    # ----------------- Analysis Function -----------------
    def analyze_labeling_results(self, df):
        print("\n" + "="*60)
        print("📊 ADVANCED LABELING RESULTS ANALYSIS")
        print("="*60 + "\n")

        # Distribusi label ultimate
        label_counts = df['label_ultimate'].value_counts()
        total = len(df)
        print("🎯 Distribusi Label Ultimate:")
        for label, count in label_counts.items():
            print(f"   {label:<10}: {count:>5} ({count/total*100:>5.2f}%)")
        print()

        # Distribusi kategori
        cat_counts = df['label_category'].value_counts()
        print("📈 Label Categories:")
        for cat, count in cat_counts.items():
            print(f"   {cat:<25}: {count:>5} ({count/total*100:>5.2f}%)")
        print()

        # Strong brands summary
        strong_brands_series = df['strong_brands_detected'].str.split(', ').explode()
        strong_brands_counts = strong_brands_series.value_counts()
        print(f"🔍 Strong Brands Detection: {strong_brands_counts.sum()} entries")
        for brand, count in strong_brands_counts.head(10).items():
            print(f"   {brand:<20}: {count}")
        print("="*60 + "\n")

    # ----------------- Save CSV -----------------
    def save_to_csv(self, df, filename=None, output_dir='output'):
        os.makedirs(output_dir, exist_ok=True)
        if not filename:
            filename = f'labeled_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        path = os.path.join(output_dir, filename)
        df.to_csv(path, index=False, encoding='utf-8')
        print(f"Hasil disimpan di {path}")
        return path

# ----------------- Contoh Penggunaan -----------------
if __name__ == "__main__":
    # Sample dataset
    sample_data = {
        'comment_text': [
            "sgi88 slot bonus 100% deposit 25rb saja",
            "hati-hati dengan judi online, saya bangkrut karenanya",
            "grok681h bakal running cycle ini udah ga perlu tanya",
            "jangan main judi online kenapa masih terdeteksi judol"
        ]
    }
    df = pd.read_csv('labeled_comments.csv')

    labeler = AdvancedJudiLabelingEngine()
    labeled_df = labeler.label_dataset(df)
    labeler.save_to_csv(labeled_df)
    labeler.analyze_labeling_results(labeled_df)

    # Print contoh
    print(labeled_df[['comment_text','label_ultimate','label_confidence','label_category','strong_brands_detected','patterns_detected']])
