import pandas as pd
import re
import numpy as np

class FinalProductionJudolDetector:
    def __init__(self):
        self.judi_sites = [
            'lazadatoto', 'pstoto99', 'mini1221', 'arwanatoto', 'garudahoki', 'garuda hoki',
            'pulauwin', 'berkah99', 'seru69', 'pesiar88', 'sgi88', 'plazabola', 'bukit4d',
            'sajak4d', 'pelatih4d', 'gelora4d', 'sendal4d', 'probet855', 'crown138', 'visi4d',
            'timo4d', 'mona4d', 'qqplay4d', 'mbak4d2', 'sikat88', 'playtoto98', 'ganas33',
            'target68', 'wokebet', 'arwanat', 'pstoto', 'togel62', 'pulau777', 'sgi', 'probetslot'
        ]
        
        self.financial_terms = [
            'wd', 'depo', 'modal', 'jp', 'menang', 'bonus', 'saldo', 'gacor', 'cuan',
            'jepe', 'maxwin', 'hoki', 'bet', 'spin', 'judi', 'judol', 'slot', 'togel',
            'kalah', 'untung', 'rugi', 'hasil', 'duit', 'uang', 'receh', 'gope', 'rtp'
        ]
        
        self.high_confidence_phrases = [
            r'minimal depo', r'modal receh', r'barusan wd', r'langsung menang',
            r'wd lancar', r'auto wd', r'wd cepat', r'cari google', r'jamin menang',
            r'pasti menang', r'gampang menang', r'depo kecil', r'hasil besar'
        ]

    def extract_final_features(self, df):
        """Final optimized feature extraction"""
        df = df.copy()
        
        # Prepare text
        df['cleaned_comment_text'] = df['cleaned_comment_text'].fillna('').astype(str)
        df['comment_text'] = df['comment_text'].fillna('').astype(str)
        df['combined_text'] = (df['cleaned_comment_text'] + ' ' + df['comment_text']).str.lower()
        
        texts = df['combined_text']
        
        # Core features - fixed currency pattern warning
        df['has_judi_site'] = texts.apply(lambda x: any(site in x for site in self.judi_sites)).astype(int)
        df['financial_term_count'] = texts.apply(lambda x: sum(1 for term in self.financial_terms if term in x))
        df['has_high_confidence_phrase'] = texts.apply(
            lambda x: any(re.search(phrase, x) for phrase in self.high_confidence_phrases)
        ).astype(int)
        df['has_currency'] = texts.str.contains(r'\d+\s*(?:jt|rb|k|juta|ribu)', na=False).astype(int)  # Fixed pattern
        df['has_large_number'] = texts.str.contains(r'\b[1-9]\d{2,}\b', na=False).astype(int)
        
        # Strategic combinations for optimal recall/precision balance
        df['site_plus_any_financial'] = ((df['has_judi_site'] == 1) & (df['financial_term_count'] >= 1)).astype(int)
        df['site_plus_multiple_financial'] = ((df['has_judi_site'] == 1) & (df['financial_term_count'] >= 2)).astype(int)
        df['site_plus_phrase'] = ((df['has_judi_site'] == 1) & (df['has_high_confidence_phrase'] == 1)).astype(int)
        df['financial_plus_currency'] = ((df['financial_term_count'] >= 2) & (df['has_currency'] == 1)).astype(int)
        
        # Confidence levels with optimized thresholds
        df['very_high_confidence'] = (
            (df['site_plus_phrase'] == 1) |
            ((df['site_plus_multiple_financial'] == 1) & (df['has_currency'] == 1))
        ).astype(int)
        
        df['high_confidence'] = (
            (df['financial_plus_currency'] == 1) |
            (df['site_plus_multiple_financial'] == 1) |
            ((df['has_judi_site'] == 1) & (df['has_high_confidence_phrase'] == 0) & (df['financial_term_count'] >= 3))
        ).astype(int)
        
        df['medium_confidence'] = (
            (df['site_plus_any_financial'] == 1) |
            ((df['financial_term_count'] >= 3) & (df['has_currency'] == 1)) |
            ((df['has_judi_site'] == 1) & (df['financial_term_count'] >= 2))
        ).astype(int)
        
        df['low_confidence'] = (
            (df['has_judi_site'] == 1) |
            (df['financial_term_count'] >= 2) |
            (df['has_high_confidence_phrase'] == 1)
        ).astype(int)
        
        return df

    def calculate_final_score(self, df):
        """Final optimized scoring"""
        df = df.copy()
        
        # Optimized weights based on performance analysis
        weights = {
            # Very high confidence (max precision)
            'very_high_confidence': 12.0,
            'site_plus_phrase': 10.0,
            
            # High confidence (good precision)
            'high_confidence': 8.0,
            'financial_plus_currency': 7.5,
            'site_plus_multiple_financial': 7.0,
            
            # Medium confidence (balanced)
            'medium_confidence': 5.0,
            'site_plus_any_financial': 4.5,
            
            # Base features (recall focused)
            'has_high_confidence_phrase': 4.0,
            'has_judi_site': 3.0,
            'financial_term_count': 1.2,  # Reduced to prevent over-scoring
            'has_currency': 2.0,
            'has_large_number': 1.0,
        }
        
        # Calculate score
        df['raw_score'] = 0
        for feature, weight in weights.items():
            if feature in df.columns:
                df['raw_score'] += df[feature] * weight
        
        # Normalize to 0-10 scale
        max_score = df['raw_score'].max()
        if max_score > 0:
            df['judol_score'] = (df['raw_score'] / max_score) * 10
        else:
            df['judol_score'] = 0
        
        # Final optimized classification
        df['risk_level'] = 'Very Low'
        df.loc[df['judol_score'] >= 0.5, 'risk_level'] = 'Low'
        df.loc[df['judol_score'] >= 2.0, 'risk_level'] = 'Medium'
        df.loc[df['judol_score'] >= 4.5, 'risk_level'] = 'High'
        df.loc[df['judol_score'] >= 7.5, 'risk_level'] = 'Very High'
        
        # Business actions
        df['action'] = 'Monitor'
        df.loc[df['risk_level'] == 'Medium', 'action'] = 'Review'
        df.loc[df['risk_level'] == 'High', 'action'] = 'Flag'
        df.loc[df['risk_level'] == 'Very High', 'action'] = 'Remove'
        
        # Confidence score
        df['confidence'] = df['judol_score'] / 10
        
        return df

    def final_performance_report(self, df):
        """Final performance report with business recommendations"""
        if 'target' not in df.columns:
            return
        
        judol_data = df[df['target'] == 1]
        non_judol_data = df[df['target'] == 0]
        
        print("=" * 80)
        print("FINAL PRODUCTION JUDOL DETECTOR - PERFORMANCE REPORT")
        print("=" * 80)
        
        # Performance by action level
        actions = ['Monitor', 'Review', 'Flag', 'Remove']
        print(f"\n{'BUSINESS ACTION PERFORMANCE':^80}")
        print("-" * 80)
        print(f"{'Action':<8} {'Judol':>6} {'Non-Judol':>9} {'Recall':>8} {'Precision':>10} {'Efficiency':>10}")
        print("-" * 80)
        
        for action in actions:
            judol_in_action = len(judol_data[judol_data['action'] == action])
            all_in_action = len(df[df['action'] == action])
            
            recall = judol_in_action / len(judol_data) * 100 if len(judol_data) > 0 else 0
            precision = judol_in_action / all_in_action * 100 if all_in_action > 0 else 0
            efficiency = recall * precision / 100  # Combined metric
            
            non_judol_in_action = len(non_judol_data[non_judol_data['action'] == action])
            
            print(f"{action:<8} {judol_in_action:>6} {non_judol_in_action:>9} {recall:>7.1f}% {precision:>9.1f}% {efficiency:>9.1f}%")
        
        # Summary for business decisions
        print(f"\n{'BUSINESS DECISION SUMMARY':^80}")
        print("-" * 80)
        review_plus = judol_data[judol_data['action'].isin(['Review', 'Flag', 'Remove'])]
        flag_plus = judol_data[judol_data['action'].isin(['Flag', 'Remove'])]
        
        print(f"Comments needing Review+: {len(review_plus):,}/{len(judol_data):,} ({len(review_plus)/len(judol_data)*100:.1f}% of judol)")
        print(f"Comments needing Flag+:   {len(flag_plus):,}/{len(judol_data):,} ({len(flag_plus)/len(judol_data)*100:.1f}% of judol)")
        print(f"False Positive Rate:      {len(non_judol_data[non_judol_data['action'] != 'Monitor'])/len(non_judol_data)*100:.1f}%")
        
        # Cost-benefit analysis
        total_actionable = len(df[df['action'].isin(['Flag', 'Remove'])])
        judol_actionable = len(judol_data[judol_data['action'].isin(['Flag', 'Remove'])])
        precision_actionable = judol_actionable / total_actionable * 100 if total_actionable > 0 else 0
        
        print(f"\n{'COST-BENEFIT ANALYSIS':^80}")
        print("-" * 80)
        print(f"High-confidence actions (Flag/Remove): {total_actionable:,} comments")
        print(f"Judol caught in high-confidence: {judol_actionable:,} comments")
        print(f"Precision in high-confidence: {precision_actionable:.1f}%")
        print(f"Manual review needed: {len(df[df['action'] == 'Review']):,} comments")

def main():
    # Load data
    df = pd.read_csv('labeled_comments.csv')
    print(f"FINAL PRODUCTION DETECTOR")
    print(f"Dataset: {len(df):,} comments, {df['target'].sum():,} judol comments")
    
    # Initialize final detector
    detector = FinalProductionJudolDetector()
    
    # Extract features
    df_features = detector.extract_final_features(df)
    
    # Calculate scores
    df_scored = detector.calculate_final_score(df_features)
    
    # Generate final report
    detector.final_performance_report(df_scored)
    
    # Save final production results
    output_file = 'final_production_judol_detection.csv'
    df_scored.to_csv(output_file, index=False)
    print(f"\n✅ FINAL PRODUCTION RESULTS saved to: {output_file}")
    
    # Deployment recommendations
    print(f"\n{'DEPLOYMENT RECOMMENDATIONS':^80}")
    print("=" * 80)
    print("1. AUTOMATIC ACTIONS: Apply 'Remove' action for Very High risk comments")
    print("2. PRIORITY REVIEW:   Manually review 'Flag' action comments") 
    print("3. SAMPLE MONITORING: Periodically check 'Review' action comments")
    print("4. FALSE POSITIVE:    Monitor 'Monitor' action for potential judol misses")
    print("=" * 80)

if __name__ == "__main__":
    main()
