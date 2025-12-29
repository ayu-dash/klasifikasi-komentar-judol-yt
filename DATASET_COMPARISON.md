# Perbandingan Dataset untuk Proyek Klasifikasi Spam Judol

## 📊 Ringkasan Dataset yang Tersedia

Berdasarkan analisis folder `datasets/`, terdapat **3 file CSV utama**:

| No | Nama File | Jumlah Baris | Deskripsi |
|----|-----------|--------------|-----------|
| 1  | `comments_from_scraping_new.csv` | **133,786** | Dataset mentah hasil scraping YouTube (belum dilabeli) |
| 2  | `comments_labeled_final.csv` | **91,179** | Dataset yang sudah dilabeli (metode final labeling) |
| 3  | `comments_labeled_ensemble.csv` | **91,178** | Dataset dengan label hasil ensemble (AI + Expert Pattern) |

---

## 📝 Detail Setiap Dataset

### 1️⃣ Dataset: `comments_from_scraping_new.csv`
**Tipe**: Dataset Raw (Hasil Scraping)

**Karakteristik**:
- ✅ **Jumlah Data**: 133,786 baris (paling banyak)
- ⚠️ **Status Label**: **BELUM DILABELI** 
- 📦 **Use Case**: Data mentah untuk labeling manual atau semi-otomatis

**Kolom yang Tersedia** (estimasi berdasarkan scraping YouTube):
```
- video_id: ID video YouTube
- author: Username pembuat komentar  
- comment_text: Teks komentar (FITUR UTAMA)
- published_at: Timestamp publikasi
- like_count: Jumlah like
```

**Kelebihan**:
- Dataset paling besar (133K+ komentar)
- Data fresh dari scraping terbaru

**Kekurangan**:
- ❌ Tidak ada kolom label → Tidak bisa langsung digunakan untuk supervised learning
- Perlu proses labeling terlebih dahulu

---

### 2️⃣ Dataset: `comments_labeled_final.csv`
**Tipe**: Dataset Labeled (Final Version)

**Karakteristik**:
- ✅ **Jumlah Data**: 91,179 baris
- ✅ **Status Label**: **SUDAH DILABELI**
- 📦 **Use Case**: Training dan testing machine learning model

**Kolom yang Tersedia**:
```
- video_id, author, comment_text, published_at, like_count (dari scraping)
- is_promo: Label promo (boolean)
- ai_prob: Probabilitas prediksi AI
- label: LABEL TARGET (0 = Safe, 1 = Judol) ← FITUR TARGET UTAMA
```

**Kelebihan**:
- ✅ Sudah memiliki label ground truth
- ✅ Bisa langsung digunakan untuk training model
- ✅ Memiliki AI probability score untuk analisis

**Kekurangan**:
- Jumlah data lebih sedikit dari raw dataset (karena proses cleaning)
- Hanya menggunakan 1 metode labeling (final version)

---

### 3️⃣ Dataset: `comments_labeled_ensemble.csv` ⭐ **REKOMENDED**
**Tipe**: Dataset Ensemble (AI + Expert Pattern)

**Karakteristik**:
- ✅ **Jumlah Data**: 91,178 baris
- ✅ **Status Label**: **SUDAH DILABELI (ENSEMBLE METHOD)**
- 📦 **Use Case**: Training model dengan label paling akurat

**Kolom yang Tersedia**:
```
- video_id, author, comment_text, published_at, like_count (dari scraping)
- is_promo: Label promo (boolean)
- ai_prob: Probabilitas prediksi AI
- label: Label dari metode tunggal
- ensemble_pred: LABEL ENSEMBLE (0 = Safe, 1 = Judol) ← LABEL TERBAIK
- ensemble_prob: Probabilitas ensemble
```

**Kelebihan**:
- ✅ **Label paling akurat** (kombinasi AI + Expert Pattern)
- ✅ **Kolom `ensemble_pred`** → hasil voting dari multiple methods
- ✅ Mengurangi bias dari single labeling method
- ✅ Cocok untuk menangani imbalanced data
- ✅ Memiliki probability score untuk confidence analysis

**Kekurangan**:
- Jumlah data sedikit lebih kecil dari labeled_final (1 baris difference, kemungkinan karena data cleaning)

---

## 🎯 Rekomendasi untuk Notebook Machine Learning

### ✅ **GUNAKAN: `comments_labeled_ensemble.csv`**

**Alasan**:

1. **Label Lebih Reliable**: 
   - Ensemble method menggabungkan prediksi AI + Expert Pattern
   - Mengurangi false positive/negative dari single method

2. **Handling Imbalanced Data**:
   - Dataset judol vs safe sangat imbalanced (~5% vs 95%)
   - Ensemble labeling lebih robust terhadap ketimpangan ini

3. **Kolom Lengkap untuk Analisis**:
   ```python
   # Kolom yang bisa digunakan:
   X = df['comment_text']          # Fitur input
   y = df['ensemble_pred']         # Target label (TERBAIK)
   
   # Analisis tambahan:
   df['ai_prob']                   # Untuk membandingkan dengan model kita
   df['ensemble_prob']             # Untuk threshold tuning
   ```

4. **Kompabilitas dengan Rubrik**:
   - ✅ Memiliki label untuk supervised learning
   - ✅ Cukup data untuk train/test split (80:20)
   - ✅ Bisa analisis imbalanced data handling
   - ✅ Cocok untuk compare Traditional ML vs Deep Learning

---

## 📌 Distribusi Data (Estimasi)

Berdasarkan informasi dari notebook yang ada:

**Dataset Ensemble** (`ensemble_pred`):
- **Kelas 0 (Safe)**: ~74,797 komentar (94.33%)
- **Kelas 1 (Judol)**: ~4,498 komentar (5.67%)
- **Imbalance Ratio**: 1:17

Total setelah cleaning: **79,295 data valid** (dari 91,178 raw)

---

## 🔄 Workflow Penggunaan Dataset

```
1. Raw Scraping
   ↓
   comments_from_scraping_new.csv (133,786 rows)
   
2. Labeling Process (AI + Expert Pattern + Manual)
   ↓
   comments_labeled_final.csv (91,179 rows)
   ↓
   comments_labeled_ensemble.csv (91,178 rows)
   
3. Data Cleaning & Preprocessing
   ↓
   Final Dataset untuk Modeling (79,295 rows)
```

---

## 💡 Tips Implementasi di Notebook

```python
# Load dataset yang direkomendasikan
import pandas as pd

df = pd.read_csv('../datasets/comments_labeled_ensemble.csv')

# Gunakan kolom ini:
X = df['comment_text']        # Text features
y = df['ensemble_pred']       # Target labels (0/1)

# Handling missing values
df = df.dropna(subset=['comment_text', 'ensemble_pred'])

# Train-test split dengan stratified (penting untuk imbalanced data)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # PENTING untuk imbalanced data
)
```

---

## ✅ Kesimpulan

| Aspek | Dataset Terpilih |
|-------|------------------|
| **File** | `comments_labeled_ensemble.csv` |
| **Jumlah Data** | 79,295 (setelah cleaning) |
| **Label Column** | `ensemble_pred` |
| **Distribusi** | 94.33% Safe, 5.67% Judol |
| **Quality** | ⭐⭐⭐⭐⭐ (Ensemble = Most Reliable) |
| **Readiness** | Ready for ML Training |

---

**Generated on**: 2025-12-26  
**Project**: Klasifikasi Komentar Spam Judi Online YouTube
