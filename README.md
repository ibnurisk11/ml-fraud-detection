# ml-fraud-detection

Eksperimen machine learning untuk deteksi fraud pada transaksi kartu kredit. Notebook ini melakukan EDA (Exploratory Data Analysis) sederhana lalu melatih model **Logistic Regression** dengan penanganan data imbalance.

## Dataset

Dataset yang digunakan adalah [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle), tersimpan di `dataset/creditcard.csv`.

| Properti | Nilai |
|---|---|
| Jumlah baris | 284.807 |
| Jumlah kolom | 31 |
| Fraud (`Class = 1`) | 492 (0,1727%) |
| Normal (`Class = 0`) | 284.315 |
| Missing values | Tidak ada |
| Kolom fitur | `Time`, `V1`–`V28` (hasil PCA), `Amount` |
| Kolom target | `Class` |

> **Catatan:** File CSV tidak di-commit ke repository (lihat `.gitignore`), jadi unduh dataset dari Kaggle terlebih dahulu.

## Struktur Proyek

```
ml-fraud-detection/
├── README.md
├── logistic-regression.ipynb   # Notebook utama (EDA + model)
├── dataset/
│   └── creditcard.csv          # Dataset (diunduh dari Kaggle)
└── .gitignore
```

## Persyaratan

- Python 3.10+
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `jupyter` / `nbconvert` (untuk menjalankan notebook)

Instalasi dependensi:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

## Cara Menjalankan

1. Clone repository:

   ```bash
   git clone <url-repo>
   cd ml-fraud-detection
   ```

2. Unduh `creditcard.csv` dari Kaggle, lalu letakkan di folder `dataset/`:

   ```
   dataset/creditcard.csv
   ```

3. Buka dan jalankan notebook:

   ```bash
   jupyter notebook logistic-regression.ipynb
   ```

   atau eksekusi seluruh cell dari terminal:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace logistic-regression.ipynb
   ```

## Isi Notebook

1. **Import Library** — `pandas`, `numpy`, `matplotlib`, `scikit-learn`
2. **EDA**
   - Membaca dataset dan mengacak urutan baris (`random_state=42`)
   - Melihat jumlah baris/kolom dan tipe data
   - Cek missing values
   - Visualisasi sebaran `Amount` dan `V22` terhadap label fraud/normal
3. **Logistic Regression**
   - `train_test_split` dengan `test_size=0.35`, `stratify=y`, `random_state=42`
   - Pipeline: `StandardScaler` → `LogisticRegression`
     - `class_weight='balanced'` untuk menangani imbalance
     - `C=1.0`, solver `lbfgs`, `max_iter=1000`
   - Evaluasi: accuracy, classification report, confusion matrix, FNR, ROC-AUC, PR-AUC
   - Plot kurva ROC dan Precision-Recall

## Hasil

Hasil pada data uji (99.683 baris, 172 fraud):

| Metric | Nilai |
|---|---|
| Accuracy | 0,9771 |
| Recall (Fraud) | 0,90 |
| Precision (Fraud) | 0,06 |
| F1-Score (Fraud) | 0,12 |
| False Negative Rate (FNR) | 0,0988 |
| ROC-AUC | 0,9771 |
| PR-AUC (Average Precision) | 0,7265 |

**Interpretasi:**

- Model berhasil menangkap ±90% kasus fraud (recall tinggi, FNR ±10%).
- Precision rendah (0,06) berarti banyak transaksi normal yang terflag sebagai fraud — trade-off yang wajar pada data imbalance ketika `class_weight='balanced'` dipakai agar fraud tidak terlewat.
- Pada dataset imbalance seperti ini, **accuracy saja menyesatkan**; gunakan recall fraud, FNR, ROC-AUC, dan PR-AUC untuk menilai model.

## Langkah Lanjutan

- Bandingkan dengan model lain: Random Forest, XGBoost, SVM, dll.
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Coba teknik resampling (SMOTE, undersampling) dan bandingkan dengan `class_weight`
- Eksplorasi threshold optimal berdasarkan biaya false positive vs false negative
