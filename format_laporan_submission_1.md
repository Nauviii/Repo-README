# Laporan Proyek Machine Learning – Fikri Kurnia  
Fraud Detection Model Using Machine Learning

---

# Domain Proyek

Deteksi penipuan (fraud detection) merupakan tantangan besar dalam industri keuangan dan e-commerce. Dengan meningkatnya transaksi digital, pelaku penipuan semakin canggih dalam mengeksploitasi celah sistem.

Menurut *Nilson Report (2022)*, kerugian global akibat penipuan kartu pembayaran diprediksi mencapai **$38.5 miliar** pada tahun 2027.  
Penelitian akademik seperti *West & Bhattacharya (2016)* dan *Rao (2019)* juga menyatakan bahwa rule-based systems tidak lagi cukup karena fraud semakin dinamis dan non-linear.

Dengan adanya pertumbuhan volume transaksi dan variasi pola penipuan, machine learning menjadi solusi efektif karena mampu:
- Mempelajari pola kompleks,
- Menangani data besar,
- Beradaptasi dengan pola fraud baru,
- Mengambil keputusan cepat pada transaksi real-time.

Proyek ini bertujuan membangun **model deteksi fraud** berbasis machine learning pada dataset berisi **±300.000 transaksi**, dengan tingkat fraud hanya **2.2%**, mencerminkan kondisi dunia nyata yang sangat imbalanced.

### Referensi Kredibel
- Nilson Report, *Global Fraud Forecast*, 2022  
- J. West & M. Bhattacharya, *Intelligent Financial Fraud Detection*, 2016  
- A. Rao & F. Machine, *Survey on Credit Card Fraud Detection*, IEEE Access, 2019

---

# Business Understanding

Fraud detection tidak hanya terkait analisis teknis, tetapi juga keputusan bisnis yang berhubungan langsung dengan risiko finansial, biaya operasional investigasi, dan kenyamanan pelanggan. Oleh karena itu, solusi yang dibangun harus selaras dengan kebutuhan perusahaan dan risiko yang ingin dikurangi.

## Problem Statements

1. **Bagaimana mendeteksi fraud secara akurat pada dataset yang sangat imbalanced (fraud hanya 2.2%)?**  
   Dengan imbalance ekstrem, model cenderung bias terhadap kelas mayoritas.

2. **Model atau algoritma mana yang paling sesuai dengan kebutuhan bisnis (precision tinggi, recall tinggi, atau balance)?**

3. **Bagaimana menentukan threshold prediksi yang optimal agar keputusan blocking, review, atau approval sesuai strategi risiko perusahaan?**

4. **Fitur apa yang paling mempengaruhi terjadinya fraud dan bagaimana insight tersebut dapat digunakan oleh tim operasional dan risk management?**

---

## Goals

1. Mengembangkan model machine learning yang mampu memisahkan transaksi fraud dan non-fraud secara akurat.
2. Menguji beberapa algoritma untuk menemukan model yang paling stabil dan optimal.
3. Melakukan threshold tuning untuk optimasi keputusan bisnis.
4. Mengidentifikasi fitur penting penyebab fraud.
5. Membangun pipeline end-to-end yang bebas data leakage.

---

## Solution Statements

Untuk mencapai goals tersebut, proyek ini menggunakan beberapa pendekatan:

### 1. Multi-Model Evaluation  
Menggunakan empat model ML:
- Logistic Regression (baseline)
- XGBoost (precision terbaik)
- LightGBM (F1-score terbaik)
- CatBoost (recall terbaik)

Perbandingan ini memberikan gambaran lengkap mengenai trade-off performa tiap algoritma.

### 2. Encoding & GPU Optimization  
Setiap model menggunakan teknik encoding berbeda:
- Logistic → Target Encoding + Scaling  
- XGBoost & LGBM → Ordinal Encoding  
- CatBoost → Native Categorical GPU  

Seluruh model tree-boosting menggunakan **GPU acceleration**.

### 3. Stratified K-Fold  
Digunakan untuk memastikan stabilitas hasil evaluasi pada data imbalanced.

### 4. Threshold Tuning (F1, F2, Cost-Based)  
Karena default threshold (0.5) tidak optimal, dilakukan optimasi untuk tiga skenario bisnis:
- Automatic Blocking  
- Analyst Review Queue  
- Hybrid Decision Engine  

### 5. Evaluasi Mendalam  
Menggunakan:
- AUPRC,
- Precision@K,
- ROC & PR Curve,
- Cost-based evaluation.

### 6. Feature Interpretation  
Mengidentifikasi fitur paling berpengaruh menggunakan:
- violin plot,
- distribusi,
- Wasserstein distance.

---

# Data Understanding

Dataset terdiri dari ±300.000 baris tanpa missing value atau duplikasi.

### Variabel pada dataset:

| Fitur | Deskripsi |
|------|-----------|
| account_age_days | Lama usia akun |
| total_transactions_user | Total histori transaksi user |
| amount | Nominal transaksi |
| country | Negara user |
| bin_country | Negara asal kartu |
| channel | Web/App |
| merchant_category | Kategori merchant |
| avs_match | Status AVS |
| cvv_result | Status CVV |
| three_ds_flag | Penggunaan 3DS |
| shipping_distance_km | Jarak pengiriman |
| transaction_time | Timestamp transaksi |
| is_fraud | Label (0/1) |

### Temuan EDA Penting

- Fraud hanya **2.2%** → highly imbalanced.  
- Fraud cenderung terjadi pada:
  - Akun baru,
  - Transaksi bernilai besar,
  - AVS/CVV mismatch,
  - Tanpa 3DS,
  - Jarak pengiriman jauh,
  - User dengan sedikit transaksi historis.

### Strong Predictors  
(Berdasarkan violin plot + Wasserstein distance)

1. shipping_distance_km  
2. amount  
3. account_age_days  
4. avs_match  
5. cvv_result  
6. three_ds_flag  

---

# Data Preparation

Proses dilakukan secara berurutan:

1. Menghapus fitur non-prediktif: `transaction_id`, `user_id`
2. Konversi tipe data datetime
3. Membuat fitur baru:
   - Membuat fitur hour, day, ,weekday, dan month dari ekstraksi fitur transaction_time
   - Membuat fitur evening dan weekend berdasarkan binning pada fitur hour dan weekday
5. Encoding berbeda sesuai model:
   - Target Encoding (Logistic)
   - Ordinal Encoding (XGB/LGBM)
   - Native categorical (CatBoost)
6. Scaling untuk model linear
7. Feature Selection berdasakran korelasi heatmap dan proses EDA
8. Penanganan imbalance dengan:
   - class_weight  
   - scale_pos_weight  
   - Precision@K metrics
9. Stratified train-test split

### Alasan Data Preparation
- Mengekstrak fitur datetime sebagai predictor yang kuat
- Menghindari data leakage  
- Meningkatkan stabilitas training  
- Memperbaiki representasi kategori  
- Mendukung performa algoritma tree vs linear  

---

# Modeling

Empat algoritma diuji:

### 1. Logistic Regression
- **Kelebihan:** interpretatif, cepat  
- **Kekurangan:** tidak menangkap pola non-linear

### 2. XGBoost (GPU)
- **Precision tertinggi**  
- Cocok untuk strategi *auto-blocking*  

### 3. LightGBM (GPU)
- **F1-score tertinggi**  
- Model paling seimbang antara precision & recall → **model terbaik**

### 4. CatBoost (GPU)
- **Recall tertinggi**  
- Cocok untuk *fraud catching* maksimal

---

# Evaluation

### Metrik yang Digunakan

- Precision  
- Recall  
- F1-score  
- AUPRC  
- ROC Curve  
- Precision@K  
- Confusion Matrix  
- Cost-based Evaluation  

### Precision
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

### Recall
$$
\text{Recall} = \frac{TP}{TP + FN}
$$

### F1-score
$$
F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

### Hasil Tahap 1

- **Precision terbaik → XGBoost**  
- **F1-score terbaik → LightGBM**  
- **Recall terbaik → CatBoost**  

### Hasil Tahap 2 – Threshold Tuning

| Threshold | Kegunaan |
|----------|----------|
| F1 (0.9533) | Auto-blocking |
| Cost (0.4560) | Analyst review |
| Hybrid | block / review / approve |

Hybrid merupakan strategi paling realistis di industri:
- prob > 0.95 → block  
- 0.45 – 0.95 → review  
- prob < 0.45 → approve  

---

# Kesimpulan Akhir

Proyek ini berhasil membangun sistem deteksi fraud yang:

- bekerja pada dataset besar & sangat imbalanced,  
- memiliki pipeline rapi & bebas leakage,  
- memanfaatkan GPU training,  
- menghasilkan model LightGBM dengan F1-score terbaik,  
- dapat disesuaikan melalui threshold tuning untuk berbagai skenario bisnis:  
  - blocking otomatis,  
  - alerting analis,  
  - hybrid decision engine.

Model siap untuk dikembangkan lebih lanjut menuju:
- real-time scoring,
- monitoring performa,
- deployment produksi.

---

# Future Work

- Integrasi real-time anomaly detection  
- SHAP-based interpretability  
- Fraud dashboard analytics  
- Deployment FastAPI/MLflow  
- Auto-retraining & drift monitoring  

---

