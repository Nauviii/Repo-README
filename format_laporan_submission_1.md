# Laporan Proyek Machine Learning – Fikri Kurnia  
Fraud Detection Model Using Machine Learning

---

## Domain Proyek

Deteksi penipuan (fraud detection) merupakan tantangan besar dalam industri keuangan dan e-commerce. Dengan meningkatnya transaksi digital, pelaku penipuan semakin canggih dalam mengeksploitasi celah sistem.

Menurut *Nilson Report (2022)*, kerugian global akibat penipuan kartu pembayaran diprediksi mencapai **$38.5 miliar** pada tahun 2027.  
Penelitian akademik seperti *West & Bhattacharya (2016)* dan *Rao (2019)* juga menyatakan bahwa rule-based systems tidak lagi cukup karena fraud semakin dinamis dan non-linear.

Dengan adanya pertumbuhan volume transaksi dan variasi pola penipuan, machine learning menjadi solusi efektif karena mampu:
- Mempelajari pola kompleks,
- Menangani data besar,
- Beradaptasi dengan pola fraud baru,
- Mengambil keputusan cepat pada transaksi real-time.

Proyek ini bertujuan membangun **model deteksi fraud** berbasis machine learning pada dataset berisi **299.695 transaksi**, dengan tingkat fraud hanya **2.2%**, mencerminkan kondisi dunia nyata yang sangat imbalanced.

### Referensi Kredibel
- Nilson Report, *Global Fraud Forecast*, 2022  
- J. West & M. Bhattacharya, *Intelligent Financial Fraud Detection*, 2016  
- A. Rao & F. Machine, *Survey on Credit Card Fraud Detection*, IEEE Access, 2019

---

## Business Understanding

Fraud detection tidak hanya terkait analisis teknis, tetapi juga keputusan bisnis yang berhubungan langsung dengan risiko finansial, biaya operasional investigasi, dan kenyamanan pelanggan. Oleh karena itu, solusi yang dibangun harus selaras dengan kebutuhan perusahaan dan risiko yang ingin dikurangi.

### Problem Statements

1. **Bagaimana mendeteksi fraud secara akurat pada dataset yang sangat imbalanced (fraud hanya 2.2%)?**  
   Dengan imbalance ekstrem, model cenderung bias terhadap kelas mayoritas.

2. **Model atau algoritma mana yang paling sesuai dengan kebutuhan bisnis (precision tinggi, recall tinggi, atau balance)?**

3. **Bagaimana menentukan threshold prediksi yang optimal agar keputusan blocking, review, atau approval sesuai strategi risiko perusahaan?**

4. **Fitur apa yang paling mempengaruhi terjadinya fraud dan bagaimana insight tersebut dapat digunakan oleh tim operasional dan risk management?**

---

### Goals

1. Mengembangkan model machine learning yang mampu memisahkan transaksi fraud dan non-fraud secara akurat.
2. Menguji beberapa algoritma untuk menemukan model yang paling stabil dan optimal.
3. Melakukan threshold tuning untuk optimasi keputusan bisnis.
4. Mengidentifikasi fitur penting penyebab fraud.
5. Membangun pipeline end-to-end yang bebas data leakage.

---

### Solution Statements

Untuk mencapai goals tersebut, proyek ini menggunakan beberapa pendekatan:

#### 1. Multi-Model Evaluation  
Menggunakan empat model ML:
- Logistic Regression (baseline)
- XGBoost (precision terbaik)
- LightGBM (F1-score terbaik)
- CatBoost (recall terbaik)

Perbandingan ini memberikan gambaran lengkap mengenai trade-off performa tiap algoritma.

#### 2. Encoding & GPU Optimization  
Setiap model menggunakan teknik encoding berbeda:
- Logistic → Target Encoding + Scaling  
- XGBoost & LGBM → Ordinal Encoding  
- CatBoost → Native Categorical GPU  

Seluruh model tree-boosting menggunakan **GPU acceleration**.

#### 3. Stratified K-Fold  
Digunakan untuk memastikan stabilitas hasil evaluasi pada data imbalanced.

#### 4. Threshold Tuning (F1, F2, Cost-Based)  
Karena default threshold (0.5) tidak optimal, dilakukan optimasi untuk tiga skenario bisnis:
- Automatic Blocking  
- Analyst Review Queue  
- Hybrid Decision Engine  

#### 5. Evaluasi Mendalam  
Menggunakan:
- AUPRC,
- Precision@K,
- ROC & PR Curve,
- Cost-based evaluation.

#### 6. Feature Interpretation  
Mengidentifikasi fitur paling berpengaruh menggunakan:
- violin plot,
- distribusi,
- Wasserstein distance.

---

## Data Understanding
🔗 **Kaggle – E-Commerce Fraud Detection Dataset**  
https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset
Pada tahap Data Understanding, dilakukan profiling lengkap terhadap struktur dataset, kualitas data, distribusi fitur, hubungan antar variabel, serta pola-pola yang mengindikasikan potensi fraud. Dataset berisi **299.695 transaksi**, terdiri dari fitur numerik, kategorikal, serta informasi waktu.

---
### Variabel pada dataset:

| Fitur | Deskripsi | Tipe Data |
|------|-----------|----------|
| transaction_id | Pengidentifikasi transaksi unik | int |
| user_id | Pengidentifikasi pengguna (setiap pengguna 40–60 transaksi) | int |
| account_age_days | Lama usia akun | int |
| total_transactions_user | Total histori transaksi user | int |
| avg_amount_user | Jumlah transaksi rata-rata pengguna | float |
| amount | Nominal transaksi | float |
| country | Negara user | object |
| bin_country | Negara asal kartu | object |
| channel | Web/App | object |
| merchant_category | Kategori merchant | object |
| promo_used | Penggunaan Promo | int |
| avs_match | Status AVS | int |
| cvv_result | Status CVV | int |
| three_ds_flag | Penggunaan 3DS | int |
| shipping_distance_km | Jarak pengiriman | float |
| transaction_time | Timestamp transaksi | object |
| is_fraud | Label (0/1) | int |


### 1. Data Quality Check

- **Tidak terdapat missing value** pada seluruh fitur.  
- **Tidak ditemukan duplikasi** → seluruh transaksi adalah kejadian unik.  
- Dataset memiliki **kombinasi numerik & kategorikal** yang seimbang.  
- Fitur waktu `transaction_time` memiliki format string → perlu konversi ke datetime.

Ini merupakan kondisi ideal untuk fraud modeling karena:
- tidak membutuhkan imputasi (menghindari bias),
- dataset murni mencerminkan pola transaksi sebenarnya.

---

### 2. Overview Statistik (Numerik)

Berdasarkan `describe()`:

#### Distribusi fitur numerik penting:
- **account_age_days**  
  - Mean: 973 hari (~2.6 tahun)  
  - Fraud memiliki *jauh lebih banyak akun baru* (distribusi condong ke kiri)

- **total_transactions_user**  
  - Rata-rata transaksi historis ~50  
  - Fraud → user dengan riwayat transaksi *lebih sedikit*

- **avg_amount_user & amount**  
  - Distribusi sangat right-skewed  
  - Fraud → cenderung terjadi pada transaksi bernilai lebih tinggi  
  - Terdapat outlier extreme hingga **16.994** (amount)

- **shipping_distance_km**  
  - Distribusi sangat right-skewed  
  - Fraud → lebih sering pada jarak jauh

Insight ini menunjukkan pola fraud umum:  
> "Akun baru, nilai transaksi tinggi, dan transaksi jarak jauh merupakan kombinasi risiko utama."

---

### 3. Overview Statistik (Kategorikal)

- **country & bin_country**  
  - Masing-masing memiliki 10 kategori  
  - Fraud rate tertinggi terdapat pada **TR**, **RO**, **PL**  
  - Fraud rate terendah: **DE**, **NL**

- **channel**  
  - 50% lebih transaksi via **web**  
  - Fraud lebih dominan pada web (potensi keamanan lebih rendah)

- **merchant_category**  
  - 5 kategori  
  - Fraud tertinggi pada kategori **electronics** (sesuai pola industri)

---

### 4. Fraud Rate per Country (Bivariate)

Dari grafik:

- **TR**, **RO**, **PL** → fraud rate tertinggi (0.027–0.028)  
- **US**, **GB**, **FR** → fraud rate moderat (~0.022)  
- **NL**, **DE** → fraud rate rendah (~0.018)

Interpretasi bisnis:

> Negara dengan fraud rate tinggi dapat menjadi prioritas untuk rule-based gating, geo-fencing, atau additional KYC.

---

### 5. EDA — Behavioral Pattern Analysis

#### account_age_days vs fraud
Fraud sangat dominan pada akun usia **0–200 hari**.  
Ini konsisten dengan literature bahwa akun baru belum memiliki reputasi historis → lebih mudah disalahgunakan.

#### amount vs fraud
- Fraud lebih sering terjadi pada nilai transaksi besar.  
- Distribusi menunjukkan pergeseran ke ekor kanan pada kelas fraud.

Interpretasi bisnis:
> Fraudster cenderung memaksimalkan keuntungan dalam sedikit transaksi sebelum akun diblokir.

#### avg_amount_user vs fraud
Fraud sering muncul pada:
- user dengan **sedikit riwayat transaksi**,  
- tetapi **melakukan transaksi jumlah besar**.

Ini pola umum:  
> “Fresh account melakukan transaksi abnormal dibanding histori.”

#### AVS & CVV mismatch
Fitur **avs_match** dan **cvv_result** sangat berbeda distribusinya:

- Fraud memiliki proporsi mismatch lebih tinggi:
  - AVS mismatch → potensi alamat palsu  
  - CVV mismatch → kartu dicuri atau credential phishing

#### three_ds_flag
Fraud jauh lebih banyak pada transaksi **tanpa 3DS**.

Alasannya jelas:
- 3DS memberikan autentikasi ekstra (OTP),  
- fraudster cenderung menghindarinya.

#### shipping_distance_km
Fraud cenderung terjadi pada jarak jauh:
- Fraud sering melibatkan drop-point palsu  
- Barang dikirim sangat jauh dari lokasi kartu

---

### 6. Outlier Analysis

Outlier signifikan ditemukan pada:
- `amount`  
- `avg_amount_user`  
- `shipping_distance_km`

Namun berdasarkan domain,
- **outlier tersebut valid**, bukan data error.  
- Transaksi besar → wajar pada e-commerce.  
- Jarak pengiriman jauh → global shipping.

Keputusan:
> “Outlier tidak dihapus karena merupakan sinyal fraud yang valid.”

---

### 7. Korelasi (Correlation Heatmap)

Beberapa korelasi menarik:

#### **Korelasi tertinggi dengan fraud:**
- `shipping_distance_km` → **0.27**  
- `amount` → **0.20**  
- `account_age_days` → **-0.12**  
- `avs_match` → **-0.22**  
- `cvv_result` → **-0.22**

Interpretasi:

- Jarak jauh → semakin besar kemungkinan fraud  
- Amount tinggi → berisiko  
- AVS/CVV mismatch → indikator kuat  
- Akun baru → rentan fraud  

#### Korelasi antar fitur:
- `amount` ↔ `avg_amount_user` → **0.73**  
  (tidak redundant, tetapi saling melengkapi)
- `avs_match` ↔ `cvv_result` → **0.53**
  (error kartu berpasangan)

Heatmap ini memperkuat fitur-fitur utama yang digunakan model boosting.

---

### 8. Kesimpulan Data Understanding

1. Dataset sangat imbalanced (fraud 2.2%).  
2. Fraud memiliki ciri umum:
   - akun baru  
   - transaksi besar  
   - shipping jarak jauh  
   - AVS/CVV mismatch  
   - tidak menggunakan 3DS  
   - user dengan sedikit transaksi historis  
3. Negara tertentu menyumbang fraud rate lebih tinggi.  
4. Outlier pada amount dan distance adalah sinyal penting, bukan anomali.  
5. Beberapa fitur memiliki korelasi kuat dengan fraud → sangat relevan untuk model tree.

Data Understanding ini menjadi fondasi kuat untuk:
- feature engineering waktu,
- pemilihan model,
- threshold tuning,
- dan strategi bisnis.



---

## Data Preparation

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

## Modeling

Pada tahap ini, empat algoritma machine learning digunakan untuk membangun sistem deteksi fraud yang **robust**, **scalable**, dan mampu beradaptasi dengan karakter dataset yang **sangat imbalanced (fraud ±2.2%)**.  
Setiap model dipilih berdasarkan pertimbangan statistik, performa generalisasi, serta relevansi dengan kebutuhan bisnis pada industri pembayaran digital.

---

### 1️. Logistic Regression (Baseline)

Logistic Regression digunakan sebagai **baseline model** untuk membandingkan performa algoritma non-linear.

#### **Alasan Pemilihan**
- Interpretabilitas tinggi — koefisien menjelaskan arah & kekuatan pengaruh fitur.
- Performa komputasi cepat — ideal untuk eksperimen awal.
- Linear baseline — menjadi titik acuan sebelum model kompleks.
- Stabil pada dataset besar.
- Cocok dengan:
  - **Target Encoding (TE)**
  - **StandardScaler**

#### **Intuisi Matematis**
Model menghitung probabilitas fraud menggunakan fungsi logistik:

\[
P(y = 1 \mid x) = \sigma(w^T x + b)
\]

Karena hanya mempelajari pola **linear**, model ini digunakan sebagai pembanding untuk model gradient boosting yang lebih kompleks.

#### **Hyperparameter (Alasan & Intuisi)**
| Hyperparameter        | Nilai | Alasan |
|----------------------|--------|---------|
| `max_iter`   | 2000 | memastikan model benar-benar konvergen dan tidak berhenti terlalu cepat |
| `class_weight` | balanced | mengurangi bias model terhadap majority class (non-fraud), meningkatkan recall fraud |

#### **✅ Kelebihan**
- Mudah diinterpretasikan.
- Cepat dilatih dan ringan.
- Risiko overfitting rendah.
- Stabil pada dataset besar.

#### **⚠️ Kekurangan**
- Tidak menangkap pola non-linear.
- Sensitif terhadap outlier.
- Performa menurun pada dataset imbalanced.
- Bergantung pada feature engineering.

---

### 2️. XGBoost — Precision Terbaik

#### **Alasan Pemilihan**
- Menangkap pola **non-linear** pada fitur numerik & kategorikal.
- Handal untuk data **imbalanced** melalui `scale_pos_weight`.
- Robust terhadap multikolinearitas.
- Mendukung **GPU acceleration** (`gpu_hist`).
- Stabil menghadapi outlier dan long-tail distributions.

#### **Hyperparameter (Alasan & Intuisi)**

| Hyperparameter        | Nilai | Alasan |
|----------------------|-------|--------|
| `n_estimators`   | 800 | mempelajari pola minoritas |
| `learning_rate` | 0.05 | model belajar lebih pelan tapi lebih stabil. mencegah overfitting |
| `max_depth`        | 6 | menangkap pola non-linear tanpa overfit |
| `subsample`      | 0.9 | stochastic boosting |
| `colsample_bytree` | 0.9 | meningkatkan keragaman tree |
| `scale_pos_weight`   | scale_pos | fokus menangani fraud |
| `tree_method + predictor` | gpu_hist & gpu_predictor | Mempercepat training hingga 10–40x |
| `eval_metric` | logloss | Logloss lebih sensitif pada perubahan probabilitas kecil |

#### **Konteks Bisnis**
XGBoost menghasilkan **precision tertinggi**, cocok untuk:
- Auto-block transaksi mencurigakan
- Situasi di mana **false positive harus ditekan**

#### **✅ Kelebihan**
- Akurasi sangat tinggi pada data tabular.
- Handal pada dataset imbalanced.
- GPU training sangat cepat.
- Robust terhadap missing value dan outlier.

#### **⚠️ Kekurangan**
- Tuning kompleks.
- Interpretabilitas rendah.
- Konsumsi memori tinggi.

---

### 3️. LightGBM — F1-score Terbaik 

LightGBM memberikan performa paling seimbang antara precision dan recall → **F1-score tertinggi**.

#### **Alasan Pemilihan**
- Efisien untuk data besar (300k+ baris).
- Leaf-wise growth → agresif menemukan pola fraud.
- Kuat terhadap outlier.
- Optimal untuk dataset tabular.
- Sangat cepat dengan GPU.

#### **Hyperparameter (Alasan & Intuisi)**

| Hyperparameter        | Nilai | Alasan |
|----------------------|-------|--------|
| `n_estimators`   | 800 | banyak tree agar learning rate kecil bisa optimal |
| `learning_rate` | 0.05 | menjaga stabilitas |
| `scale_pos_weight`   | scale_pos | untuk imbalance |
| `objective` | binary | cocok untuk fraud scoring |
| `device` | gpu | LightGBM GPU menggunakan histogram algorithm yang cepat dan efisien |

#### **Konteks Bisnis**
LightGBM ideal untuk:
- Fraud detection umum
- Hybrid decision engine
- Kebutuhan balance antara risiko & kenyamanan pengguna

#### **✅ Kelebihan**
- Training sangat cepat.
- Memory usage rendah.
- Sangat akurat di data tabular.
- Menangkap pola kompleks.

#### **⚠️ Kekurangan**
- Lebih mudah overfit dibanding XGBoost.
- Sensitif terhadap `num_leaves`.
- Membutuhkan encoding manual untuk kategori.

---

### 4️. CatBoost — Recall Terbaik

#### **Alasan Pemilihan**
- Native categorical encoding → aman tanpa preprocessing.
- Order-based boosting → stabil pada kategori.
- Sangat baik untuk fitur biner (AVS, CVV, 3DS).
- Recall tertinggi → menangkap fraud sebanyak mungkin.

#### **Hyperparameter (Alasan & Intuisi)**

| Hyperparameter        | Nilai | Alasan |
|----------------------|-------|--------|
| `iterations`     | 800 | banyak tree agar learning rate kecil bisa optimal |
| `depth`            | 6 | cukup menangkap interaksi fitur fraud yang kompleks. |
| `learning_rate` | 0.05 | fairness antar model |
| `scale_pos_weight`   | scale_pos | untuk imbalance |
| `loss_function` | Logloss | untuk konsistensi evaluasi |
| `task_type` + `devices` | GPU & 0 | GPU acceleration |

#### **Konteks Bisnis**
CatBoost ideal untuk:
- Sistem fraud alert
- Kasus di mana **false negative sangat mahal**

#### **✅ Kelebihan**
- Native handling categorical.
- Recall tertinggi.
- Minim tuning.
- Stabil terhadap overfitting.

#### **⚠️ Kekurangan**
- Training lebih lambat dari LightGBM.
- Ukuran model lebih besar.
- Interpretabilitas menengah.

---

### Threshold Tuning (Tahap Kedua Modeling)

Meskipun model menghasilkan probabilitas fraud, threshold default **0.5 tidak optimal** untuk dataset yang sangat imbalanced.

#### **Mengapa Threshold Tuning Penting?**
- Fraud hanya **2.2%**
- Model bisa mendapat akurasi 97% dengan memprediksi semua transaksi sebagai non-fraud
- Akurasi menjadi **tidak bermakna**
- Threshold 0.5 sering gagal “menangkap” fraud

#### **Tujuan Threshold Tuning**
- Meningkatkan recall → mendeteksi lebih banyak fraud
- Menjaga precision → mengurangi false positive
- Menyelaraskan model dengan strategi bisnis:
  - Auto-block
  - Review oleh fraud analyst
  - Hybrid decision engine

---

### Metode Threshold Tuning

#### **Threshold F1 (Balance Precision–Recall)**

\[
F1 = \frac{2PR}{P + R}
\]

Cocok untuk:
- Sistem auto-block
- Keputusan paling seimbang

---

#### **Threshold F2 (Recall Lebih Ditekankan)**

\[
F2 = \frac{5PR}{4P + R}
\]

Cocok untuk:
- Kasus di mana false negative sangat mahal
- Industri finansial berisiko tinggi

---

#### **Cost-Based Threshold (Biaya Fraud vs Review)**

\[
Cost = FN \times C_{fn} + FP \times C_{fp}
\]

Dimana:
- **FN lebih mahal** dari **FP**
- Digunakan untuk queue review analis

---

#### Manfaat Bisnis Threshold Tuning
- Mengurangi kerugian finansial
- Minim false positive
- Mengoptimalkan kerja analis
- Mendukung risk engine yang efisien
- Meningkatkan akurasi keputusan bisnis

---


## Evaluation

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

## Kesimpulan Akhir

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

## Future Work

- Integrasi real-time anomaly detection  
- SHAP-based interpretability  
- Fraud dashboard analytics  
- Deployment FastAPI/MLflow  
- Auto-retraining & drift monitoring  

---

