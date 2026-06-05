# 🛡️ Guardian Alert: Fall Detection System for Elderly Care

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Guardian Alert** adalah sistem deteksi jatuh otomatis berbasis kecerdasan buatan (*Artificial Intelligence*) yang dirancang khusus sebagai solusi proteksi non-invasif bagi kelompok lansia. Dengan memanfaatkan model Deep Learning berarsitektur **LSTM (Long Short-Term Memory)**, sistem ini mampu menganalisis sinyal deret waktu (*time-series*) dari sensor gerakan IMU secara *real-time* untuk membedakan aktivitas harian normal (ADL) dari insiden jatuh nyata.

---

## 📌 Daftar Isi
1. [Latar Belakang & Urgensi Medis](#-Latar-Belakang--Urgensi-Medis)
2. [Spesifikasi & Distribusi Dataset](#-Spesifikasi--Distribusi-Dataset)
3. [Arsitektur Model & Metodologi](#-Arsitektur-Model--Metodologi)
4. [Hasil Evaluasi & Performa](#-Hasil-Evaluasi--Performa)
5. [Struktur Direktori Proyek](#-Struktur-Direktori-Proyek)
6. [Petunjuk Instalasi & Setup Environment](#-Petunjuk-Instalasi--Setup-Environment)
7. [Cara Menjalankan Aplikasi](#-Cara-Menjalankan-Aplikasi)
8. [Rencana Implementasi Lokal](#-Rencana-Implementasi-Lokal)
9. [Referensi](#-Referensi)

---

## 🏥 Latar Belakang & Urgensi Medis

Insiden jatuh merupakan salah satu ancaman keselamatan fisik paling fatal bagi kelompok usia rentan (lansia). Berdasarkan data dari **World Health Organization (WHO)**, diperkirakan sekitar **30% lansia berusia 65 tahun ke atas mengalami insiden jatuh minimal sekali setiap tahun**.

Sistem ini dirancang untuk menyelesaikan tiga krisis utama dalam penanganan medis lansia:
*   **Krisis Golden Hour:** Risiko komplikasi klinis atau cedera permanen meningkat hingga dua kali lipat jika korban tidak dievakuasi dalam waktu 1 jam pertama setelah jatuh.
*   **Keterbatasan Tombol Manual:** Perangkat keselamatan konvensional berupa *panic button* fisik tidak dapat diandalkan apabila korban langsung pingsan atau mengalami cidera lumpuh mendadak.
*   **Kesenjangan Pengawasan:** Anggota keluarga atau pengasuh memiliki keterbatasan waktu, sehingga diperlukan sistem otomatis yang aktif memantau selama 24 jam penuh tanpa melanggar privasi visual (non-invasif).

---

## 📊 Spesifikasi & Distribusi Dataset

Model dikembangkan menggunakan **SisFall Dataset**, sebuah dataset publik terstandar yang berisi rekaman pergerakan berbasis sensor *wearable* dari 38 subjek (23 dewasa dan 15 lansia).

### 1. Komposisi Data Jendela (Data Windows)
Dataset ini memiliki karakteristik *imbalanced data* dengan total **153.705 window**, yang terbagi menjadi:
*   **ADL (Activities of Daily Living):** 101.686 window (66.1%) — Mencakup aktivitas berjalan, berbaring, duduk, dan naik tangga.
*   **FALL (Kejadian Jatuh):** 52.019 window (33.9%) — Mencakup jatuh ke depan, ke belakang, lateral, dan jatuh akibat pingsan.

### 2. Informasi Sensor & Frekuensi Sampling
Sistem mengekstrak data dari dua jenis sensor dengan frekuensi sampling tinggi sebesar **200 Hz** (200 baris data per 1 detik aktivitas):
*   **Akselerometer ADXL345** (Kolom 0–2): Mengukur tingkat percepatan gravitasi ($g$).
*   **Giroskop ITG3200** (Kolom 3–5): Mengukur kecepatan sudut atau rotasi ($deg/s$).

---

## 🧠 Arsitektur Model & Metodologi

Sistem ini meninggalkan metode rekayasa fitur manual (*feature engineering*) tradisional dan beralih ke pendekatan **Sequence-based Deep Learning**. 

### 1. Mengapa Memilih LSTM?
Model **LSTM (Long Short-Term Memory)** dipilih karena memiliki *memory cell* yang sangat adaptif dalam mengenali dependensi jangka panjang data *time-series*. LSTM mampu mendeteksi *spike* (hentakan) impulsif pada akselerometer saat fase *impact* jatuh, serta membedakannya dari pola ritmik teratur pada aktivitas harian biasa.

### 2. Formula Kalibrasi Data Mentah (ADC)
Sebelum data masuk ke model LSTM dengan bentuk input tensor `(1, 200, 6)`, data mentah ADC dikonversi terlebih dahulu ke satuan fisik melalui rumus berikut:
*   **Akselerometer (g):** $Raw \times \frac{32}{8192}$
*   **Giroskop (deg/s):** $Raw \times \frac{1}{14.375}$

---

## 📈 Hasil Evaluasi & Performa

Model dievaluasi secara menyeluruh menggunakan seluruh data pada dataset SisFall untuk menjamin performa yang representatif di dunia nyata.

### 1. Klasifikasi Metrik Utama
*   **Overall Accuracy:** 87%
*   **ADL Recall:** 93% (Sangat krusial untuk memastikan tingkat *False Alarm* yang rendah agar pengasuh tidak jenuh dengan alarm palsu).
*   **FALL Precision:** 84% (Menandakan alert tanda bahaya jatuh yang dikeluarkan sistem sangat valid).
*   **Macro Avg F1-Score:** 0.86 (Menunjukkan keseimbangan performa model yang stabil).

### 2. Confusion Matrix Summary
| Kondisi Aktual | Prediksi: ADL (Aman) | Prediksi: FALL (Jatuh) |
| :--- | :--- | :--- |
| **ADL (Aman)** | **94.194** (Benar) | **7.492** (False Alarm) |
| **FALL (Jatuh)** | **11.770** (Terlewat) | **40.249** (Benar) |

---

## 📁 Struktur Direktori Proyek

Hierarki folder di bawah ini dirancang secara modular dan terstruktur untuk memisahkan logika riset (*notebook*), ekstraksi fitur (*src*), dan visualisasi produk (*app*):

```text
.
├── Data/
│   ├── Processed/
│   │   └── file_metadata.csv
│   └── features/
│       ├── eda_summary.json
│       ├── feature_importance.csv
│       └── features_extracted.csv
├── models/
│   ├── fall_detection_lstm.h5         # Bobot Model LSTM Final (Format H5)
│   └── fall_detection_lstm.keras      # Model Produksi Final (Format Keras)
├── notebooks/
│   └── fall_detection_analysis.ipynb  # Notebook Eksperimen EDA & Pelatihan Model
├── src/
│   └── inference_pipeline_dl.py       # Pipa Pemuatan & Inferensi Model Independen
├── 06_streamlit_dashboard.py          # Skrip Eksperimen Komponen Dashboard
├── app.py                             # Skrip Utama Dashboard Aplikasi Streamlit
├── README.md                          # Dokumentasi Utama Proyek
├── requirements.txt                   # Daftar Dependensi Pustaka Python
└── technical_report.pdf               # Laporan Teknis Lengkap Proyek
