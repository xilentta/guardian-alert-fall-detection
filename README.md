# 🚨 Guardian Alert: Fall Detection System for Elderly Care

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Guardian Alert** adalah sistem deteksi jatuh berbasis kecerdasan buatan (AI) yang dirancang untuk meningkatkan keamanan lansia. Proyek ini menggunakan model Deep Learning **LSTM (Long Short-Term Memory)** yang dilatih menggunakan dataset **SisFall** untuk mendeteksi kejadian jatuh secara real-time melalui data sensor IMU (Inertial Measurement Unit).

## 🎯 Fitur Utama
- **Deteksi Real-time**: Memproses data sensor pada frekuensi 200 Hz dengan jendela waktu 1 detik (200 timesteps).
- **Akurasi Tinggi**: Mencapai akurasi keseluruhan **87%** pada seluruh dataset SisFall.
- **Minim False Alarm**: Memiliki nilai Recall ADL sebesar **93%**, memastikan kenyamanan pengguna tanpa gangguan alarm palsu yang berlebihan.
- **Dashboard Interaktif**: Visualisasi data EDA dan antarmuka inferensi model menggunakan Streamlit.

## 📊 Hasil Evaluasi Model (LSTM)
| Metrik | Hasil |
| :--- | :--- |
| **Akurasi Keseluruhan** | **87%** |
| **ADL Recall (Aman)** | **93%** |
| **FALL Precision (Jatuh)** | **84%** |
| **Macro Avg F1-Score** | **0.86** |

## 📂 Struktur Proyek
```text
guardian-alert-fall-detection/
├── app.py                     # Dashboard Utama Streamlit
├── requirements.txt           # Daftar Library Dependency
├── technical_report.pdf       # Laporan Teknis Lengkap (PDF)
├── README.md                  # Dokumentasi Proyek
├── models/
│   └── fall_detection_lstm.h5 # Model LSTM Final (.h5)
├── src/
│   └── inference_pipeline_dl.py # Skrip Inferensi Produksi
└── notebooks/
    └── fall_detection_analysis_final.ipynb # Notebook Eksperimen
```

## 🚀 Cara Menjalankan Secara Lokal

1. **Clone Repository**
   ```bash
   git clone https://github.com/username/guardian-alert-fall-detection.git
   cd guardian-alert-fall-detection
   ```

2. **Instalasi Dependency**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan Dashboard Streamlit**
   ```bash
   streamlit run app.py
   ```

## 🛠️ Spesifikasi Teknis
- **Sensor**: Accelerometer (ADXL345) & Gyroscope (ITG3200).
- **Input**: 6-axis raw sensor data (ADC values).
- **Preprocessing**: Konversi unit fisik (g & deg/s) dan normalisasi sekuensial.
- **Threshold Rekomendasi**: 0.8 (untuk meminimalkan false alarm).

## 📚 Referensi
- Sucerquia, A., López, J. F., & Vargas-Bonilla, J. F. (2016). *SisFall: An Open Source Fall and Activities Dataset*.
- Dicoding Indonesia - Capstone Project AI Mentorship.

---
**Project Lead:** [Nama Anda]
**Status:** Production Ready | Dicoding Capstone Project
