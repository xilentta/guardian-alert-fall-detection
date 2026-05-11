import numpy as np
import tensorflow as tf
import pandas as pd

class FallDetectionInference:
    def __init__(self, model_path='models/fall_detection_lstm.h5'):
        """Inisialisasi dan muat model LSTM"""
        print(f"Memuat model dari {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        self.window_size = 200
        self.n_features = 6

    def _convert_to_physical(self, raw_df):
        """
        Konversi data ADC mentah ke unit fisik (g dan deg/s)
        Sesuai spesifikasi sensor SisFall (ADXL345 & ITG3200)
        """
        df = raw_df.copy()
        # Kolom 0-2: Accel 1 (ADXL345) -> Konversi ke 'g'
        df.iloc[:, 0:3] = df.iloc[:, 0:3] * (32 / 8192)
        # Kolom 3-5: Gyro (ITG3200) -> Konversi ke 'deg/s'
        df.iloc[:, 3:6] = df.iloc[:, 3:6] * (1 / 14.375)
        return df.iloc[:, 0:6] # Ambil hanya 6 kolom pertama

    def predict(self, raw_data):
        """
        Menerima data sensor mentah (DataFrame atau Array) 
        dan mengembalikan prediksi.
        Input: Array/DF dengan bentuk (200, 9) atau (200, 6)
        """
        # 1. Pastikan format adalah DataFrame untuk kemudahan manipulasi
        if isinstance(raw_data, np.ndarray):
            raw_data = pd.DataFrame(raw_data)
        
        # 2. Konversi ke unit fisik
        processed_data = self._convert_to_physical(raw_data)
        
        # 3. Reshape menjadi format 3D untuk LSTM: (batch, timesteps, features)
        # Karena ini inferensi tunggal, batch = 1
        input_3d = processed_data.values.reshape(1, self.window_size, self.n_features)
        
        # 4. Prediksi
        prediction_prob = self.model.predict(input_3d, verbose=0)[0][0]
        
        # 5. Tentukan Label (Threshold default 0.5, bisa disesuaikan)
        label = "FALL (BAHAYA)" if prediction_prob > 0.5 else "ADL (AMAN)"
        confidence = prediction_prob if prediction_prob > 0.5 else (1 - prediction_prob)
        
        return {
            "label": label,
            "confidence": float(confidence),
            "raw_probability": float(prediction_prob)
        }

# Contoh penggunaan (untuk testing)
if __name__ == "__main__":
    # Inisialisasi pipeline
    pipeline = FallDetectionInference()
    
    # Simulasi data dummy (200 baris, 6 kolom sensor)
    dummy_data = np.random.uniform(1000, 2000, (200, 6))
    
    # Jalankan prediksi
    result = pipeline.predict(dummy_data)
    print(f"\nHasil Prediksi: {result['label']}")
    print(f"Tingkat Keyakinan: {result['confidence']:.2%}")
