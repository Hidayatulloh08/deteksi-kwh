import numpy as np
import os

try:
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler
except:
    load_model = None

model = None
scaler = MinMaxScaler()

WINDOW = 7  # harus sama dengan config

def load_ai():
    global model
    try:
        if load_model is None:
            print("⚠️ TensorFlow tidak tersedia")
            return

        if os.path.exists("model_listrik.h5"):
            model = load_model("model_listrik.h5")
            print("✅ Model AI berhasil dimuat")
        else:
            print("⚠️ Model tidak ditemukan, pakai fallback")

    except Exception as e:
        print("❌ ERROR LOAD MODEL:", e)
        model = None


def prediksi_besok(df):
    try:
        # fallback jika data kurang
        if df is None or len(df) < WINDOW:
            return 0, 0.3

        data = df["biaya"].values.reshape(-1, 1)

        # ===== fallback jika model tidak ada =====
        if model is None:
            rata = float(np.mean(data))
            return rata, 0.5

        # ===== scaling =====
        data_scaled = scaler.fit_transform(data)

        last_data = data_scaled[-WINDOW:]
        X = np.array([last_data])

        pred_scaled = model.predict(X, verbose=0)[0][0]
        pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]

        return float(pred_real), 0.85

    except Exception as e:
        print("❌ ERROR AI:", e)
        return float(df["biaya"].mean()), 0.4