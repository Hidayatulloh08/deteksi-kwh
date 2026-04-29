import numpy as np
import joblib
import os
import pandas as pd

model = None
scaler = None

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_lstm.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.save")


def load_ai():
    global model, scaler

    try:
        from tensorflow.keras.models import load_model

        if not os.path.exists(MODEL_PATH):
            print("❌ MODEL TIDAK ADA")
            return

        if not os.path.exists(SCALER_PATH):
            print("❌ SCALER TIDAK ADA")
            return

        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        print("✅ AI BERHASIL LOAD")

    except Exception as e:
        print("⚠️ AI OFF:", e)


def fallback_prediksi(df):
    try:
        if len(df) == 0 or "biaya" not in df.columns:
            return 0, 0.5

        rata = df["biaya"].mean()
        return float(rata), 0.6

    except:
        return 0, 0.5


def prediksi_besok(df, window=7):
    try:
        if model is None or scaler is None:
            return fallback_prediksi(df)

        if len(df) < window:
            return fallback_prediksi(df)

        if "biaya" not in df.columns:
            return fallback_prediksi(df)

        df = df.copy()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce"
            )
            df["hour"] = df["timestamp"].dt.hour.fillna(0)
            df["day"] = df["timestamp"].dt.day.fillna(0)
        else:
            df["hour"] = 0
            df["day"] = 0

        fitur = df[["biaya", "hour", "day"]].fillna(0).values

        scaled = scaler.transform(fitur)

        last = scaled[-window:]
        X = np.array([last])

        pred = model.predict(X, verbose=0)

        pred_value = pred[0][0]

        dummy = np.zeros((1, 3))
        dummy[0][0] = pred_value

        hasil = scaler.inverse_transform(dummy)

        value = float(hasil[0][0])

        if np.isnan(value) or np.isinf(value):
            return fallback_prediksi(df)

        return max(0, value), 0.85

    except Exception as e:
        print("❌ ERROR AI:", e)
        return fallback_prediksi(df)