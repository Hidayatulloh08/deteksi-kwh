import numpy as np
import joblib
import os
import pandas as pd

model = None
scaler = None

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_lstm.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.save")

# =========================
# LOAD MODEL
# =========================
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


# =========================
# FALLBACK SAFE MODE
# =========================
def fallback_prediksi(df):
    try:
        if len(df) == 0 or "biaya" not in df.columns:
            return 0.0, 0.5

        rata = df["biaya"].mean()
        return float(rata), 0.6

    except:
        return 0.0, 0.5


# =========================
# MAIN PREDICTION ENGINE
# =========================
def prediksi_besok(df, window=14):
    try:
        if model is None or scaler is None:
            return fallback_prediksi(df)

        if len(df) < window:
            return fallback_prediksi(df)

        if "biaya" not in df.columns:
            return fallback_prediksi(df)

        df = df.copy()

        # =========================
        # FEATURE ENGINEERING
        # =========================
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["hour"] = df["timestamp"].dt.hour.fillna(0)
            df["day"] = df["timestamp"].dt.day.fillna(0)
        else:
            df["hour"] = 0
            df["day"] = 0

        # =========================
        # FEATURE MATRIX
        # =========================
        fitur = df[["biaya", "hour", "day"]].astype(float).values

        # =========================
        # SCALING
        # =========================
        scaled = scaler.transform(fitur)

        # =========================
        # SEQUENCE INPUT
        # =========================
        sequence = scaled[-window:]
        X = np.array([sequence])

        # =========================
        # PREDICTION
        # =========================
        pred_scaled = model.predict(X, verbose=0)[0][0]

        # =========================
        # INVERSE TRANSFORM (SAFE)
        # =========================
        n_features = scaler.n_features_in_ if hasattr(scaler, "n_features_in_") else 3

        temp = np.zeros((1, n_features))
        temp[0][0] = pred_scaled

        value = scaler.inverse_transform(temp)[0][0]

        # =========================
        # VALIDATION
        # =========================
        if np.isnan(value) or np.isinf(value):
            return fallback_prediksi(df)

        # =========================
        # CONFIDENCE SIMPLE (SAFE VERSION)
        # =========================
        last_actual = df["biaya"].iloc[-1]

        error = abs(value - last_actual)
        confidence = max(0.5, 1 - (error / (last_actual + 1)))

        return max(0.0, float(value)), float(confidence)

    except Exception as e:
        print("❌ ERROR AI:", e)
        return fallback_prediksi(df)