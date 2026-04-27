import numpy as np
import joblib
import os

# =========================
# GLOBAL
# =========================
model = None
scaler = None

# =========================
# PATH (ANTI ERROR RAILWAY)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "model_lstm.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.save")

# =========================
# LOAD AI
# =========================
def load_ai():
    global model, scaler

    try:
        from tensorflow.keras.models import load_model

        print("📁 BASE DIR:", BASE_DIR)
        print("📁 MODEL PATH:", MODEL_PATH)
        print("📁 SCALER PATH:", SCALER_PATH)

        # cek file
        if not os.path.exists(MODEL_PATH):
            print("❌ MODEL TIDAK DITEMUKAN")
            return

        if not os.path.exists(SCALER_PATH):
            print("❌ SCALER TIDAK DITEMUKAN")
            return

        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        print("✅ AI BERHASIL DI-LOAD")

    except Exception as e:
        print("⚠️ AI OFF:", e)


# =========================
# PREDIKSI
# =========================
def prediksi_besok(df, window=7):
    try:
        # cek AI
        if model is None or scaler is None:
            return None

        # cek data cukup
        if len(df) < window:
            return None

        if 'biaya' not in df.columns:
            return None

        # ===== PREPROCESS =====
        data = df[['biaya']].values
        scaled = scaler.transform(data)

        last = scaled[-window:]
        X = np.array([last[:, 0]]).reshape((1, window, 1))

        # ===== PREDICT =====
        pred = model.predict(X, verbose=0)
        hasil = scaler.inverse_transform(pred)

        value = float(hasil[0][0])

        # ===== VALIDASI =====
        if np.isnan(value) or np.isinf(value):
            return None

        return value

    except Exception as e:
        print("❌ ERROR PREDIKSI:", e)
        return None