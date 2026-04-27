import numpy as np
import joblib
import os

# =========================
# GLOBAL
# =========================
model = None
scaler = None

# =========================
# PATH (AMAN UNTUK RAILWAY)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "model_lstm.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.save")

# =========================
# LOAD AI (AMAN)
# =========================
def load_ai():
    global model, scaler

    try:
        from tensorflow.keras.models import load_model

        print("📁 BASE:", BASE_DIR)

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
# FALLBACK (ANTI ERROR)
# =========================
def fallback_prediksi(df):
    try:
        if 'biaya' not in df.columns or len(df) == 0:
            return 0

        rata = df['biaya'].mean()

        # sedikit "AI rasa"
        if len(df) > 3:
            last = df['biaya'].tail(3).values
            trend = (last[-1] - last[0]) / 3
            return float(rata + trend)

        return float(rata)

    except:
        return 0


# =========================
# PREDIKSI UTAMA
# =========================
def prediksi_besok(df, window=7):
    try:
        # ===== VALIDASI =====
        if 'biaya' not in df.columns:
            return fallback_prediksi(df)

        if len(df) < window:
            return fallback_prediksi(df)

        # ===== JIKA AI TIDAK ADA =====
        if model is None or scaler is None:
            return fallback_prediksi(df)

        # ===== PREPROCESS =====
        data = df[['biaya']].values
        scaled = scaler.transform(data)

        last = scaled[-window:]
        X = np.array([last[:, 0]]).reshape((1, window, 1))

        # ===== PREDIKSI =====
        pred = model.predict(X, verbose=0)
        hasil = scaler.inverse_transform(pred)

        value = float(hasil[0][0])

        # ===== VALIDASI OUTPUT =====
        if np.isnan(value) or np.isinf(value):
            return fallback_prediksi(df)

        # clamp biar ga aneh
        if value < 0:
            value = 0

        if value > 1_000_000:
            value = fallback_prediksi(df)

        return value

    except Exception as e:
        print("❌ ERROR AI:", e)
        return fallback_prediksi(df)