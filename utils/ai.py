import numpy as np
import joblib
import os
import pandas as pd

# =========================
# GLOBAL
# =========================
model = None
scaler = None

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
# FEATURE TAMBAHAN (TIME FEATURE)
# =========================
def add_time_feature(df):
    try:
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        df['hour'] = df['timestamp'].dt.hour.fillna(0)
        df['day'] = df['timestamp'].dt.day.fillna(0)

        return df

    except:
        return df


# =========================
# FALLBACK (SMART)
# =========================
def fallback_prediksi(df):
    try:
        if 'biaya' not in df.columns or len(df) == 0:
            return 0, 0.5

        biaya = df['biaya'].values
        rata = np.mean(biaya)

        # trend sederhana
        if len(biaya) > 5:
            trend = (biaya[-1] - biaya[-5]) / 5
        else:
            trend = 0

        pred = rata + trend

        if pred < 0:
            pred = 0

        return float(pred), 0.6  # confidence rendah

    except:
        return 0, 0.5


# =========================
# PREDIKSI UTAMA (FIX SHAPE)
# =========================
def prediksi_besok(df, window=7):
    try:
        if 'biaya' not in df.columns:
            return fallback_prediksi(df)

        if len(df) < window:
            return fallback_prediksi(df)

        df = add_time_feature(df)

        # ===== fallback kalau model/scaler tidak ada
        if model is None or scaler is None:
            return fallback_prediksi(df)

        # ===== ambil fitur
        fitur = df[['biaya', 'hour', 'day']].fillna(0).values

        # ===== scaling
        scaled = scaler.transform(fitur)

        # ===== ambil window terakhir
        last = scaled[-window:]

        # shape: (1, timestep, fitur)
        X = np.array([last])

        # ===== predict
        pred = model.predict(X, verbose=0)

        # handle berbagai bentuk output
        if len(pred.shape) == 3:
            pred_biaya = pred[0][0][0]
        else:
            pred_biaya = pred[0][0]
        # inverse scaling khusus biaya
        dummy = np.zeros((1, 3))
        dummy[0][0] = pred_biaya
        hasil = scaler.inverse_transform(dummy)

        value = float(hasil[0][0])

        # ===== VALIDASI =====
        if np.isnan(value) or np.isinf(value):
            return fallback_prediksi(df)

        if value < 0:
            value = 0

        if value > 1_000_000:
            return fallback_prediksi(df)

        return value, 0.85  # confidence tinggi

    except Exception as e:
        print("❌ ERROR AI:", e)
        return fallback_prediksi(df)


# =========================
# ERROR METRIC (MAE)
# =========================
def hitung_error(df):
    try:
        if len(df) < 10 or 'biaya' not in df.columns:
            return 0

        real = df['biaya'].tail(5).values
        pred = np.mean(df['biaya'].iloc[-10:-5])

        mae = np.mean(np.abs(real - pred))

        return int(mae)

    except:
        return 0
# =========================
# ERROR METRIC TAMBAHAN (MAPE)
# =========================
def hitung_mape(df):
    try:
        if len(df) < 10 or 'biaya' not in df.columns:
            return 0

        real = df['biaya'].tail(5).values
        pred = np.mean(df['biaya'].iloc[-10:-5])

        # hindari division by zero
        real = np.where(real == 0, 1, real)

        mape = np.mean(np.abs((real - pred) / real)) * 100

        return round(mape, 2)

    except:
        return 0