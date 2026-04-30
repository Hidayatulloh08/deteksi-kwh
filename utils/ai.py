import numpy as np
import os

MODEL = None
SCALER = None


# =========================
# LOAD MODEL
# =========================
def load_ai():
    global MODEL, SCALER

    try:
        from tensorflow.keras.models import load_model
        import joblib

        if os.path.exists("model/model_lstm.keras"):
            MODEL = load_model("model/model_lstm.keras")
            print("✅ Model AI loaded")

        if os.path.exists("model/scaler.save"):
            SCALER = joblib.load("model/scaler.save")
            print("✅ Scaler loaded")

    except Exception as e:
        print("⚠️ AI load gagal:", e)
        MODEL = None
        SCALER = None


# =========================
# PREDIKSI BESOK
# =========================
def prediksi_besok(df):
    try:
        if MODEL is None or SCALER is None:
            return fallback(df)

        if len(df) < 10:
            return fallback(df)

        data = df["biaya"].values.reshape(-1, 1)

        scaled = SCALER.transform(data)

        X = []
        window = 5

        for i in range(window, len(scaled)):
            X.append(scaled[i-window:i])

        X = np.array(X)

        if len(X) == 0:
            return fallback(df)

        last_seq = X[-1].reshape(1, window, 1)

        pred_scaled = MODEL.predict(last_seq, verbose=0)
        pred = SCALER.inverse_transform(pred_scaled)[0][0]

        confidence = 0.8  # default

        return float(pred), float(confidence)

    except Exception as e:
        print("⚠️ AI ERROR:", e)
        return fallback(df)


# =========================
# FALLBACK (ANTI CRASH)
# =========================
def fallback(df):
    try:
        if len(df) == 0:
            return 0, 0.5

        rata = df["biaya"].mean()
        return float(rata), 0.5

    except:
        return 0, 0.5