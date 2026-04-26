import numpy as np
import joblib

model = None
scaler = None

# =========================
# LOAD MODEL
# =========================
def load_ai():
    global model, scaler
    try:
        from tensorflow.keras.models import load_model
        model = load_model("model/model_lstm.keras")
        scaler = joblib.load("model/scaler.save")
        print("✅ AI loaded")
    except Exception as e:
        print("⚠️ AI OFF:", e)


# =========================
# PREDIKSI
# =========================
def prediksi_besok(df, window=7):
    try:
        if model is None or scaler is None or len(df) < window:
            return None

        if 'biaya' not in df.columns:
            return None

        data = df[['biaya']].values
        scaled = scaler.transform(data)

        last = scaled[-window:]
        X = np.array([last[:, 0]]).reshape((1, window, 1))

        pred = model.predict(X, verbose=0)
        hasil = scaler.inverse_transform(pred)

        if np.isnan(hasil[0][0]) or np.isinf(hasil[0][0]):
            return None

        return float(hasil[0][0])

    except Exception as e:
        print("❌ ERROR AI:", e)
        return None