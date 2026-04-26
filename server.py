from flask import Flask, request, jsonify
import pandas as pd
import os
from datetime import datetime
import requests
import time
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')  # ✅ FIX
import matplotlib.pyplot as plt
import joblib

app = Flask(__name__)
FILE = "data.csv"

# ===== TELEGRAM =====
TOKEN = os.environ.get("TOKEN", "").strip()
CHAT_ID = os.environ.get("CHAT_ID", "").strip()

# ===== AI CONFIG =====
MODEL_PATH = "model_lstm.keras"
WINDOW = 7
BUDGET = 350000

# ===== LOAD MODEL & SCALER =====
model = None
scaler = None

try:
    model = load_model(MODEL_PATH)
    print("✅ Model LSTM loaded")
except Exception as e:
    print("⚠️ Model error:", e)

try:
    scaler = joblib.load("scaler.save")
    print("✅ Scaler loaded")
except Exception as e:
    print("⚠️ Scaler error:", e)

# ===== GLOBAL =====
last_notif_time = 0
last_status = {"high": False, "off": False, "spike": False}

# =========================
# 🔧 SAFE FLOAT
# =========================
def to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return 0.0

# =========================
# 🔥 SAFE CSV LOADER
# =========================
def load_csv_safe():
    if not os.path.exists(FILE):
        return pd.DataFrame()

    try:
        return pd.read_csv(FILE, on_bad_lines='skip')
    except Exception as e:
        print("🔥 CSV RUSAK → RESET:", e)
        os.remove(FILE)
        return pd.DataFrame()

# =========================
# 🔔 TELEGRAM
# =========================
def kirim_notif(pesan):
    if not TOKEN or not CHAT_ID:
        print("❌ TOKEN kosong")
        return

    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": CHAT_ID,
            "text": pesan,
            "parse_mode": "HTML"
        }, timeout=10)
    except Exception as e:
        print("❌ ERROR TELEGRAM:", e)

# =========================
# 🤖 AI PREDIKSI
# =========================
def prediksi_besok(df):
    try:
        if model is None or scaler is None or len(df) < WINDOW:
            return None

        if 'biaya' not in df.columns:
            return None

        data = df[['biaya']].values
        scaled = scaler.transform(data)  # ✅ FIX (tidak fit ulang)

        last = scaled[-WINDOW:]
        X = np.array([last[:, 0]]).reshape((1, WINDOW, 1))

        pred = model.predict(X, verbose=0)
        hasil = scaler.inverse_transform(pred)

        if np.isnan(hasil[0][0]) or np.isinf(hasil[0][0]):
            return None

        return float(hasil[0][0])

    except Exception as e:
        print("❌ ERROR AI:", e)
        return None

# =========================
# ROOT
# =========================
@app.route('/')
def home():
    return "API AKTIF"

# =========================
# GET DATA
# =========================
@app.route('/get-data')
def get_data():
    df = load_csv_safe()
    return df.tail(50).to_json(orient='records')

# =========================
# 🚀 MAIN API
# =========================
@app.route('/data', methods=['POST'])
def receive_data():
    global last_notif_time, last_status

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON"}), 400

        voltage = to_float(data.get('voltage'))
        current = to_float(data.get('current'))
        power = to_float(data.get('power'))
        kwh = to_float(data.get('kwh'))
        biaya = to_float(data.get('biaya'))

        now = datetime.now()
        df_old = load_csv_safe()

        prev_power = df_old['power'].iloc[-1] if len(df_old) > 0 and 'power' in df_old.columns else power

        if len(df_old) > 10 and 'power' in df_old.columns:
            mean_power = df_old['power'].mean()
            std_power = df_old['power'].std()
            threshold = mean_power + (2 * std_power)
        else:
            threshold = 500

        status = "NORMAL"

        if power > threshold:
            status = "HIGH_LOAD_ADAPTIVE"
        elif power <= 1:
            status = "POWER_OFF"
        elif abs(power - prev_power) > 200:
            status = "SPIKE"

        prediksi_ai = prediksi_besok(df_old)

        mae, mape = 0, 0
        if prediksi_ai is not None and biaya != 0:
            error = abs(biaya - prediksi_ai)
            mae = error
            mape = (error / biaya) * 100

        columns = [
            "timestamp","day","hour","voltage","current","power","kwh","biaya",
            "status","threshold","prediksi_ai","mae","mape"
        ]

        row = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "day": now.day,
            "hour": now.hour,
            "voltage": voltage,
            "current": current,
            "power": power,
            "kwh": kwh,
            "biaya": biaya,
            "status": status,
            "threshold": threshold,
            "prediksi_ai": prediksi_ai if prediksi_ai else 0,
            "mae": mae,
            "mape": mape
        }

        df_new = pd.DataFrame([row])[columns]

        df_new.to_csv(
            FILE,
            mode='a',
            header=not os.path.exists(FILE),
            index=False
        )

        now_time = time.time()

        if now_time - last_notif_time > 300:
            kirim_notif(f"⚡ {power}W | {status}")
            last_notif_time = now_time

        return jsonify({
            "status": "ok",
            "event": status,
            "threshold": threshold,
            "mae": mae,
            "mape": mape
        })

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"error": str(e)}), 500


# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)