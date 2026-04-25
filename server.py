from flask import Flask, request, jsonify
import pandas as pd
import os
from datetime import datetime
import requests
import time
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
FILE = "data.csv"

# ===== TELEGRAM CONFIG =====
TOKEN = os.environ.get("TOKEN", "").strip()
CHAT_ID = os.environ.get("CHAT_ID", "").strip()

# ===== AI CONFIG =====
MODEL_PATH = "model_lstm.keras"
WINDOW = 7

model = None
scaler = MinMaxScaler()

try:
    model = load_model(MODEL_PATH)
    print("✅ Model LSTM loaded")
except Exception as e:
    print("⚠️ Model tidak ditemukan:", e)

# ===== GLOBAL =====
last_notif_time = 0
last_power_state = None  # 🔥 untuk deteksi ON/OFF

# ===== TELEGRAM =====
def kirim_notif(pesan):
    if not TOKEN or not CHAT_ID:
        print("❌ TOKEN kosong")
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    try:
        res = requests.post(url, json={
            "chat_id": CHAT_ID,
            "text": pesan,
            "parse_mode": "HTML"
        }, timeout=10)

        print("📩 TELEGRAM:", res.status_code)
    except Exception as e:
        print("❌ ERROR TELEGRAM:", e)

# ===== AI =====
def prediksi_besok(df):
    if model is None or len(df) < WINDOW:
        return None

    try:
        data = df[['biaya']].values
        scaled = scaler.fit_transform(data)

        last = scaled[-WINDOW:]
        X = np.array([last[:, 0]]).reshape((1, WINDOW, 1))

        pred = model.predict(X, verbose=0)
        hasil = scaler.inverse_transform(pred)

        return float(hasil[0][0])
    except Exception as e:
        print("❌ ERROR AI:", e)
        return None

# ===== EVALUASI AI =====
def hitung_error(df):
    try:
        if model is None or len(df) < WINDOW + 1:
            return None, None

        y_true = df['biaya'].values[WINDOW:]
        y_pred = []

        for i in range(WINDOW, len(df)):
            window_data = df['biaya'].values[i-WINDOW:i]
            scaled = scaler.fit_transform(window_data.reshape(-1,1))

            X = np.array([scaled[:,0]]).reshape((1, WINDOW, 1))
            pred = model.predict(X, verbose=0)
            hasil = scaler.inverse_transform(pred)

            y_pred.append(hasil[0][0])

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return float(mae), float(mape)

    except Exception as e:
        print("❌ ERROR EVALUASI:", e)
        return None, None

# ===== ROOT =====
@app.route('/')
def home():
    return "API AKTIF"

# ===== DATA MASUK =====
@app.route('/data', methods=['POST'])
def receive_data():
    global last_notif_time, last_power_state

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON"}), 400

        print("\n=== DATA MASUK ===")
        print(data)

        # ===== PARSE =====
        voltage = float(data.get('voltage', 0))
        current = float(data.get('current', 0))
        power = float(data.get('power', 0))
        kwh = float(data.get('kwh', 0))
        biaya = float(data.get('biaya', 0))

        now = datetime.now()

        # ===== DETEKSI ON/OFF =====
        current_state = "ON" if power > 1 else "OFF"

        if last_power_state is not None:
            if last_power_state == "ON" and current_state == "OFF":
                kirim_notif("⚫ Listrik mati!")

            elif last_power_state == "OFF" and current_state == "ON":
                kirim_notif("🟢 Listrik menyala kembali!")

        last_power_state = current_state

        # ===== DETEKSI EVENT =====
        event = "NORMAL"

        if power > 500:
            event = "HIGH_LOAD"

        elif power <= 1:
            event = "POWER_OFF"

        # SPIKE
        if os.path.exists(FILE):
            df_old = pd.read_csv(FILE)
            if len(df_old) > 2:
                prev = df_old['power'].iloc[-1]
                if abs(power - prev) > 200:
                    event = "SPIKE"

        print("EVENT:", event)

        # ===== SAVE =====
        row = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "day": now.day,
            "hour": now.hour,
            "voltage": voltage,
            "current": current,
            "power": power,
            "kwh": kwh,
            "biaya": biaya,
            "event": event
        }

        df_new = pd.DataFrame([row])

        if not os.path.exists(FILE):
            df_new.to_csv(FILE, index=False)
        else:
            df_new.to_csv(FILE, mode='a', header=False, index=False)

        df = pd.read_csv(FILE)

        # ===== ANALISIS =====
        total = df['biaya'].sum()
        rata = df['biaya'].mean()
        pred_bulanan = rata * 30

        # ===== AI =====
        prediksi_ai = prediksi_besok(df)

        # ===== EVALUASI =====
        mae, mape = hitung_error(df)

        # ===== NOTIF EVENT =====
        if event == "HIGH_LOAD":
            kirim_notif("🔴 HIGH LOAD terdeteksi!")

        elif event == "SPIKE":
            kirim_notif("⚠️ LONJAKAN DAYA!")

        # ===== TIMER =====
        current_time = time.time()

        if current_time - last_notif_time > 30:
            pesan = f"""
<b>⚡ MONITORING LISTRIK</b>

💡 Power: {power} W
💰 Total: Rp {int(total)}
📈 Bulanan: Rp {int(pred_bulanan)}
📊 Event: {event}
"""

            if prediksi_ai:
                pesan += f"\n🤖 Prediksi: Rp {int(prediksi_ai)}"

            if mae and mape:
                pesan += f"\n📉 MAE: {int(mae)} | MAPE: {round(mape,2)}%"

            kirim_notif(pesan)
            last_notif_time = current_time

        return jsonify({
            "status": "ok",
            "event": event,
            "power_state": current_state,
            "total": int(total),
            "prediksi_ai": int(prediksi_ai) if prediksi_ai else 0,
            "mae": mae,
            "mape": mape
        })

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ===== RUN =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)