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
    print("⚠️ Model error:", e)

# ===== SMART ALERT STATE =====
last_status = {
    "high": False,
    "off": False,
    "spike": False
}

# ===== GLOBAL TIMER =====
last_notif_time = 0

# ===== BUDGET =====
budget = 350000


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
        print("ERROR TELEGRAM:", e)


# ===== AI =====
def prediksi_besok(df):
    try:
        if model is None or len(df) < WINDOW:
            return None

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


# ===== ROOT =====
@app.route('/')
def home():
    return "API AKTIF"


# ===== DATA =====
@app.route('/data', methods=['POST'])
def receive_data():
    global last_notif_time, last_status

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON"}), 400

        print("\n=== DATA MASUK ===")
        print(data)

        # ===== SAFE PARSE =====
        voltage = float(data.get('voltage', 0))
        current = float(data.get('current', 0))
        power = float(data.get('power', 0))
        kwh = float(data.get('kwh', 0))
        biaya = float(data.get('biaya', 0))

        now = datetime.now()

        row = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "day": now.day,
            "hour": now.hour,
            "voltage": voltage,
            "current": current,
            "power": power,
            "kwh": kwh,
            "biaya": biaya
        }

        df_new = pd.DataFrame([row])

        # ===== SAVE =====
        if not os.path.exists(FILE):
            df_new.to_csv(FILE, index=False)
        else:
            df_new.to_csv(FILE, mode='a', header=False, index=False)

        df = pd.read_csv(FILE)

        # ===== ANALISIS =====
        total = float(df['biaya'].sum())
        rata = float(df['biaya'].mean())
        pred_bulanan = rata * 30

        # ===== AI =====
        prediksi_ai = prediksi_besok(df)

        # ===== POLA =====
        jam_boros = 0
        hari_boros = 0

        if len(df) > 3:
            jam_boros = int(df.groupby('hour')['biaya'].mean().idxmax())
            hari_boros = int(df.groupby('day')['biaya'].sum().idxmax())

        # =========================
        # 🔥 SMART ALERT (ANTI SPAM)
        # =========================


# 🔥 SMART ALERT (ANTI SPAM + DEBUG)
# =========================

        print("POWER SEKARANG:", power)

        # 🔴 BEBAN TINGGI
       
        if power > 500:
            print("DETEKSI: BEBAN TINGGI (FORCE)")
            kirim_notif("🔴 TEST Beban tinggi terdeteksi!")
        else:
            last_status["high"] = False


        # ⚫ LISTRIK MATI
        if power <= 1:  # 🔥 FIX: jangan == 0 (sensor kadang tidak 0 persis)
            print("DETEKSI: LISTRIK MATI")
            if not last_status["off"]:
                kirim_notif("⚫ Listrik mati!")
                last_status["off"] = True
        else:
            last_status["off"] = False


        # ⚠️ LONJAKAN DAYA
        if len(df) > 2:
            prev = float(df['power'].iloc[-2])
            selisih = abs(power - prev)

            print("PREV:", prev, "NOW:", power, "SELISIH:", selisih)

            if selisih > 200:
                print("DETEKSI: LONJAKAN DAYA")
                if not last_status["spike"]:
                    kirim_notif("⚠️ Lonjakan daya terdeteksi!")
                    last_status["spike"] = True
            else:
                last_status["spike"] = False

        # =========================
        # ⏱ TIMER NOTIF
        # =========================
        current_time = time.time()

        if current_time - last_notif_time > 30:
            pesan = f"""
<b>⚡ MONITORING</b>

🔌 {voltage} V
⚡ {current} A
💡 {power} W

💰 Total: Rp {int(total)}
📈 Bulanan: Rp {int(pred_bulanan)}
⏰ Jam boros: {jam_boros}
📅 Hari boros: {hari_boros}
"""

            # 🔥 ALERT AI
            if prediksi_ai and prediksi_ai > budget:
                pesan += "\n⚠️ Prediksi over budget!"

            if prediksi_ai:
                pesan += f"\n🤖 Prediksi: Rp {int(prediksi_ai)}"

            kirim_notif(pesan)
            last_notif_time = current_time

        return jsonify({
            "status": "ok",
            "total": int(total),
            "prediksi_ai": int(prediksi_ai) if prediksi_ai else 0,
            "jam_boros": jam_boros,
            "hari_boros": hari_boros
        })

    except Exception as e:
        print("🔥 ERROR BESAR:", e)
        return jsonify({"error": str(e)}), 500


# ===== RUN =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)