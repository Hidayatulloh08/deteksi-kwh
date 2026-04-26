from flask import Flask, request, jsonify
from datetime import datetime
import pandas as pd
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.helper import to_float, load_csv_safe
from utils.notifier import kirim_notif
from utils.ai import prediksi_besok, load_ai
from config import THRESHOLD_DEFAULT, NOTIF_INTERVAL

app = Flask(__name__)

FILE = "data/data.csv"

# ===== INIT =====
last_notif_time = 0
last_power_state = None
last_status = None  # biar gak spam notif sama

# 🔥 LOAD AI SEKALI SAJA
load_ai()


@app.route('/')
def home():
    return "API AKTIF"


@app.route('/data', methods=['POST'])
def receive_data():
    global last_notif_time, last_power_state, last_status

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
        df_old = load_csv_safe(FILE)

        # ===== THRESHOLD ADAPTIVE =====
        if len(df_old) > 10 and 'power' in df_old.columns:
            mean = df_old['power'].mean()
            std = df_old['power'].std()
            threshold = mean + 2 * std
        else:
            threshold = THRESHOLD_DEFAULT

        prev_power = df_old['power'].iloc[-1] if len(df_old) > 0 else power

        # ===== STATUS DETECTION =====
        status = "NORMAL"

        if power > threshold:
            status = "HIGH_LOAD"
        elif power <= 1:
            status = "POWER_OFF"
        elif abs(power - prev_power) > 200:
            status = "SPIKE"

        # ===== 🔥 ABNORMAL PATTERN (upgrade) =====
        if len(df_old) > 5:
            last_powers = df_old['power'].tail(5)

            # spike berulang
            spike_count = sum(abs(last_powers.diff().fillna(0)) > 200)
            if spike_count >= 3:
                status = "REPEATED_SPIKE"

            # nyala di jam tidak wajar (00-04)
            if now.hour <= 4 and power > 50:
                status = "ABNORMAL_NIGHT_USAGE"

        # ===== AI =====
        prediksi_ai = prediksi_besok(df_old)

        # ===== SAVE DATA =====
        row = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "power": power,
            "biaya": biaya,
            "status": status,
            "threshold": threshold,
            "prediksi_ai": prediksi_ai or 0
        }

        pd.DataFrame([row]).to_csv(
            FILE,
            mode='a',
            header=not os.path.exists(FILE),
            index=False
        )

        # ===== NOTIF SYSTEM =====
        now_time = time.time()

        # ⚡ pertama kali
        if last_power_state is None:
            last_power_state = power > 1

        # ⚡ listrik nyala/mati
        if power > 1 and last_power_state is False:
            kirim_notif("⚡ Listrik MENYALA")
        elif power <= 1 and last_power_state is True:
            kirim_notif("⚫ Listrik MATI")

        last_power_state = power > 1

        # ⚡ kirim notif kalau status berubah (anti spam)
        if status != last_status:
            kirim_notif(f"🚨 {status} | {power}W")
            last_status = status

        # ⚡ normal tiap interval
        elif status == "NORMAL" and now_time - last_notif_time > NOTIF_INTERVAL:
            kirim_notif(f"⚡ NORMAL | {power}W")
            last_notif_time = now_time

        return jsonify({
            "status": "ok",
            "event": status,
            "threshold": threshold,
            "prediksi_ai": prediksi_ai
        })

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)