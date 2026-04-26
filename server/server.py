from flask import Flask, request, jsonify
from datetime import datetime
import pandas as pd
import time
import os

from utils.helper import to_float, load_csv_safe
from utils.notifier import kirim_notif
from utils.ai import prediksi_besok

app = Flask(__name__)

FILE = "data/data.csv"

last_notif_time = 0
last_power_state = None  # untuk deteksi nyala/mati

@app.route('/')
def home():
    return "API AKTIF"

@app.route('/data', methods=['POST'])
def receive_data():
    global last_notif_time, last_power_state

    try:
        data = request.get_json()

        voltage = to_float(data.get('voltage'))
        current = to_float(data.get('current'))
        power = to_float(data.get('power'))
        kwh = to_float(data.get('kwh'))
        biaya = to_float(data.get('biaya'))

        now = datetime.now()
        df_old = load_csv_safe(FILE)

        # ===== THRESHOLD =====
        if len(df_old) > 10 and 'power' in df_old.columns:
            threshold = df_old['power'].mean() + 2 * df_old['power'].std()
        else:
            threshold = 500

        prev_power = df_old['power'].iloc[-1] if len(df_old) > 0 else power

        # ===== STATUS =====
        status = "NORMAL"

        if power > threshold:
            status = "HIGH_LOAD"
        elif power <= 1:
            status = "POWER_OFF"
        elif abs(power - prev_power) > 200:
            status = "SPIKE"

        # ===== AI =====
        prediksi_ai = prediksi_besok(df_old)

        # ===== SAVE =====
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

        # ===== NOTIF CERDAS =====
        now_time = time.time()

        # ⚡ listrik nyala / mati
        if last_power_state is None:
            last_power_state = power > 1

        if power > 1 and last_power_state is False:
            kirim_notif("⚡ Listrik MENYALA")
        elif power <= 1 and last_power_state is True:
            kirim_notif("⚫ Listrik MATI")

        last_power_state = power > 1

        # ⚡ abnormal langsung kirim
        if status in ["HIGH_LOAD", "SPIKE", "POWER_OFF"]:
            kirim_notif(f"🚨 {status} | {power}W")

        # ⚡ normal tiap 5 menit
        elif now_time - last_notif_time > 300:
            kirim_notif(f"⚡ NORMAL | {power}W")
            last_notif_time = now_time

        return jsonify({"status": "ok", "event": status})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)