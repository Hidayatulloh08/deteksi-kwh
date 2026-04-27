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
last_status = None

# 🔥 LOAD AI SEKALI
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

        # ===== AMBIL DATA =====
        voltage = to_float(data.get('voltage'))
        current = to_float(data.get('current'))
        power = to_float(data.get('power'))
        kwh = to_float(data.get('kwh'))
        biaya = to_float(data.get('biaya'))

        now = datetime.now()
        df_old = load_csv_safe(FILE)

        # ===== THRESHOLD =====
        if len(df_old) > 10 and 'power' in df_old.columns:
            mean = df_old['power'].mean()
            std = df_old['power'].std()
            threshold = mean + 2 * std
        else:
            threshold = THRESHOLD_DEFAULT

        prev_power = df_old['power'].iloc[-1] if len(df_old) > 0 else power

        # ===== STATUS =====
        status = "NORMAL"

        if power > threshold:
            status = "HIGH_LOAD"
        elif power <= 1:
            status = "POWER_OFF"
        elif abs(power - prev_power) > 200:
            status = "SPIKE"

        # ===== POLA LANJUTAN =====
        if len(df_old) > 5:
            last_powers = df_old['power'].tail(5)

            spike_count = sum(abs(last_powers.diff().fillna(0)) > 200)
            if spike_count >= 3:
                status = "REPEATED_SPIKE"

            if now.hour <= 4 and power > 50:
                status = "ABNORMAL_NIGHT_USAGE"

        # ===== AI =====
        prediksi_ai = prediksi_besok(df_old)

        # ===== ANALISIS =====
        total = df_old['biaya'].sum() if len(df_old) > 0 else biaya
        rata = df_old['biaya'].mean() if len(df_old) > 0 else biaya
        pred_bulanan = rata * 30

        jam_boros = 0
        hari_boros = 0

        if len(df_old) > 0 and 'timestamp' in df_old.columns:
            df_old['timestamp'] = pd.to_datetime(df_old['timestamp'], errors='coerce')

            df_old['hour'] = df_old['timestamp'].dt.hour
            df_old['day'] = df_old['timestamp'].dt.day

            boros = df_old[df_old['power'] > threshold]

            if not boros.empty:
                jam_mode = boros['hour'].mode()
                hari_mode = boros['day'].mode()

                jam_boros = int(jam_mode[0]) if not jam_mode.empty else 0
                hari_boros = int(hari_mode[0]) if not hari_mode.empty else 0

        # ===== SAVE DATA =====
        row = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "voltage": voltage,
            "current": current,
            "power": power,
            "kwh": kwh,
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

        # ===== FORMAT NOTIF =====
        pesan = (
            f"⚡ MONITORING LISTRIK\n\n"
            f"🔌 {round(voltage,1)} V\n"
            f"⚡ {round(current,2)} A\n"
            f"💡 {round(power,1)} W\n\n"
            f"💰 Total: Rp {int(total)}\n"
            f"📈 Bulanan: Rp {int(pred_bulanan)}\n"
            f"⏰ Jam boros: {jam_boros}\n"
            f"📅 Hari boros: {hari_boros}\n\n"
            f"🤖 Prediksi besok: Rp {int(prediksi_ai) if prediksi_ai else 0}"
        )

        # ===== NOTIF SYSTEM =====
        now_time = time.time()

        # ⚡ INIT STATE
        if last_power_state is None:
            last_power_state = power > 1

        # ⚡ listrik nyala/mati
        if power > 1 and last_power_state is False:
            kirim_notif("⚡ Listrik MENYALA")
        elif power <= 1 and last_power_state is True:
            kirim_notif("⚫ Listrik MATI")

        last_power_state = power > 1

        # ⚡ kirim notif kalau status berubah
        if status != last_status:
            kirim_notif(pesan)
            last_status = status

        # ⚡ kirim tiap interval (misal 10 menit)
        elif now_time - last_notif_time > NOTIF_INTERVAL:
            kirim_notif(pesan)
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


# ===== RUN =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)