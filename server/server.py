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
ANOMALI_FILE = "data/anomali_log.csv"

# ===== INIT =====
last_notif_time = 0
last_power_state = None
last_status = None

# LOAD AI
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

        # ===== THRESHOLD ADAPTIF =====
        if len(df_old) > 10 and 'power' in df_old.columns:
            mean = df_old['power'].mean()
            std = df_old['power'].std()
            threshold = mean + 2 * std
        else:
            threshold = THRESHOLD_DEFAULT

        prev_power = df_old['power'].iloc[-1] if len(df_old) > 0 else power

        # ===== STATUS =====
        status = "NORMAL"

        if power <= 1:
            status = "POWER_OFF"
        elif power > threshold:
            status = "HIGH_LOAD"
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

        # ===== LEVEL SISTEM (FIX UTAMA 🔥) =====
        level = "🟢 NORMAL"

        if status in ["HIGH_LOAD", "SPIKE", "REPEATED_SPIKE", "ABNORMAL_NIGHT_USAGE"]:
            level = "🔴 ANOMALI"
        elif power > threshold * 0.7:
            level = "🟡 BOROS"

        # ===== ANALISIS =====
        if len(df_old) > 0 and 'biaya' in df_old.columns:
            total = df_old['biaya'].sum()
            rata = df_old['biaya'].mean()
        else:
            total = biaya
            rata = biaya

        pred_bulanan = rata * 30

        # ===== TREND =====
        trend = "STABIL"
        if len(df_old) > 5:
            last5 = df_old['biaya'].tail(5).values
            if last5[-1] > last5[0]:
                trend = "NAIK 📈"
            elif last5[-1] < last5[0]:
                trend = "TURUN 📉"

        # ===== BOROS TIME =====
        jam_boros, hari_boros = 0, 0
        if len(df_old) > 0 and 'timestamp' in df_old.columns:
            df_old['timestamp'] = pd.to_datetime(df_old['timestamp'], errors='coerce')
            df_old['hour'] = df_old['timestamp'].dt.hour
            df_old['day'] = df_old['timestamp'].dt.day

            boros = df_old[df_old['power'] > threshold]

            if not boros.empty:
                jam_mode = boros['hour'].mode()
                hari_mode = boros['day'].mode()

                jam_boros = int(jam_mode.iloc[0]) if not jam_mode.empty else 0
                hari_boros = int(hari_mode.iloc[0]) if not hari_mode.empty else 0

        # ===== AI =====
        prediksi_ai = prediksi_besok(df_old)
        if prediksi_ai is None:
            prediksi_ai = rata

        # ===== REKOMENDASI =====
        if level == "🔴 ANOMALI":
            rekomendasi = "Periksa alat listrik ⚠️"
        elif level == "🟡 BOROS":
            rekomendasi = "Kurangi beban listrik"
        else:
            rekomendasi = "Penggunaan normal"

        # ===== SAVE DATA =====
        row = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "voltage": voltage,
            "current": current,
            "power": power,
            "kwh": kwh,
            "biaya": biaya,
            "status": status,
            "level": level
        }

        pd.DataFrame([row]).to_csv(
            FILE,
            mode='a',
            header=not os.path.exists(FILE),
            index=False
        )

        # ===== LOG ANOMALI =====
        if level == "🔴 ANOMALI":
            pd.DataFrame([{
                "timestamp": row["timestamp"],
                "power": power,
                "status": status
            }]).to_csv(
                ANOMALI_FILE,
                mode='a',
                header=not os.path.exists(ANOMALI_FILE),
                index=False
            )

        # ===== NOTIF FINAL 🔥 =====
        pesan = (
            f"⚡ MONITORING LISTRIK\n\n"
            f"🔌 {round(voltage,1)} V\n"
            f"⚡ {round(current,2)} A\n"
            f"💡 {round(power,1)} W\n\n"

            f"🚦 Level: {level}\n"
            f"📊 Status: {status}\n\n"

            f"💰 Total: Rp {int(total)}\n"
            f"📈 Bulanan: Rp {int(pred_bulanan)}\n"
            f"📊 Trend: {trend}\n\n"

            f"⏰ Jam boros: {jam_boros}\n"
            f"📅 Hari boros: {hari_boros}\n\n"

            f"🤖 Prediksi besok: Rp {int(prediksi_ai)}\n\n"

            f"💡 Rekomendasi: {rekomendasi}"
        )

        # ===== NOTIF SYSTEM =====
        now_time = time.time()

        if last_power_state is None:
            last_power_state = power > 1

        if power > 1 and last_power_state is False:
            kirim_notif("⚡ Listrik MENYALA")
        elif power <= 1 and last_power_state is True:
            kirim_notif("⚫ Listrik MATI")

        last_power_state = power > 1

        if status != last_status:
            kirim_notif(pesan)
            last_status = status

        elif now_time - last_notif_time > NOTIF_INTERVAL:
            kirim_notif(pesan)
            last_notif_time = now_time

        return jsonify({
            "status": "ok",
            "event": status,
            "level": level
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ===== API GRAFIK =====
@app.route('/grafik', methods=['GET'])
def grafik():
    if not os.path.exists(FILE):
        return jsonify([])

    df = pd.read_csv(FILE).tail(100)

    return jsonify({
        "timestamp": df['timestamp'].tolist(),
        "power": df['power'].tolist(),
        "biaya": df['biaya'].tolist()
    })


# ===== API ANOMALI =====
@app.route('/anomali', methods=['GET'])
def anomali():
    if not os.path.exists(ANOMALI_FILE):
        return jsonify([])

    df = pd.read_csv(ANOMALI_FILE).tail(50)
    return df.to_json(orient='records')


# ===== RUN =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)