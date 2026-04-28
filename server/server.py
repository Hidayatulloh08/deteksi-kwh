from flask import Flask, request, jsonify
from datetime import datetime
import pandas as pd
import time
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.helper import to_float, load_csv_safe
from utils.notifier import kirim_notif
from utils.ai import prediksi_besok, load_ai
from config import THRESHOLD_DEFAULT, NOTIF_INTERVAL

app = Flask(__name__)

FILE = "data/data.csv"
ANOMALI_FILE = "data/anomali_log.csv"
ERROR_FILE = "data/error_log.csv"

# ===== INIT =====
last_notif_time = 0
last_power_state = None
last_status = None

# LOAD AI
load_ai()


# =========================
# 🔥 MAE
# =========================
def hitung_mae(df):
    try:
        if len(df) < 5 or 'biaya' not in df.columns:
            return 0

        real = df['biaya'].values
        pred = df['biaya'].shift(1).bfill().values

        mae = np.mean(np.abs(real - pred))
        return int(mae)

    except:
        return 0


# =========================
# 🔥 MAPE (WAJIB PUBLIKASI)
# =========================
def hitung_mape(df):
    try:
        if len(df) < 5 or 'biaya' not in df.columns:
            return 0

        real = df['biaya'].values
        pred = df['biaya'].shift(1).bfill().values

        real = np.where(real == 0, 1, real)

        mape = np.mean(np.abs((real - pred) / real)) * 100
        return round(mape, 2)

    except:
        return 0


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
        label = "NORMAL"

        if power <= 1:
            status = "POWER_OFF"
            label = "OFF"

        elif power > threshold:
            status = "HIGH_LOAD"
            label = "BOROS"

        elif abs(power - prev_power) > 200:
            status = "SPIKE"
            label = "ANOMALI"

        # ===== POLA LANJUTAN =====
        if len(df_old) > 5:
            last_powers = df_old['power'].tail(5)
            spike_count = sum(abs(last_powers.diff().fillna(0)) > 200)

            if spike_count >= 3:
                status = "REPEATED_SPIKE"
                label = "ANOMALI"

            if now.hour <= 4 and power > 50:
                status = "ABNORMAL_NIGHT_USAGE"
                label = "ANOMALI"

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

        # ===== POLA =====
        habit = "Normal"
        jam_boros, hari_boros = 0, 0

        if len(df_old) > 0 and 'timestamp' in df_old.columns:
            df_old['timestamp'] = pd.to_datetime(df_old['timestamp'], errors='coerce')
            df_old['hour'] = df_old['timestamp'].dt.hour
            df_old['day'] = df_old['timestamp'].dt.day

            boros = df_old[df_old['power'] > threshold]

            if not boros.empty:
                jam_boros = int(boros['hour'].mode()[0])
                hari_boros = int(boros['day'].mode()[0])

                if jam_boros <= 4:
                    habit = "Malam tinggi 🌙"
                elif 17 <= jam_boros <= 21:
                    habit = "Jam sibuk 🔥"

        # ===== AI =====
        prediksi_ai, conf_ai = prediksi_besok(df_old)

        if prediksi_ai is None:
            prediksi_ai = rata
            conf_ai = 0.5

        # ===== ERROR =====
        mae = hitung_mae(df_old)
        mape = hitung_mape(df_old)

        # ===== CONFIDENCE FINAL =====
        confidence = int(conf_ai * 100)

        # ===== STATUS BIAYA =====
        status_biaya = "HEMAT ✅" if pred_bulanan < 300000 else "BOROS ⚠️"

        # ===== REKOMENDASI =====
        if label == "BOROS":
            rekomendasi = "Kurangi penggunaan alat daya besar"
        elif label == "ANOMALI":
            rekomendasi = "Cek instalasi listrik"
        else:
            rekomendasi = "Pemakaian stabil"

        # ===== SAVE =====
        row = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "voltage": voltage,
            "current": current,
            "power": power,
            "kwh": kwh,
            "biaya": biaya,
            "status": status,
            "label": label
        }

        pd.DataFrame([row]).to_csv(
            FILE, mode='a',
            header=not os.path.exists(FILE),
            index=False
        )

        # ===== LOG ERROR =====
        pd.DataFrame([{
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "mae": mae,
            "mape": mape
        }]).to_csv(
            ERROR_FILE,
            mode='a',
            header=not os.path.exists(ERROR_FILE),
            index=False
        )

        # ===== NOTIF =====
        pesan = (
            f"⚡ MONITORING LISTRIK\n\n"
            f"🔌 {round(voltage,1)} V\n"
            f"⚡ {round(current,2)} A\n"
            f"💡 {round(power,1)} W\n\n"
            f"📊 Status: {label}\n"
            f"💰 Total: Rp {int(total)}\n"
            f"📈 Bulanan: Rp {int(pred_bulanan)}\n"
            f"📊 Trend: {trend}\n\n"
            f"⏰ Jam boros: {jam_boros}\n"
            f"📅 Hari boros: {hari_boros}\n"
            f"📊 Pola: {habit}\n\n"
            f"🤖 Prediksi: Rp {int(prediksi_ai)}\n"
            f"📉 MAE: {mae}\n"
            f"📊 MAPE: {mape}%\n"
            f"📊 Confidence: {confidence}%\n\n"
            f"💡 {status_biaya}\n"
            f"🧠 {rekomendasi}"
        )

        # ===== NOTIF CONTROL =====
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
            "label": label,
            "mae": mae,
            "mape": mape,
            "confidence": confidence
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500