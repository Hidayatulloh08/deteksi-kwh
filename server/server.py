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
ERROR_FILE = "data/error_log.csv"

# ===== GLOBAL =====
last_notif_time = 0
last_power_state = None
last_status = None

# ===== LOAD AI =====
load_ai()


# =========================
# AUTO CREATE CSV
# =========================
def ensure_csv():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(FILE):
        pd.DataFrame(columns=[
            "timestamp", "voltage", "current",
            "power", "kwh", "biaya",
            "status", "label"
        ]).to_csv(FILE, index=False)
        print("✅ data.csv dibuat")

    if not os.path.exists(ERROR_FILE):
        pd.DataFrame(columns=[
            "timestamp", "mae", "mape"
        ]).to_csv(ERROR_FILE, index=False)
        print("✅ error_log.csv dibuat")


ensure_csv()


# =========================
# METRIC
# =========================
def hitung_mae(df):
    try:
        if len(df) < 5:
            return 0

        real = df["biaya"].values
        pred = df["biaya"].shift(1).bfill().values

        return int(np.mean(np.abs(real - pred)))
    except:
        return 0


def hitung_mape(df):
    try:
        if len(df) < 5:
            return 0

        real = df["biaya"].values
        pred = df["biaya"].shift(1).bfill().values

        real = np.where(real == 0, 1, real)

        return round(np.mean(np.abs((real - pred) / real)) * 100, 2)
    except:
        return 0


@app.route("/")
def home():
    return "API AKTIF 🚀"


# =========================
# API DATA
# =========================
@app.route("/data", methods=["POST"])
def receive_data():
    global last_notif_time
    global last_power_state
    global last_status

    try:
        ensure_csv()

        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON"}), 400

        # ===== AMBIL DATA =====
        voltage = to_float(data.get("voltage"))
        current = to_float(data.get("current"))
        power = to_float(data.get("power"))
        kwh = to_float(data.get("kwh"))
        biaya = to_float(data.get("biaya"))

        now = datetime.now()
        df_old = load_csv_safe(FILE)

        # ===== FIX KOLOM =====
        if df_old.empty:
            df_old = pd.DataFrame(columns=[
                "timestamp", "power", "biaya"
            ])

        if "power" not in df_old.columns:
            df_old["power"] = 0

        if "biaya" not in df_old.columns:
            df_old["biaya"] = 0

        # ===== FIX TIME FEATURE =====
        if "timestamp" in df_old.columns:
            df_old["timestamp"] = pd.to_datetime(
                df_old["timestamp"], errors="coerce"
            )
            df_old["hour"] = df_old["timestamp"].dt.hour.fillna(0)
            df_old["day"] = df_old["timestamp"].dt.day.fillna(0)

        # ===== ANTI SPAM LOW POWER =====
        if power < 5:
            return jsonify({"status": "skip low power"})

        # ===== THRESHOLD =====
        if len(df_old) > 10:
            mean = df_old['power'].mean()
            std = df_old['power'].std()
            threshold = mean + 2 * std

            if threshold < 100:
                threshold = 100
        else:
            threshold = THRESHOLD_DEFAULT

        prev_power = df_old["power"].iloc[-1] if len(df_old) > 0 else power

        # =========================
        # SMART ELECTRICAL PROTECTION
        # =========================
        if voltage < 50:
            label = "PLN_MATI"

        elif power <= 1:
            label = "OFF"

        elif power > 800 and abs(power - prev_power) > 300:
            label = "KONSLETING"

        elif voltage < 180 and power > 50:
            label = "VOLTAGE_DROP"

        elif power > 300:
            label = "BOROS"

        elif power > 100:
            label = "WASPADA"

        else:
            label = "NORMAL"

        status = label

        # ===== ANALISIS =====
        total = biaya if len(df_old) == 0 else df_old["biaya"].sum()
        rata = biaya if len(df_old) == 0 else df_old["biaya"].mean()
        pred_bulanan = rata * 30

        # ===== TREND =====
        trend = "STABIL"
        if len(df_old) > 5:
            last5 = df_old["biaya"].tail(5).values
            if last5[-1] > last5[0]:
                trend = "NAIK 📈"
            elif last5[-1] < last5[0]:
                trend = "TURUN 📉"

        # ===== AI =====
        try:
            prediksi_ai, conf_ai = prediksi_besok(df_old)
        except Exception as e:
            print("❌ ERROR AI:", e)
            prediksi_ai = rata
            conf_ai = 0.5

        mae = hitung_mae(df_old)
        mape = hitung_mape(df_old)
        confidence = int(conf_ai * 100)

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

        pd.DataFrame([row]).to_csv(FILE, mode="a", header=False, index=False)

        pd.DataFrame([{
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "mae": mae,
            "mape": mape
        }]).to_csv(ERROR_FILE, mode="a", header=False, index=False)

        # =========================
        # NOTIF
        # =========================
        if label == "KONSLETING":
            pesan = "🚨 BAHAYA KONSLETING!\nSegera cek instalasi listrik!"

        elif label == "PLN_MATI":
            pesan = "⚫ PLN MATI!\nTidak ada tegangan terdeteksi"

        elif label == "VOLTAGE_DROP":
            pesan = "⚠️ Tegangan turun!\nBerbahaya untuk perangkat elektronik"

        else:
            pesan = (
                f"⚡ MONITORING LISTRIK\n\n"
                f"🔌 {round(voltage,1)} V\n"
                f"⚡ {round(current,2)} A\n"
                f"💡 {round(power,1)} W\n\n"
                f"📊 Status: {label}\n"
                f"💰 Total: Rp {int(total)}\n"
                f"📈 Bulanan: Rp {int(pred_bulanan)}\n"
                f"📊 Trend: {trend}\n\n"
                f"🤖 Prediksi: Rp {int(prediksi_ai)}\n"
                f"📊 Confidence: {confidence}%\n\n"
                f"💡 {'Kurangi beban listrik besar' if label=='BOROS' else 'Pemakaian aman'}"
            )

        now_time = time.time()

        # ===== ON/OFF =====
        if last_power_state is None:
            last_power_state = power > 1

        if power > 1 and last_power_state is False:
            kirim_notif("⚡ Listrik MENYALA")

        elif power <= 1 and last_power_state is True:
            kirim_notif("⚫ Listrik MATI")

        last_power_state = power > 1

        # ===== NOTIF CONTROL =====
        if status != last_status:
            kirim_notif(pesan)
            last_status = status
            last_notif_time = now_time

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
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)