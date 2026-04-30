from flask import Flask, request, jsonify
import pandas as pd
import time
import os
import sys
import numpy as np
import threading
from datetime import datetime, timedelta

# 🔥 TAMBAHAN SINTA 2
from ml_pipeline.drift import check_drift, detect_anomaly
from ml_pipeline.fusion import fusion_engine
from ml_pipeline.load_classifier import classify_load
from ml_pipeline.retrain import retrain

# =========================
# PATH SETUP
# =========================
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.helper import to_float, load_csv_safe
from utils.notifier import kirim_notif
from utils.ai import prediksi_besok, load_ai
from config import NOTIF_INTERVAL

app = Flask(__name__)

# =========================
# FILE PATH
# =========================
FILE = "data/data.csv"
ERROR_FILE = "data/error_log.csv"

# =========================
# GLOBAL STATE
# =========================
SYSTEM_STATE = {
    "pln_on": True,
    "last_pln_on": True,  # penting untuk deteksi transisi
    "last_status": None,
    "last_notif_time": 0,
    "last_data_time": time.time(),
    "no_data_sent": False
}

LAST_RETRAIN = 0

load_ai()

# =========================
# INIT FILE
# =========================
def ensure_csv():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(FILE):
        pd.DataFrame(columns=[
            "timestamp","voltage","current",
            "power","kwh","biaya","status","label"
        ]).to_csv(FILE, index=False)

    if not os.path.exists(ERROR_FILE):
        pd.DataFrame(columns=["timestamp","mae","mape"]).to_csv(ERROR_FILE, index=False)

ensure_csv()

# =========================
# CORE LOGIC
# =========================
def is_pln_mati(voltage):
    return voltage <= 1

def is_load_normal(power):
    if power < 5:
        return "NO_LOAD"
    elif power > 800:
        return "OVERLOAD"
    return "NORMAL"

# =========================
# ROUTE
# =========================
@app.route("/")
def home():
    return "API AKTIF 🚀"

# =========================
# MAIN API
# =========================
@app.route("/data", methods=["POST"])
def receive_data():
    global LAST_RETRAIN

    try:
        ensure_csv()

        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON"}), 400

        SYSTEM_STATE["last_data_time"] = time.time()

        voltage = to_float(data.get("voltage", 0))
        current = to_float(data.get("current", 0))
        power   = to_float(data.get("power", 0))
        kwh     = to_float(data.get("kwh", 0))
        biaya   = to_float(data.get("biaya", 0))

        now = datetime.utcnow() + timedelta(hours=7)

        # =========================
        # PLN STATE
        # =========================
        pln_mati = is_pln_mati(voltage)
        SYSTEM_STATE["pln_on"] = not pln_mati

        # 🔥 DETEKSI PLN KEMBALI (JANGAN UPDATE DULU)
        pln_kembali = False
        if not SYSTEM_STATE["last_pln_on"] and SYSTEM_STATE["pln_on"]:
            pln_kembali = True

        # 🔥 DEBUG PLN STATE
        print("PLN STATE:",
              "NOW:", SYSTEM_STATE["pln_on"],
              "LAST:", SYSTEM_STATE["last_pln_on"],
              "KEMBALI:", pln_kembali)

        # =========================
        # LOAD ANALYSIS
        # =========================
        load_status = is_load_normal(power)
        load_type = classify_load(power)

        df_old = load_csv_safe(FILE)

        if len(df_old) > 0:
            df_old["timestamp"] = pd.to_datetime(df_old["timestamp"], errors='coerce')

        # =========================
        # STATISTICAL
        # =========================
        if len(df_old) < 5:
            power_mean, power_std = 0, 0
            voltage_mean, voltage_std = 0, 0
        else:
            power_mean = df_old["power"].mean()
            power_std = df_old["power"].std()
            voltage_mean = df_old["voltage"].mean()
            voltage_std = df_old["voltage"].std()

        anomaly = False
        if power_std > 0 and voltage_std > 0:
            anomaly = detect_anomaly(
                power, voltage,
                power_mean, power_std,
                voltage_mean, voltage_std
            )

        # =========================
        # LABELING
        # =========================
        if pln_mati:
            label = "PLN_MATI"

        elif current > 8 and voltage < 120 and power > 100:
            label = "KONSLETING"

        elif load_status == "NO_LOAD":
            label = "NO_LOAD"

        elif load_status == "OVERLOAD":
            label = "OVERLOAD"

        elif power_std > 10 and power > power_mean + 3 * power_std:
            label = "ANOMALY_SPIKE"

        elif voltage_std > 0 and voltage < voltage_mean - 2 * voltage_std:
            label = "VOLTAGE_ANOMALY"

        elif voltage < 180 and power > 50:
            label = "VOLTAGE_DROP"

        elif power > 300:
            label = "BOROS"

        elif power > 100:
            label = "WASPADA"

        else:
            label = "NORMAL"

        # =========================
        # AI
        # =========================
        try:
            prediksi_ai, conf_ai = prediksi_besok(df_old)
        except Exception as e:
            print("❌ AI ERROR:", e)
            prediksi_ai = biaya
            conf_ai = 0.5

        label = fusion_engine(label, anomaly, prediksi_ai, conf_ai)
        confidence = int(conf_ai * 100)

        if confidence < 60 and label == "CRITICAL_ANOMALY":
            label = "NORMAL"

        # 🔥 DEBUG DATA
        print("DATA MASUK:",
              "V:", voltage,
              "I:", current,
              "P:", power,
              "LABEL:", label,
              "CONF:", confidence)

        # =========================
        # SAVE
        # =========================
        row = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "voltage": voltage,
            "current": current,
            "power": power,
            "kwh": kwh,
            "biaya": biaya,
            "status": label,
            "label": label
        }

        pd.DataFrame([row]).to_csv(FILE, mode="a", header=False, index=False)

        # =========================
        # MESSAGE
        # =========================
        if pln_kembali and voltage > 180:
            pesan = "✅ PLN MENYALA KEMBALI"

        elif label in ["CRITICAL_SHORT", "CRITICAL_ANOMALY"]:
            pesan = "🚨 BAHAYA KONSLETING!"

        elif label in ["PLN_MATI", "CRITICAL_OFF"]:
            pesan = "⚫ PLN MATI!"

        else:
            pesan = f"⚡ Status: {label} | {voltage}V | {power}W"

        # =========================
        # NOTIF
        # =========================
        now_time = time.time()

        is_critical = label in ["CRITICAL_SHORT", "CRITICAL_ANOMALY", "CRITICAL_OFF"]

        kirim_interval = (now_time - SYSTEM_STATE["last_notif_time"] > NOTIF_INTERVAL)

        if (
            is_critical or
            pln_kembali or
            kirim_interval  # 🔥 ini yang bikin monitoring jalan lagi
        ):
            try:
                kirim_notif(pesan)
            except Exception as e:
                print("❌ TELEGRAM ERROR:", e)

            SYSTEM_STATE["last_status"] = label
            SYSTEM_STATE["last_notif_time"] = now_time
        # 🔥 UPDATE STATE DI AKHIR (PENTING)
        SYSTEM_STATE["last_pln_on"] = SYSTEM_STATE["pln_on"]

        return jsonify({
            "status": "ok",
            "label": label,
            "confidence": confidence
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


# =========================
# BACKGROUND MONITOR
# =========================
def cek_listrik_mati():
    while True:
        now = time.time()

        if now - SYSTEM_STATE["last_data_time"] > 10:
            if not SYSTEM_STATE["no_data_sent"]:
                kirim_notif("⚠️ SENSOR TIDAK MENGIRIM DATA")

                if not SYSTEM_STATE["pln_on"]:
                    kirim_notif("⚫ PLN MATI (NO DATA)")

                SYSTEM_STATE["no_data_sent"] = True
        else:
            SYSTEM_STATE["no_data_sent"] = False

        time.sleep(5)


threading.Thread(target=cek_listrik_mati, daemon=True).start()

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)