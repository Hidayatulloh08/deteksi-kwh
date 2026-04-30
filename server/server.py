from flask import Flask, request, jsonify
import pandas as pd
import time
import os
import sys
import numpy as np
import threading

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.helper import to_float, load_csv_safe
from utils.notifier import kirim_notif
from utils.ai import prediksi_besok, load_ai
from config import THRESHOLD_DEFAULT, NOTIF_INTERVAL
from datetime import datetime, timedelta

app = Flask(__name__)

FILE = "data/data.csv"
ERROR_FILE = "data/error_log.csv"
STATE_FILE = "data/state.txt"

# ===== GLOBAL =====
last_notif_time = 0
last_status = None
last_data_time = time.time()

load_ai()

# =========================
# FILE SETUP
# =========================
def ensure_csv():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(FILE):
        pd.DataFrame(columns=[
            "timestamp","voltage","current",
            "power","kwh","biaya","status","label"
        ]).to_csv(FILE, index=False)

    if not os.path.exists(ERROR_FILE):
        pd.DataFrame(columns=[
            "timestamp","mae","mape"
        ]).to_csv(ERROR_FILE, index=False)

ensure_csv()

if not os.path.exists(STATE_FILE):
    with open(STATE_FILE, "w") as f:
        f.write("0")
# =========================
# STATE LISTRIK (PERSIST)
# =========================
def load_last_state():
    try:
        if not os.path.exists(STATE_FILE):
            return None
        with open(STATE_FILE, "r") as f:
            return f.read().strip() == "1"
    except:
        return None

def save_last_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            f.write("1" if state else "0")
    except:
        pass

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
        return round(np.mean(np.abs((real - pred)/real))*100, 2)
    except:
        return 0

# =========================
# ROUTE
# =========================
@app.route("/")
def home():
    return "API AKTIF 🚀"

# =========================
# API
# =========================
@app.route("/data", methods=["POST"])
def receive_data():
    global last_notif_time, last_status

    try:
        ensure_csv()

        data = request.get_json()
        if not data:
          return jsonify({"error":"No JSON"}),400

        global last_data_time
        last_data_time = time.time()

        print("DEBUG JSON:", data)
        voltage = to_float(data.get("voltage", 0))
        current = to_float(data.get("current", 0))
        power   = to_float(data.get("power", 0))
        kwh     = to_float(data.get("kwh", 0))
        biaya   = to_float(data.get("biaya", 0))

        now = datetime.utcnow() + timedelta(hours=7)
        waktu = now.strftime("%d-%m-%Y %H:%M:%S")
        now_time = time.time()
                
        df_old = load_csv_safe(FILE)

# pastikan kolom tidak hilang (FIX ERROR 'power')
        for col in ["power", "biaya"]:
            if col not in df_old.columns:
                df_old[col] = 0

        if df_old.empty:
            df_old = pd.DataFrame(columns=["power","biaya"])

        last_state = load_last_state()
        # =========================
        # 🔥 DETEKSI ON/OFF (FIX TOTAL)
        # =========================
        is_on = not (voltage < 50 or power < 1)
        print("DEBUG STATUS:",
            "V=", voltage,
            "P=", power,
            "C=", current,
            "=> ON" if is_on else "OFF")

        last_state = load_last_state()

        if last_state is None:
            save_last_state(is_on)

        elif is_on != last_state:
            if is_on:
                kirim_notif("⚡ Listrik MENYALA")
            else:
                kirim_notif("⚫ PLN MATI")

            save_last_state(is_on)

        # =========================
        # PROTECTION
        # =========================
        if voltage < 50:
            label = "PLN_MATI"
        elif power <= 1:
            label = "OFF"
        elif power > 800 and len(df_old)>0 and abs(power-df_old["power"].iloc[-1])>300:
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

        # =========================
        # ANALISIS
        # =========================
        total = biaya if len(df_old)==0 else df_old["biaya"].sum()
        rata  = biaya if len(df_old)==0 else df_old["biaya"].mean()
        pred_bulanan = rata * 30

        trend = "STABIL"
        if len(df_old)>5:
            last5 = df_old["biaya"].tail(5).values
            if last5[-1] > last5[0]: trend="NAIK 📈"
            elif last5[-1] < last5[0]: trend="TURUN 📉"

        # =========================
        # AI
        # =========================
        try:
            prediksi_ai, conf_ai = prediksi_besok(df_old)
        except:
            prediksi_ai = rata
            conf_ai = 0.5

        mae = hitung_mae(df_old)
        mape = hitung_mape(df_old)
        confidence = int(conf_ai*100)

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
            "status": status,
            "label": label
        }

        pd.DataFrame([row]).to_csv(FILE, mode="a", header=False, index=False)

        # =========================
        # PESAN
        # =========================
        if label == "KONSLETING":
            pesan = "🚨 BAHAYA KONSLETING!"
        elif label == "PLN_MATI":
            pesan = "⚫ PLN MATI!"
        elif label == "VOLTAGE_DROP":
            pesan = "⚠️ Tegangan turun!"
        else:
            pesan = (
                f"⚡ MONITORING LISTRIK\n\n"
                f"🕒 {waktu}\n\n"
                f"🔌 {round(voltage,1)} V\n"
                f"⚡ {round(current,2)} A\n"
                f"💡 {round(power,1)} W\n\n"
                f"📊 Status: {label}\n"
                f"💰 Total: Rp {int(total)}\n"
                f"📈 Bulanan: Rp {int(pred_bulanan)}\n"
                f"📊 Trend: {trend}\n\n"
                f"💡 Pemakaian aman\n\n"
                f"🤖 Prediksi: Rp {int(prediksi_ai)}\n"
                f"📊 Confidence: {confidence}%"
            )

        # =========================
        # NOTIF SMART
        # =========================
        if status != last_status:
            kirim_notif(pesan)
            last_status = status
            last_notif_time = now_time

        elif now_time - last_notif_time > NOTIF_INTERVAL:
            kirim_notif(pesan)
            last_notif_time = now_time

        return jsonify({
            "status":"ok",
            "label":label,
            "mae":mae,
            "mape":mape,
            "confidence":confidence
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error":str(e)}),500

def cek_listrik_mati():
    global last_data_time

    while True:
        now = time.time()

        if now - last_data_time > 10:
            print("⚠️ TIDAK ADA DATA → ANGGAP MATI")

            kirim_notif("⚫ PLN MATI (NO DATA)")

            last_data_time = now

        time.sleep(5)
        threading.Thread(target=cek_listrik_mati, daemon=True).start()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)