from flask import Flask, request, jsonify
import pandas as pd
import time
import os
import sys
import numpy as np
import threading
import threading
from ml_pipeline.retrain import start_retrain
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.helper import to_float, load_csv_safe
from utils.notifier import kirim_notif
from utils.ai import prediksi_besok, load_ai
from config import THRESHOLD_DEFAULT, NOTIF_INTERVAL, BUDGET_BULANAN
from datetime import datetime, timedelta

app = Flask(__name__)

FILE = "data/data.csv"
ERROR_FILE = "data/error_log.csv"
STATE_FILE = "data/state.txt"

# ===== GLOBAL =====
last_notif_time = 0
last_status = None
last_data_time = time.time()
no_data_sent = False 

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
# STATE LISTRIK
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
# 🔥 HYBRID PROTECTION
# =========================
def deteksi_proteksi(voltage, power, df_old):

    if voltage < 50:
        return "PLN_MATI"

    if power > 800 and len(df_old) > 0:
        if abs(power - df_old["power"].iloc[-1]) > 300:
            return "KONSLETING"

    if voltage < 180 and power > 50:
        return "VOLTAGE_DROP"

    if power > 300:
        return "BOROS"

    if power > 100:
        return "WASPADA"

    if power <= 1:
        return "NO_LOAD"

    return "NORMAL"

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
    global last_notif_time, last_status, last_data_time

    try:
        ensure_csv()

        data = request.get_json()
        if not data:
            return jsonify({"error":"No JSON"}),400

        last_data_time = time.time()

        voltage = to_float(data.get("voltage", 0))
        current = to_float(data.get("current", 0))
        power   = to_float(data.get("power", 0))
        kwh     = to_float(data.get("kwh", 0))
        biaya   = to_float(data.get("biaya", 0))

        now = datetime.utcnow() + timedelta(hours=7)
        waktu = now.strftime("%d-%m-%Y %H:%M:%S")
        now_time = time.time()

        df_old = load_csv_safe(FILE)

        for col in ["power","biaya"]:
            if col not in df_old.columns:
                df_old[col] = 0

        if df_old.empty:
            df_old = pd.DataFrame(columns=["power","biaya"])

        # =========================
        # 🔌 STATUS LISTRIK (FIX)
        # =========================
        is_on = voltage >= 50

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
        # 🔥 HYBRID LABEL
        # =========================
        label = deteksi_proteksi(voltage, power, df_old)
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
        # 🤖 AI
        # =========================
        try:
            prediksi_ai, conf_ai = prediksi_besok(df_old)
        except:
            prediksi_ai = rata
            conf_ai = 0.5

        if prediksi_ai > BUDGET_BULANAN:
            ai_status = "OVER_BUDGET"
        elif prediksi_ai > rata * 1.2:
            ai_status = "NAIK"
        else:
            ai_status = "AMAN"

        mae = hitung_mae(df_old)
        mape = hitung_mape(df_old)
        confidence = int(conf_ai*100)

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

        # 🔍 DEBUG (letakkan DI SINI)
        print("📥 DATA MASUK:", row)
        print("📁 FILE PATH:", FILE)
        print("📊 FILE ADA:", os.path.exists(FILE))

        pd.DataFrame([row]).to_csv(
        FILE,
        mode="a",
        header=not os.path.exists(FILE),
        index=False
    )

        print("💾 BERHASIL SIMPAN")
        # PESAN
        # =========================
        if label == "KONSLETING":
            pesan = "🚨 BAHAYA KONSLETING!"
        elif label == "PLN_MATI":
            pesan = "⚫ PLN MATI!"
        elif label == "VOLTAGE_DROP":
            pesan = "⚠️ Tegangan turun!"
        elif label == "NO_LOAD":
            pesan = "💤 Tidak ada beban listrik"
        else:
            pesan = (
                f"⚡ MONITORING LISTRIK\n\n"
                f"🕒 {waktu}\n\n"
                f"🔌 {round(voltage,1)} V\n"
                f"⚡ {round(current,2)} A\n"
                f"💡 {round(power,1)} W\n\n"
                f"📊 Status: {label}\n"
                f"🤖 AI: {ai_status}\n\n"
                f"💰 Total: Rp {int(total)}\n"
                f"📈 Bulanan: Rp {int(pred_bulanan)}\n"
                f"📊 Trend: {trend}\n\n"
                f"🤖 Prediksi: Rp {int(prediksi_ai)}\n"
                f"📊 Confidence: {confidence}%"
            )

        # =========================
        # NOTIF
        # =========================
        if status != last_status:
            kirim_notif(pesan)
            last_status = status
            last_notif_time = now_time
        elif now_time - last_notif_time > NOTIF_INTERVAL:
            kirim_notif(pesan)
            last_notif_time = now_time

        # BONUS AI WARNING
        if ai_status == "OVER_BUDGET" and label == "NORMAL":
            kirim_notif("⚠️ Prediksi tagihan melebihi budget!")

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

# =========================
# CEK NO DATA
# =========================
def cek_listrik_mati():
    global last_data_time, no_data_sent

    while True:
        now = time.time()

        if now - last_data_time > 10:
            if not no_data_sent:
                last_state = load_last_state()

                if last_state is True:
                    kirim_notif("⚫ PLN MATI (NO DATA)")
                    save_last_state(False)

                no_data_sent = True
        else:
            no_data_sent = False

        time.sleep(5)

threading.Thread(target=cek_listrik_mati, daemon=True).start()
threading.Thread(target=start_retrain, daemon=True).start()
# 🔥 JALANKAN AUTO RETRAIN (BACKGROUND)
# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)