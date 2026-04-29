from flask import Flask, request, jsonify
from datetime import datetime
import pandas as pd
import time
import os
import numpy as np
import requests

app = Flask(__name__)

# =========================
# CONFIG
# =========================
FILE = "data/data.csv"
STATE_FILE = "data/state.txt"

TOKEN = os.environ.get("TOKEN", "").strip()
CHAT_ID = os.environ.get("CHAT_ID", "").strip()

NOTIF_INTERVAL = 60  # detik (testing dulu 1 menit)

last_notif_time = 0
last_status = None

# =========================
# SETUP
# =========================
os.makedirs("data", exist_ok=True)

if not os.path.exists(FILE):
    pd.DataFrame(columns=[
        "timestamp","voltage","current","power","kwh","biaya","status"
    ]).to_csv(FILE, index=False)

# =========================
# HELPER
# =========================
def to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return 0.0

def kirim_notif(pesan):
    if not TOKEN or not CHAT_ID:
        print("❌ TOKEN / CHAT_ID KOSONG")
        return

    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        res = requests.post(url, json={
            "chat_id": CHAT_ID,
            "text": pesan
        }, timeout=10)

        print("📨 TELEGRAM:", res.text)

    except Exception as e:
        print("❌ TELEGRAM ERROR:", e)

# =========================
# STATE
# =========================
def load_state():
    try:
        if not os.path.exists(STATE_FILE):
            return None
        with open(STATE_FILE, "r") as f:
            return f.read().strip() == "1"
    except:
        return None

def save_state(val):
    try:
        with open(STATE_FILE, "w") as f:
            f.write("1" if val else "0")
    except:
        pass

# =========================
# ROUTE
# =========================
@app.route("/")
def home():
    return "API AKTIF 🔥"

# =========================
# MAIN API
# =========================
@app.route("/data", methods=["POST"])
def receive_data():
    global last_notif_time, last_status

    try:
        data = request.get_json()

        voltage = to_float(data.get("voltage"))
        current = to_float(data.get("current"))
        power   = to_float(data.get("power"))
        kwh     = to_float(data.get("kwh"))
        biaya   = to_float(data.get("biaya"))

        now = datetime.now()
        now_time = time.time()

        print("📥 DATA:", voltage, power)

        # =========================
        # 🔥 DETEKSI NYALA / MATI (FIX)
        # =========================
        is_on = voltage > 150 and power > 5

        last_state = load_state()

        print("DEBUG STATE:", last_state, "->", is_on)

        if last_state is None:
            save_state(is_on)

        elif is_on != last_state:
            if is_on:
                kirim_notif("⚡ LISTRIK MENYALA")
                print("NOTIF: MENYALA")
            else:
                kirim_notif("⚫ LISTRIK PADAM")
                print("NOTIF: PADAM")

            save_state(is_on)

        # =========================
        # STATUS SEDERHANA
        # =========================
        if not is_on:
            status = "OFF"
        elif power > 500:
            status = "HIGH"
        else:
            status = "NORMAL"

        # =========================
        # SAVE CSV
        # =========================
        row = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "voltage": voltage,
            "current": current,
            "power": power,
            "kwh": kwh,
            "biaya": biaya,
            "status": status
        }

        pd.DataFrame([row]).to_csv(FILE, mode="a", header=False, index=False)

        # =========================
        # 🔔 NOTIF BERKALA
        # =========================
        pesan = f"""
⚡ MONITORING LISTRIK

🔌 {voltage} V
💡 {power} W
📊 Status: {status}
"""

        if now_time - last_notif_time > NOTIF_INTERVAL:
            kirim_notif(pesan)
            last_notif_time = now_time

        return jsonify({"status": "ok", "state": is_on})

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)}), 500


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)