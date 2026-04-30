import requests
import os

TOKEN = os.environ.get("TOKEN", "").strip()
CHAT_ID = os.environ.get("CHAT_ID", "").strip()

def kirim_notif(pesan):
    try:
        if not TOKEN or not CHAT_ID:
            print("⚠️ TOKEN / CHAT_ID belum diset")
            print("📩 PESAN:", pesan)
            return

        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

        payload = {
            "chat_id": CHAT_ID,
            "text": pesan,
            "parse_mode": "HTML"
        }

        r = requests.post(url, data=payload, timeout=5)

        if r.status_code == 200:
            print("✅ Notif terkirim")
        else:
            print("❌ Gagal kirim notif:", r.text)

    except Exception as e:
        print("❌ ERROR NOTIF:", e)