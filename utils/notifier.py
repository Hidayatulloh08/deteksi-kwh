import requests
import os

# ambil dari environment (Railway)
TOKEN = os.environ.get("TOKEN", "").strip()
CHAT_ID = os.environ.get("CHAT_ID", "").strip()

print("✅ notifier loaded")

def kirim_notif(pesan):
    if not TOKEN or not CHAT_ID:
        print("❌ TOKEN / CHAT_ID kosong")
        return

    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        res = requests.post(url, json={
            "chat_id": CHAT_ID,
            "text": pesan
        }, timeout=10)

        print("📨 Notif:", res.text)

    except Exception as e:
        print("❌ ERROR TELEGRAM:", e)