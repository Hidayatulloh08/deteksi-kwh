import requests
import os

# =========================
# CONFIG
# =========================
TOKEN = os.environ.get("TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")


# =========================
# SEND TELEGRAM
# =========================
def kirim_notif(pesan):
    if not TOKEN or not CHAT_ID:
        print("❌ TOKEN / CHAT_ID belum diset")
        print("PESAN:", pesan)
        return False

    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

        payload = {
            "chat_id": CHAT_ID,
            "text": pesan
        }

        res = requests.post(url, json=payload, timeout=5)

        if res.status_code != 200:
            print("❌ Gagal kirim Telegram:", res.text)
            return False

        print("📨 NOTIF TERKIRIM")
        return True

    except Exception as e:
        print("❌ ERROR TELEGRAM:", e)
        return False