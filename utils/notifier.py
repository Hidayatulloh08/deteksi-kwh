import requests
import os
import time

TOKEN = os.environ.get("TOKEN", "").strip()
CHAT_ID = os.environ.get("CHAT_ID", "").strip()

print("✅ notifier loaded")

last_send = 0

def kirim_notif(pesan):
    global last_send

    if not TOKEN or not CHAT_ID:
        print("❌ TOKEN / CHAT_ID kosong")
        return

    now = time.time()

    # =========================
    # 🔥 PRIORITY DETECTION
    # =========================
    PRIORITY_KEYWORDS = [
        "MENYALA",
        "MATI",
        "KONSLETING",
        "PLN MATI",
        "BAHAYA"
    ]

    is_priority = any(k in pesan.upper() for k in PRIORITY_KEYWORDS)

    # =========================
    # ANTI SPAM (HANYA NORMAL)
    # =========================
    if not is_priority:
        if now - last_send < 5:
            return

    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

        for i in range(2):  # retry 2x
            try:
                res = requests.post(url, json={
                    "chat_id": CHAT_ID,
                    "text": pesan
                }, timeout=5)

                print("📨 Notif:", res.text)

                # hanya update timer untuk notif normal
                if not is_priority:
                    last_send = now

                break

            except:
                time.sleep(1)

    except Exception as e:
        print("❌ ERROR TELEGRAM:", e)