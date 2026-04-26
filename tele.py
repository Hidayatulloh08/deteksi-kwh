import requests
import os

TOKEN = os.environ.get("TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

res = requests.post(url, json={
    "chat_id": CHAT_ID,
    "text": "✅ TEST NOTIF BERHASIL",
})

print(res.text)