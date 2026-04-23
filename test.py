import requests

TOKEN = "8237045990:AAHjLOe62gX96guhsH1BQcXhkp83sxdcJLw"
CHAT_ID = "7510387628"

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

data = {
    "chat_id": CHAT_ID,
    "text": "🔥 TEST BERHASIL!"
}

print("Mengirim ke Telegram...")

try:
    res = requests.post(url, data=data)
    print("Status Code:", res.status_code)
    print("Response:", res.text)
except Exception as e:
    print("ERROR:", e)