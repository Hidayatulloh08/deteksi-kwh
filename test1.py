import requests

url = "https://deteksi-kwh-production.up.railway.app/data"

data = {
    "voltage": 220,
    "current": 1.5,
    "power": 300,
    "kwh": 0.3,
    "biaya": 500
}

res = requests.post(url, json=data)
print(res.text)