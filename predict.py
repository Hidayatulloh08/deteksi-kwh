import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("data.csv")

dataset = data[['biaya']].values

scaler = MinMaxScaler()
scaled = scaler.fit_transform(dataset)

model = load_model("model_lstm.keras")

window = 7

last = scaled[-window:]
X_test = np.array([last[:,0]])
X_test = X_test.reshape((1, window, 1))

pred = model.predict(X_test)
pred_real = scaler.inverse_transform(pred)

hasil = pred_real[0][0]

print("Prediksi besok: Rp", int(hasil))

# analisis
rata = dataset.mean()
estimasi = rata * 30

print("Rata harian:", int(rata))
print("Estimasi bulanan:", int(estimasi))

if estimasi > 350000:
    print("⚠️ Over Budget")
else:
    print("✅ Aman")