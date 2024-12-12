import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# 1. Persiapan Data
data = {
    'Tahun': [2019, 2020, 2021, 2022, 2023, 2024],
    'Stunting': [464562, 556073, 982047, 977104, 884000, 728524]  # Data stunting
}
df = pd.DataFrame(data)

# Normalisasi data
scaler = MinMaxScaler(feature_range=(0, 1))
stunting_scaled = scaler.fit_transform(df['Stunting'].values.reshape(-1, 1))

# 2. Parameter dan Pembagian Data
look_back = 2  # Menggunakan 2 data sebelumnya untuk prediksi

# Pembagian data: Train (4 data pertama), Validation (1 data), Test (1 data terakhir)
train_size = 4
val_size = 1
test_size = 1

train = stunting_scaled[:train_size + look_back - 1]  # 4 data + look_back
val = stunting_scaled[train_size - look_back + 1:train_size + val_size + look_back - 1]
test = stunting_scaled[train_size + val_size - look_back + 1:]

# 3. Membuat Dataset
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

X_train, Y_train = create_dataset(train, look_back)
X_val, Y_val = create_dataset(val, look_back)
X_test, Y_test = create_dataset(test, look_back) if len(test) >= look_back else (np.array([]), np.array([]))

# Reshape data untuk LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
if len(X_test) > 0:
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 4. Membuat Model LSTM dengan Early Stopping
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(20, return_sequences=True, input_shape=(look_back, 1)),  # LSTM pertama
    tf.keras.layers.Dropout(0.2),  # Dropout untuk mengurangi overfitting
    tf.keras.layers.LSTM(10),  # LSTM kedua
    tf.keras.layers.Dropout(0.2),  # Dropout kedua
    tf.keras.layers.Dense(1)  # Layer output
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Callback Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 5. Melatih Model
history = model.fit(X_train, Y_train, 
                    epochs=200, 
                    batch_size=1, 
                    verbose=1, 
                    validation_data=(X_val, Y_val), 
                    callbacks=[early_stopping])

# 6. Evaluasi MAE dan MSE untuk 2019-2024
predictions_2019_2024 = []
current_input = stunting_scaled[:look_back]

for i in range(len(df)):
    pred = model.predict(current_input.reshape(1, look_back, 1), verbose=0)
    predictions_2019_2024.append(pred[0, 0])
    if i + look_back < len(stunting_scaled):
        current_input = np.append(current_input[1:], stunting_scaled[i + look_back]).reshape(look_back, 1)

predictions_2019_2024_rescaled = scaler.inverse_transform(np.array(predictions_2019_2024).reshape(-1, 1))
actual_values = df['Stunting'].values
mae = np.mean(np.abs(actual_values - predictions_2019_2024_rescaled.flatten()))
mse = np.mean((actual_values - predictions_2019_2024_rescaled.flatten()) ** 2)

print(f"\nMAE untuk tahun 2019-2024: {mae:.2f}")
print(f"MSE untuk tahun 2019-2024: {mse:.2f}")

# 7. Prediksi Masa Depan (2025-2030)
current_input = stunting_scaled[-look_back:]
predictions_2025_2030 = []

for _ in range(6):
    pred = model.predict(current_input.reshape(1, look_back, 1), verbose=0)
    predictions_2025_2030.append(pred[0, 0])
    current_input = np.append(current_input[1:], pred).reshape(look_back, 1)

predictions_2025_2030_rescaled = scaler.inverse_transform(np.array(predictions_2025_2030).reshape(-1, 1))

# Gabungkan hasil prediksi untuk semua tahun
all_years = list(range(2019, 2031))
all_predictions = np.concatenate([predictions_2019_2024_rescaled.flatten(), predictions_2025_2030_rescaled.flatten()])

# 8. Output Prediksi
print("\nPrediksi untuk tahun 2019-2030:")
for year, value in zip(all_years, all_predictions):
    print(f"Tahun {year}: {value:.2f}")

# 9. Plot Hasil
plt.figure(figsize=(12, 6))
plt.plot(df['Tahun'], df['Stunting'], label='Data Asli', marker='o', color='blue')
plt.plot(range(2019, 2025), predictions_2019_2024_rescaled, label='Prediksi (2019-2024)', linestyle='dotted', marker='s', color='red')
plt.plot(range(2025, 2031), predictions_2025_2030_rescaled, label='Prediksi (2025-2030)', linestyle='dashed', marker='x', color='orange')
plt.legend()
plt.xlabel('Tahun')
plt.ylabel('Kasus Stunting')
plt.title('Prediksi Kasus Stunting (2019-2030)')
plt.grid()
plt.show()