import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# 1. Load data
path_data = 'dataset/permintaan_darah_per_bulan_berdasarkan_komponen_darah.csv'

if not os.path.exists(path_data):
    print(f"Error: File {path_data} tidak ditemukan!")
else:
    df = pd.read_csv(path_data)
    # Ambil komponen PRC saja
    df_prc = df[df['komponen_darah'] == 'PRC'].copy()

    # 2. Fitur (X) dan Target (y)
    X = df_prc[['tahun', 'kode_bulan']]
    y = df_prc['jumlah']

    # 3. Latih Model
    model = LinearRegression()
    model.fit(X, y)

    # 4. Evaluasi
    y_pred = model.predict(X)
    print("=== HASIL EVALUASI MODEL ===")
    print(f"MAE : {mean_absolute_error(y, y_pred):.2f}")
    print(f"MSE : {mean_squared_error(y, y_pred):.2f}")
    print(f"R2  : {r2_score(y, y_pred):.4f}")

    # 5. Simpan Model
    if not os.path.exists('model'):
        os.makedirs('model')
    joblib.dump(model, 'model/model_darah.pkl')
    print("Model disimpan di model/model_darah.pkl")