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

    # 2. Filtering Data 
    df_prc = df[df['komponen_darah'] == 'PRC'].copy()

    # 3. Menentukan Fitur (X) dan Target (y)
    X = df_prc[['tahun', 'kode_bulan']]
    y = df_prc['jumlah']

    # 4. Melatih Model
    model = LinearRegression()
    model.fit(X, y)

    # 5. Evaluasi Model
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("=== HASIL EVALUASI MODEL ===")
    print(f"Mean Absolute Error (MAE) : {mae:.2f}")
    print(f"Mean Squared Error (MSE)  : {mse:.2f}")
    print(f"R2 Score (Akurasi)        : {r2:.4f}")
    print("============================")

   
    if not os.path.exists('model'):
        os.makedirs('model')
    
    joblib.dump(model, 'model/model_darah.pkl')
    print("Model berhasil disimpan di 'model/model_darah.pkl'")