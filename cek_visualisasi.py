import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# 1. Load data
path_data = 'dataset/permintaan_darah_per_bulan_berdasarkan_komponen_darah.csv'

if not os.path.exists(path_data):
    print(f"Error: File {path_data} tidak ditemukan!")
else:
    df = pd.read_csv(path_data)
    df_prc = df[df['komponen_darah'] == 'PRC'].copy()

    # 2. Fitur Multiple (Tahun & Bulan)
    X = df_prc[['tahun', 'kode_bulan']]
    y = df_prc['jumlah']

    # 3. Latih Model
    model = LinearRegression()
    model.fit(X, y)

    # 4. Visualisasi
    plt.figure(figsize=(10, 6))

    # Plot Data Aktual (Titik Biru - Sesuai Web)
    plt.scatter(df_prc['kode_bulan'], y, color='blue', label='Data Aktual', alpha=0.6, s=40)

    # Plot Garis Regresi (Semuanya Merah - Sesuai Web)
    years = sorted(df_prc['tahun'].unique())
    for year in years:
        X_dummy = pd.DataFrame({
            'tahun': [year] * 12,
            'kode_bulan': range(1, 13)
        })
        y_pred = model.predict(X_dummy)
        # Kita set semua warna jadi RED biar konsisten sama web
        plt.plot(range(1, 13), y_pred, color='red', linewidth=2, alpha=0.8)

    # Tambahin satu garis merah dummy buat legend biar gak numpuk
    plt.plot([], [], color='red', label='Garis Regresi (Tren)')

    # Styling
    plt.title('Multiple Linear Regression: Tren Permintaan Darah PRC\nKota Tasikmalaya', fontsize=14)
    plt.xlabel('Bulan (1-12)', fontsize=12)
    plt.ylabel('Jumlah Permintaan (Kantong)', fontsize=12)
    plt.xticks(range(1, 13))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    print("Grafik sudah disesuaikan warnanya dengan Web (Merah & Biru).")
    plt.show()