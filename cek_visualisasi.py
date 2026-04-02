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

    # 2. Filtering Data (Ambil komponen PRC)
    df_prc = df[df['komponen_darah'] == 'PRC'].copy()
    df_prc = df_prc.sort_values(by=['tahun', 'kode_bulan']).reset_index(drop=True)

    # 3. Buat Index Waktu Kronologis (Sumbu X)
    tahun_minimal = df_prc['tahun'].min()
    df_prc['index_waktu'] = (df_prc['tahun'] - tahun_minimal) * 12 + df_prc['kode_bulan']

    # 4. Hitung Garis Regresi untuk Visualisasi
    # Kita buat model sementara di sini khusus buat narik garis merah
    X_viz = df_prc[['index_waktu']]
    y_viz = df_prc['jumlah']
    
    model_viz = LinearRegression()
    model_viz.fit(X_viz, y_viz)
    y_pred_viz = model_viz.predict(X_viz)

    # 5. Membuat Visualisasi Lengkap
    plt.figure(figsize=(10, 6))
    
    # Plot Data Aktual (Titik Biru)
    plt.scatter(df_prc['index_waktu'], df_prc['jumlah'], color='blue', label='Data Aktual', s=30)
    
    # Plot Garis Regresi (Garis Merah)
    plt.plot(df_prc['index_waktu'], y_pred_viz, color='red', linewidth=2, label='Garis Regresi (Tren)')
    
    # Styling
    plt.title('Analisis Regresi Linear: Tren Permintaan Darah PRC\nKota Tasikmalaya', fontsize=14)
    plt.xlabel('Urutan Waktu (Bulan ke-n)', fontsize=12)
    plt.ylabel('Jumlah Permintaan (Kantong)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    print("Grafik dengan garis regresi siap dimunculkan...")
    plt.show()