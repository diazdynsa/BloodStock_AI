from flask import Flask, render_template, request
import joblib
import os
import pandas as pd
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# 1. Load Model (Pastikan file .pkl sudah ada di folder model)
model_path = 'model/model_darah.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

def generate_plot(df_prc, model_multiple):
    # Buat urutan waktu kronologis (1, 2, 3... 48)
    # Ini supaya sumbu X memanjang ke samping, bukan numpuk 1-12
    x_kronologis = (df_prc['tahun'] - df_prc['tahun'].min()) * 12 + df_prc['kode_bulan']
    y_aktual = df_prc['jumlah']

    plt.figure(figsize=(9, 5))
    
    # 1. Plot Titik Biru (Data Aktual)
    plt.scatter(x_kronologis, y_aktual, color='blue', label='Data Aktual', s=30, alpha=0.7)
    
    # 2. Plot Garis Merah (Tren Lurus)
    # Kita buat model simple khusus buat visualisasi biar garisnya GAK ZIG-ZAG
    model_simple = LinearRegression()
    X_simple = x_kronologis.values.reshape(-1, 1)
    model_simple.fit(X_simple, y_aktual)
    y_trend = model_simple.predict(X_simple)
    
    plt.plot(x_kronologis, y_trend, color='red', linewidth=2.5, label='Garis Regresi (Tren)')
    
    # Styling biar formal buat Jurnal
    plt.title('Analisis Regresi Linear: Tren Permintaan Darah PRC', fontsize=12, pad=15)
    plt.xlabel('Urutan Waktu (Bulan ke-n)', fontsize=10)
    plt.ylabel('Jumlah Kantong', fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Simpan ke Base64
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Error: File model_darah.pkl tidak ditemukan di folder model!"
    
    try:
        tahun = int(request.form['tahun'])
        bulan = int(request.form['bulan'])
        
        # Hitung Prediksi (Pake Multiple Regression: Tahun & Bulan)
        prediksi = model.predict([[tahun, bulan]])
        hasil = int(round(prediksi[0]))
        
        # Load data buat bikin grafik
        csv_path = 'dataset/permintaan_darah_per_bulan_berdasarkan_komponen_darah.csv'
        df = pd.read_csv(csv_path)
        df_prc = df[df['komponen_darah'] == 'PRC'].sort_values(['tahun', 'kode_bulan'])
        
        # Generate Grafik
        plot_url = generate_plot(df_prc, model)
        
        return render_template('index.html', 
                             prediction_text=f'Estimasi Stok Darah: {hasil} Kantong',
                             plot_url=plot_url, 
                             tahun=tahun, 
                             bulan=bulan)
    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)