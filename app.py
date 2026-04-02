from flask import Flask, render_template, request
import joblib
import os
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

# 1. Load model
model_path = 'model/model_darah.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

# 2. Fungsi Helper untuk Membuat Grafik
def generate_plot(X, y, model):
    plt.figure(figsize=(8, 4))
    
    # Plot data asli (titik merah)
    plt.scatter(X['kode_bulan'], y, color='red', label='Data Asli')
    
    # Plot garis regresi (garis biru)
    y_pred = model.predict(X)
    plt.plot(X['kode_bulan'], y_pred, color='blue', linewidth=2, label='Garis Regresi')
    
    plt.title('Tren Kebutuhan Stok Darah PRC')
    plt.xlabel('Bulan (Kode)')
    plt.ylabel('Jumlah Kantong')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Konversi grafik ke format Base64 agar bisa tampil di HTML
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url

# 3. Route Utama
@app.route('/')
def index():
    return render_template('index.html')

# 4. Route Prediksi + Visualisasi
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model belum dibuat, jalankan train_model.py dulu bre!"
    
    if request.method == 'POST':
        try:
            # Ambil data dari form
            tahun = int(request.form['tahun'])
            bulan = int(request.form['bulan'])
            
            # Eksekusi Prediksi
            prediksi = model.predict([[tahun, bulan]])
            hasil = round(prediksi[0])
            
            # Load dataset untuk keperluan grafik
            path_data = 'dataset/permintaan_darah_per_bulan_berdasarkan_komponen_darah.csv'
            df = pd.read_csv(path_data)
            df_prc = df[df['komponen_darah'] == 'PRC'].copy()
            
            # Generate Grafik Tren
            plot_url = generate_plot(df_prc[['tahun', 'kode_bulan']], df_prc['jumlah'], model)
            
            return render_template('index.html', 
                                 prediction_text=f'Estimasi Stok Darah: {hasil} Kantong',
                                 plot_url=plot_url,
                                 tahun=tahun, 
                                 bulan=bulan)
        except Exception as e:
            return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)