from flask import Flask, render_template, request
import joblib
import os
import pandas as pd
import io
import base64
import matplotlib
matplotlib.use('Agg') # Biar gak error pas di server/Vercel
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load model
model_path = 'model/model_darah.pkl'
model = joblib.load(model_path) if os.path.exists(model_path) else None

def generate_plot(X, y, model):
    plt.figure(figsize=(8, 4))
    
    # Engineer sumbu X kronologis (asumsi data mulai 2021)
    X_viz = X.copy()
    X_viz['index_waktu'] = (X_viz['tahun'] - 2021) * 12 + X_viz['kode_bulan']

    # 1. Plot Data Aktual (Titik Biru)
    plt.scatter(X_viz['index_waktu'], y, color='blue', label='Data Aktual', s=30)
    
    # 2. Plot Garis Regresi (Garis Merah)
    y_pred = model.predict(X)
    plt.plot(X_viz['index_waktu'], y_pred, color='red', linewidth=2, label='Garis Regresi')
    
    plt.title('Analisis Regresi Linear: Tren Permintaan Darah PRC')
    plt.xlabel('Urutan Waktu (Bulan ke-n)')
    plt.ylabel('Jumlah Kantong')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None: return "Model not found!"
    
    try:
        tahun = int(request.form['tahun'])
        bulan = int(request.form['bulan'])
        
        # Prediksi
        prediksi = model.predict([[tahun, bulan]])
        hasil = round(prediksi[0])
        
        # Data untuk grafik
        df = pd.read_csv('dataset/permintaan_darah_per_bulan_berdasarkan_komponen_darah.csv')
        df_prc = df[df['komponen_darah'] == 'PRC'].copy()
        plot_url = generate_plot(df_prc[['tahun', 'kode_bulan']], df_prc['jumlah'], model)
        
        return render_template('index.html', 
                             prediction_text=f'Estimasi Stok Darah: {hasil} Kantong',
                             plot_url=plot_url, tahun=tahun, bulan=bulan)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)