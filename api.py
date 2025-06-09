from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

# Inisialisasi Flask
app = Flask(__name__)

# Konfigurasi
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'mobilenetv2_sibi_model.h5'

# Pastikan folder upload ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model berhasil dimuat")
except Exception as e:
    print(f"❌ Gagal memuat model: {str(e)}")
    exit(1)

# Daftar label SIBI (A-Y tanpa J)
CLASS_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
    'T', 'U', 'V', 'W', 'X', 'Y'
]

# Helper Functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_path, target_size=(224, 224)):
    """Mempersiapkan gambar untuk prediksi"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Routes
@app.route('/')
def home():
    return jsonify({
        "message": "API Klasifikasi SIBI",
        "status": "ready",
        "endpoints": {
            "/predict": "POST gambar untuk klasifikasi"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    # Validasi request
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file yang dikirim"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "Nama file kosong"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "error": "Format file tidak didukung",
            "allowed_extensions": list(ALLOWED_EXTENSIONS)
        }), 400
    
    try:
        # Simpan file sementara
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Proses gambar
        processed_img = prepare_image(filepath)
        
        # Prediksi
        predictions = model.predict(processed_img)
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_LABELS[predicted_index]
        confidence = float(np.max(predictions[0]))
        
        # Format respons
        response = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_labels": CLASS_LABELS,
            "all_predictions": {k: float(v) for k, v in zip(CLASS_LABELS, predictions[0])}
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": "Terjadi kesalahan saat memproses gambar",
            "details": str(e)
        }), 500
    
    finally:
        # Bersihkan file sementara
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)

# Jalankan server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)