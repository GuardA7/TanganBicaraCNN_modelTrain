Berikut adalah **langkah-langkah minimum (wajib saja)** untuk bisa mengakses API-nya:

---

## ✅ Langkah Wajib Menjalankan API `TanganBicaraTrainingCNN`

### 1. **Clone Repository**

```bash
git clone https://github.com/GuardA7/TanganBicaraCNN_modelTrain.git
cd TanganBicaraCNN_modelTrain
```

### 2. **Install Library yang Dibutuhkan**

```bash
pip install -r requirements.txt
```

> **Catatan**: Pastikan kamu sudah install Python ≥ 3.8

### 3. **Pastikan File Model Tersedia**

Pastikan file `mobilenetv2_sibi_model.h5` sudah berada di folder root proyek. Kalau belum, minta file model dari pembuat atau upload manual ke sana.

### 4. **Jalankan API Server**

```bash
python api.py
```

> Jika berhasil, server akan berjalan di:

```
http://localhost:5000/
```

---

### 🔍 Akses Endpoint Prediksi

Kamu bisa mengakses API dengan mengirim gambar ke:

```
http://localhost:5000/predict
```

Contoh dengan `curl`:

```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@nama_file.jpg"
```

---

Selesai! Sekarang API siap menerima gambar gesture tangan untuk diklasifikasi.

Jika kamu ingin, saya bisa bantu juga setup deploy ke server publik (ngrok, railway, dll).
