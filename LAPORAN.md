# LAPORAN AKHIR PROYEK — Optimasi dan Implementasi CNN dalam Aplikasi Sederhana
## Klasifikasi Jenis Bunga (MobileNetV2 + Streamlit) — Tugas Kelompok 2, Week 8 (LO 3)

> **Catatan pengisian:** bagian bertanda **[TODO — isi setelah Run all notebook baru]** wajib diganti angka/screenshot hasil baru.
> Jangan memakai angka model lama (bug `zmodel`, val acc 90,74%) untuk laporan final.

**Kelompok:** [TODO — nama + NIM anggota]
**Tanggal:** [TODO]
**Repo:** https://github.com/Josheaa/Kerja_Kelompok_Artificial-Intelligence_CNN_Week-8
**Aplikasi:** [TODO — link Streamlit Cloud] (lokal: `python -m streamlit run app.py` → http://localhost:8501)

---

## 1. Pendahuluan

### 1.1 Latar belakang
Klasifikasi gambar bunga adalah contoh klasik penerapan Convolutional Neural Network (CNN).
Melatih CNN dari nol membutuhkan data besar dan waktu lama, sehingga proyek ini memakai
**transfer learning**: memakai kembali model MobileNetV2 yang sudah dilatih pada ImageNet,
lalu menyesuaikan bagian akhirnya untuk 5 jenis bunga.

### 1.2 Tujuan (sesuai soal)
1. Mengoptimasi model dengan hyperparameter tuning, dropout, batch normalization, dan transfer learning.
2. Membangun aplikasi web sederhana (Streamlit) yang menerima input gambar dan menampilkan hasil prediksi CNN.
3. Menguji model dengan gambar baru di luar dataset dan mendokumentasikan hasilnya.

### 1.3 Batasan masalah
Model hanya mengenali 5 kelas bunga (closed-set). Gambar di luar itu (mis. anggrek, Rafflesia,
screenshot) tetap dipaksa ke salah satu kelas — dibahas di Bab 5.

---

## 2. Dataset

### 2.1 Sumber data
Dataset `flower_photos` (TensorFlow example images), total **3670 file, 5 kelas**:
`"daisy"`, `"dandelion"`, `"roses"`, `"sunflowers"`, `"tulips"`.
Dibagi 80/20 dengan `seed=42`: **2936 latih / 734 validasi**, ukuran 224×224, batch 32.

### 2.2 Gambar uji baru
Folder `Contoh_Sample_Gambar_Bunga/` berisi 15 foto bunga (3 per kelas, di luar `flower_photos`)
ditambah file berprefix `ood_*` (di luar 5 kelas, mis. anggrek) untuk uji out-of-distribution.
Sel uji gambar baru di notebook memakai folder `gambar_baru/` dengan format nama
`<kelas>_<n>.jpg` dan prefix `ood_` untuk OOD; hasilnya disimpan ke `hasil_uji_gambar_baru.csv`.

---

## 3. Arsitektur Model

### 3.1 Baseline (sebelum optimasi)
MobileNetV2 (`include_top=False`, `weights="imagenet"`, beku) +
`GlobalAveragePooling2D` + `BatchNormalization` + `Dropout(0.3)` +
`Dense(128, relu)` + `Dropout(0.3)` + `Dense(5, softmax)`, optimizer Adam `lr=1e-3`,
loss `sparse_categorical_crossentropy`, 5 epoch. **Baseline ini belum memakai augmentasi.**

### 3.2 Hasil baseline
[TODO — isi setelah run baru: val acc, val loss, classification report, `confusion_matrix_baseline.png`]

---

## 4. Optimasi Model

### 4.1 Hyperparameter tuning (8 konfigurasi × 5 epoch)
Ruang pencarian: `lr` {1e-3, 3e-4, 1e-4} × `dropout` {0.3, 0.5} × `units` {128, 256},
`seed=42`, `clear_session()` tiap iterasi. Konfigurasi pertama = titik awal (bukan baseline murni).

[TODO — tempel tabel tuning (`tabel`) + tulis konfigurasi terbaik dan val acc-nya]

### 4.2 Augmentasi dan regularisasi
`RandomFlip` + `RandomRotation(0.1)` + `RandomZoom(0.1)` aktif saat training,
ditambah Dropout dan BatchNormalization. EarlyStopping (`patience=3`,
`restore_best_weights=True`) + ReduceLROnPlateau mencegah overfitting.

### 4.3 Fine-tuning
30 layer teratas MobileNetV2 dibuka (`BatchNormalization` tetap beku),
dilanjut 6 epoch dengan `lr=1e-5`.

### 4.4 Ringkasan sebelum vs sesudah
[TODO — tempel tabel `ringkas`: Baseline vs Tuning+fine-tuning (val acc, macro F1, val loss)]

---

## 5. Aplikasi Web (Streamlit)

### 5.1 Cara menjalankan
1. `pip install -r requirements.txt`
2. `python -m streamlit run app.py`
3. Buka http://localhost:8501 (atau link Cloud di atas).

### 5.2 Alur dan fitur
Judul + kotak hijau Kesimpulan; sumber gambar **Upload** atau **Contoh_Sample_Gambar_Bunga**
(dropdown, hanya `.jpg/.jpeg/.png`); preprocessing di dalam model (Lambda
`mobilenet_v2.preprocess_input`, tanpa preprocess ganda); hasil berupa prediksi +
confidence tiap kelas (progress bar); foto HP dikoreksi rotasinya (EXIF).

### 5.3 Batas Unidentified 75%
Confidence < 75% ditampilkan sebagai **Unidentified (xx,xx%)** — model belum yakin,
kemungkinan bunga di luar 5 jenis. Ambang ini [TODO — pertahankan/ubah setelah evaluasi
tabel batas pada data OOD].

[TODO — screenshot: 1 prediksi benar + 1 kasus Unidentified]

---

## 6. Pengujian dan Evaluasi Akhir

### 6.1 Metrik validasi
[TODO — classification report final + `confusion_matrix.png` + analisis kesalahan terbesar
per kelas, mis. pasangan yang tertukar dan alasan visualnya]

### 6.2 Uji gambar baru
[TODO — tempel `hasil_uji_gambar_baru.csv` sebagai tabel: file, label asli, prediksi,
confidence, BENAR/SALAH + akurasinya]

### 6.3 Uji di luar dataset (OOD)
Anggrek (`ood_anggrek_*`) dan Rafflesia pada pengujian awal: [TODO — tulis hasil baru:
prediksi, confidence, Unidentified muncul/tidak]. Model closed-set wajar yakin-salah pada
OOD (softmax selalu membagi 100% ke kelas yang dikenal) — ini temuan, bukan kegagalan
pemenuhan soal.

### 6.4 Keterbatasan yang diakui
1. Val set dipakai untuk memilih konfigurasi, early stopping, dan laporan akhir
   (sedikit optimistis); gambar baru menjadi pemeriksa independen.
2. Perbandingan sebelum–sesudah mengukur efek gabungan tuning + augmentasi + fine-tuning
   (baseline sudah transfer learning, bukan CNN dari nol).
3. Threshold 75% berbasis keyakinan, bukan detektor bukan-bunga; input non-bunga
   (mis. screenshot merah → `roses`) tetap lolos jika skornya tinggi.

---

## 7. Kesimpulan dan Saran

[TODO — tulis setelah angka final ada: apakah optimasi menaikkan performa, berapa,
bukti aplikasinya, dan temuan OOD.]
Saran: tambah kelas unknown/OOD + data, coba augmentasi lebih kuat atau L2/label smoothing,
dan pisahkan test set 70/15/15 sejak awal.

---

## Lampiran
- A. Tabel tuning lengkap (screenshot `tabel` + `Konfigurasi terbaik`).
- B. Kurva training (`kurva_training.png`) + confusion matrix final dan baseline.
- C. Screenshot aplikasi (prediksi benar + Unidentified).
- D. Link demo video (maks. 5 menit): [TODO].
