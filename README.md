# Klasifikasi Jenis Bunga dengan CNN (Tugas Kelompok 2 - Week 8)

Aplikasi web (Streamlit) untuk memprediksi jenis bunga dari gambar.
Model: MobileNetV2 (transfer learning) + BatchNormalization + Dropout, hasil hyperparameter tuning (8 konfigurasi) + fine-tuning.

Kelas: "daisy", "dandelion", "roses", "sunflowers", "tulips"

## Struktur folder
- `app.py`: aplikasi Streamlit (yang dijalankan)
- `Praktikum Kelompok_Artificial Intelligence_LEC_Week 8.ipynb`: notebook training lengkap (data + baseline + tuning + fine-tuning + evaluasi)
- `model_cnn.keras`: HASIL training dari Colab (wajib ada sejajar `app.py`)
- `kelas.json`: daftar kelas
- `Contoh_Sample_Gambar_Bunga/`: gambar untuk testing/demo (`ood_*` = uji di luar dataset)
- `kurva_training.png`, `confusion_matrix.png`, `confusion_matrix_baseline.png`: grafik untuk laporan
- `hasil_uji_gambar_baru.csv`: hasil uji gambar baru dari Colab (dibuat saat cell uji dijalankan)

## A. Training di Google Colab (T4 GPU)
1. Upload `Praktikum Kelompok_Artificial Intelligence_LEC_Week 8.ipynb` ke Colab, `Runtime > Run all`.
2. Download dari Colab: `model_cnn.keras`, `kelas.json`, `kurva_training.png`, `confusion_matrix.png`, `confusion_matrix_baseline.png`, `hasil_uji_gambar_baru.csv`.
3. Pindahkan `model_cnn.keras` + `kelas.json` ke folder ini sejajar `app.py`.

## B. Jalan lokal (Windows)
1. Install Python 3.11 atau 3.12
2. pip install -r requirements.txt
3. python -m streamlit run app.py
4. Buka http://localhost:8501
5. Pilih sumber `Upload` atau `Contoh_Sample_Gambar_Bunga`. Untuk laporan: uji tiap kelas 1x + file `ood_*` sebagai kasus OOD.

## Catatan
- Preprocessing (resize 224x224 + `mobilenet_v2.preprocess_input`) ada di dalam model (Lambda layer),
  app hanya resize tanpa preprocess ganda.
- Confidence di bawah 75% ditampilkan sebagai Unidentified.
- Kalau `model_cnn.keras` belum ada, app menampilkan petunjuk, bukan crash.
