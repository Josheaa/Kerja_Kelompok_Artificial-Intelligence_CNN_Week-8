"""Klasifikasi Jenis Bunga - Streamlit App (Tugas Kelompok 2 Week 8).

Cara jalan: python -m streamlit run app.py
File yang wajib ada di folder ini: model_cnn.keras + kelas.json (dari Google Colab).
"""

# BAGIAN 1: import library yang dipakai
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# BAGIAN 2: lokasi file dan ukuran gambar (sama seperti saat training)
BASE_DIR = Path(__file__).parent          # folder tempat app.py berada
MODEL_PATH = BASE_DIR / "model_cnn.keras" # model hasil training Colab
KELAS_PATH = BASE_DIR / "kelas.json"      # daftar nama bunga
CONTOH_DIR = BASE_DIR / "Contoh_Sample_Gambar_Bunga"  # gambar contoh
IMG_SIZE = (224, 224)                     # MobileNetV2 butuh 224x224
BATAS_UNIDENTIFIED = 0.75                 # confidence di bawah ini = tidak dikenali


# BAGIAN 3: load model + daftar kelas (sekali saja, lalu disimpan di cache)
@st.cache_resource
def load_model_dan_kelas():
    if not MODEL_PATH.exists() or not KELAS_PATH.exists():
        return None, None
    # custom_objects wajib karena model memakai Lambda(preprocess_input)
    custom = {"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input}
    model = tf.keras.models.load_model(str(MODEL_PATH), custom_objects=custom, safe_mode=False)
    kelas = json.loads(KELAS_PATH.read_text(encoding="utf-8"))
    return model, kelas


# BAGIAN 4: siapkan gambar sebelum diprediksi (resize saja, tanpa preprocess ganda)
def siapkan_gambar(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.array(img), axis=0).astype("float32")


# BAGIAN 5: judul + sembunyikan tombol Deploy dan ikon loading bawaan Streamlit
st.set_page_config(page_title="Kerja Kelompok AI - Klasifikasi Jenis Bunga", page_icon="🌸")
st.markdown(
    "<style>"
    ".stDeployButton, .stAppDeployButton, [data-testid='stDeployButton'], "
    "[data-testid='stAppDeployButton'], button[title='Deploy'], [aria-label='Deploy'], "
    "[data-testid='stStatusWidget']"
    "{display:none !important;}"
    "</style>",
    unsafe_allow_html=True,
)
st.title("🌸 Kerja Kelompok AI - Klasifikasi Jenis Bunga")

model, kelas = load_model_dan_kelas()

# BAGIAN 6: jika model belum ada, tampilkan petunjuk lalu berhenti
if model is None:
    st.error(
        "Model belum ditemukan.\n\n"
        "1. Buka Google Colab, jalankan notebook sampai cell `model.save(\"model_cnn.keras\")`.\n"
        "2. Download `model_cnn.keras` + `kelas.json` dari Colab.\n"
        "3. Taruh kedua file itu di folder yang sama dengan `app.py` ini, lalu refresh."
    )
    st.info(f"Folder app: {BASE_DIR}\nIsi folder: {', '.join(sorted(os.listdir(BASE_DIR)))}")
    st.stop()

# BAGIAN 7: kotak hijau kesimpulan
st.success(
    "Kesimpulan  \n"
    "Aplikasi pengenalan bunga ini dibuat menggunakan CNN MobileNetV2 dengan metode transfer learning. "
    "Antarmuka web aplikasi dibangun dengan Streamlit. "
    "Berikut beberapa hal utama dari aplikasi yang dibuat:  \n"
    "1. Model dapat mengenali lima jenis bunga, yaitu \"daisy\", \"dandelion\", \"roses\", \"sunflowers\", dan \"tulips\".  \n"
    "2. Pengguna cukup mengunggah gambar bunga. Gambar akan otomatis diubah ukurannya menjadi 224x224 piksel "
    "sebelum diproses oleh model.  \n"
    "3. Setelah gambar diproses, aplikasi menampilkan hasil prediksi jenis bunga beserta nilai confidence "
    "untuk setiap kelas.  \n"
    "4. Jika nilai confidence di bawah 75%, hasilnya ditampilkan sebagai Unidentified. Hal ini menunjukkan "
    "bahwa model belum cukup yakin dengan prediksinya, misalnya ketika gambar bunga tidak termasuk dalam dataset.  \n"
    "5. Aplikasi menggunakan model yang disimpan dalam file model_cnn.keras. Untuk pengujian, beberapa gambar "
    "juga tersedia di folder Contoh_Sample_Gambar_Bunga."
)

# BAGIAN 8: pilih sumber gambar (contoh disembunyikan jika foldernya tidak ada, mis. di hosting)
EKSTENSI_GAMBAR = (".jpg", ".jpeg", ".png")  # hanya file gambar yang ditampilkan
daftar_contoh = []
if CONTOH_DIR.is_dir():
    for nama_file in sorted(os.listdir(CONTOH_DIR)):
        if nama_file.lower().endswith(EKSTENSI_GAMBAR):
            daftar_contoh.append(nama_file)
pilihan_sumber = ["Upload"]
if daftar_contoh:
    pilihan_sumber.append("Contoh_Sample_Gambar_Bunga")
sumber = st.radio("Sumber gambar:", pilihan_sumber, horizontal=True)
img = None
if sumber == "Upload":
    f = st.file_uploader("Upload gambar bunga", type=["jpg", "jpeg", "png"])
    if f is not None:
        img = ImageOps.exif_transpose(Image.open(f))  # betulkan rotasi foto HP
else:
    contoh = st.selectbox(
        "Pilih contoh (folder Contoh_Sample_Gambar_Bunga):",
        daftar_contoh,
    )
    img = ImageOps.exif_transpose(Image.open(CONTOH_DIR / contoh))

# BAGIAN 9: prediksi dan tampilkan hasil
if img is None:
    st.info("Upload gambar dulu untuk melihat hasil prediksi.")
else:
    st.image(img, use_container_width=True)  # use_container_width: warning depresiasi saja, tetap jalan
    with st.spinner("🌸 Memprediksi bunga..."):
        prob = model.predict(siapkan_gambar(img), verbose=0)[0]
    idx = int(np.argmax(prob))   # kelas dengan confidence terbesar
    conf = float(prob[idx])

    if conf < BATAS_UNIDENTIFIED:
        conf_txt = f"{conf * 100:.2f}".replace(".", ",") + "%"
        st.warning(
            f"**Unidentified** ({conf_txt}). Tingkat confidence masih di bawah batas 75%, "
            f"sehingga model belum dapat mengenali bunga ini dengan yakin. "
            f"Kemungkinan bunga tidak termasuk dalam lima jenis yang dikenali."
        )
    nama_hasil = kelas[idx] if conf >= BATAS_UNIDENTIFIED else f"{kelas[idx]} (?)"
    st.metric("Hasil Prediksi", nama_hasil, f"{conf * 100:.2f}% yakin")

    st.subheader("Confidence tiap kelas")
    for nama, nilai in sorted(zip(kelas, prob), key=lambda x: -x[1]):
        kolom_nama, kolom_bar = st.columns([1, 4])
        kolom_nama.write(nama)
        nilai_aman = min(1.0, max(0.0, float(nilai)))  # progress wajib 0-1
        kolom_bar.progress(nilai_aman, text=f"{float(nilai) * 100:.2f}%")

