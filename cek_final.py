import os, json
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import tensorflow as tf

BASE = Path('C:/Users/Josua/Downloads/Kerja_Kelompok_Artificial Intelligence_CNN_Week 8')
custom = {"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input}
model = tf.keras.models.load_model(str(BASE / "model_cnn.keras"), custom_objects=custom, safe_mode=False)
kelas = json.load(open(BASE / "kelas.json", encoding="utf-8"))
benar, latih, ood_ok, ood = 0, 0, 0, 0
for fn in sorted(os.listdir(BASE / "Contoh_Sample_Gambar_Bunga")):
    if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img = Image.open(BASE / "Contoh_Sample_Gambar_Bunga" / fn)
    img = ImageOps.exif_transpose(img).convert("RGB").resize((224, 224))
    prob = model.predict(np.expand_dims(np.array(img), axis=0).astype("float32"), verbose=0)[0]
    idx, conf = int(np.argmax(prob)), float(max(prob))
    if fn.startswith("ood_"):
        ood += 1
        ok = conf < 0.75
        ood_ok += ok
        print(f"{fn} -> {kelas[idx]} {conf*100:.2f}% [{'Unidentified-OK' if ok else 'LOLOS'}]")
    else:
        latih += 1
        ok = kelas[idx].lower().startswith(fn.split("_")[0].lower()[:4])
        benar += ok
        print(f"{fn} -> {kelas[idx]} {conf*100:.2f}% [{'OK' if ok else 'SALAH'}]")
print(f"SKOR: {benar}/{latih} | OOD: {ood_ok}/{ood}")