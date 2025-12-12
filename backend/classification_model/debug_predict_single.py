
# pyright: reportMissingImports=false, typeCheckingMode=off
# pyright: ignore
# type: ignore
import os
import json
import sys
import numpy as np
try:
    import tensorflow as tf
except Exception:
    tf = None
from PIL import Image

MODEL_CANDIDATES = [
    "model_best.keras",
    "model_finetuned.keras",
    "model_best.h5",
    "model_finetuned.h5",
]

IMG_SIZE = (224, 224)


def find_model():
    here = os.getcwd()
    for name in MODEL_CANDIDATES:
        p = os.path.join(here, name)
        if os.path.exists(p) and os.path.getsize(p) > 100:
            return p
    return None


def load_class_map():
    p = os.path.join(os.getcwd(), "class_indices.json")
    if not os.path.exists(p):
        return None
    with open(p, "r") as fh:
        d = json.load(fh)
    return {int(v): k for k, v in d.items()}


def preprocess(path):
    im = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(im, dtype=np.float32)
    return np.expand_dims(arr, 0)


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_predict_single.py /path/to/image.png")
        sys.exit(1)
    img = sys.argv[1]
    if not os.path.exists(img):
        print("Image not found:", img)
        sys.exit(1)

    model_path = find_model()
    print("Model candidate found:", model_path)
    class_map = load_class_map()
    print("Class map (idx->name):", class_map)

    if tf is None:
        print("TensorFlow not available in this environment. Activate your TF venv.")
        sys.exit(1)

    model = tf.keras.models.load_model(model_path)
    x = preprocess(img)
    preds = model.predict(x)[0]
    for i, p in enumerate(preds):
        print(f"  idx={i} name={class_map.get(i, i)} prob={p:.6f}")
    pred_idx = int(np.argmax(preds))
    print("Predicted idx:", pred_idx, "->", class_map.get(pred_idx, pred_idx))


if __name__ == '__main__':
    main()
