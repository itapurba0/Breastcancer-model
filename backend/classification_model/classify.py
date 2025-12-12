# pyright: reportMissingImports=false, typeCheckingMode=off
# pyright: ignore
# type: ignore
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
# ...existing code...

MODEL_PATH = "model_best.keras" 
MODEL_PATH_FALLBACK = "model_best.h5"
CLASS_JSON = "class_indices.json"
IMG_SIZE = (224, 224)

def load_mapping():
    if not os.path.exists(CLASS_JSON):
        return None
    with open(CLASS_JSON, "r") as f:
        m = json.load(f)
    # keys in json are class_name -> index (string), invert to index -> class_name (int keys)
    inv = {int(v): k for k, v in m.items()}
    return inv

def preprocess(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    # load image, convert grayscale to RGB automatically by keras loader
    img = image.load_img(img_path, target_size=IMG_SIZE, color_mode="rgb")
    # NOTE: training used images from image_dataset_from_directory without
    # explicit rescaling. To keep preprocessing identical to training we do
    # NOT divide by 255 here. Keep values in [0,255] as float32.
    arr = image.img_to_array(img).astype('float32')
    arr = np.expand_dims(arr, 0)
    return arr

def predict(img_path):
    # prefer .keras model, fall back to .h5
    model_path = None
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100:
        model_path = MODEL_PATH
    elif os.path.exists(MODEL_PATH_FALLBACK) and os.path.getsize(MODEL_PATH_FALLBACK) > 100:
        model_path = MODEL_PATH_FALLBACK
    else:
        raise FileNotFoundError(f"No valid model file found. Checked: {MODEL_PATH} and {MODEL_PATH_FALLBACK}")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_path}': {e}")
    inv_map = load_mapping()
    x = preprocess(img_path)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    cls = inv_map.get(idx, str(idx)) if inv_map else str(idx)
    return cls, float(preds[idx]), preds

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classify.py /path/to/image.png [--model path] [--labels path]")
        sys.exit(1)
    img = sys.argv[1]
    # optional args (simple parsing)
    for i, a in enumerate(sys.argv[2:], start=2):
        if a == '--model' and i + 1 <= len(sys.argv) - 1:
            MODEL_PATH = sys.argv[i+1]
        if a == '--labels' and i + 1 <= len(sys.argv) - 1:
            CLASS_JSON = sys.argv[i+1]
    try:
        cls, prob, raw = predict(img)
        print("Predicted:", cls, "prob:", prob)
        print("Raw scores:", raw)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
# ...existing code...