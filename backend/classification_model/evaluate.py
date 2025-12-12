# pyright: reportMissingImports=false, typeCheckingMode=off
# pyright: ignore
# type: ignore
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = os.path.join(os.getcwd(), "data_prepared", "val")
MODEL_PATH = os.path.join(os.getcwd(), "model_best.keras")
MODEL_PATH_FALLBACK = os.path.join(os.getcwd(), "model_best.h5")
ALT_MODEL_PATH = os.path.join(os.getcwd(), "model_finetuned.keras")
ALT_MODEL_PATH_FALLBACK = os.path.join(os.getcwd(), "model_finetuned.h5")
CLASS_JSON = os.path.join(os.getcwd(), "class_indices.json")
IMG_SIZE = (224, 224)

def load_mapping():
    if not os.path.exists(CLASS_JSON):
        return None
    with open(CLASS_JSON, 'r') as f:
        m = json.load(f)
    inv = {int(v): k for k, v in m.items()}
    return inv

def build_dataset():
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        label_mode='int',
        image_size=IMG_SIZE,
        batch_size=32,
        shuffle=False,
        color_mode='rgb'
    )
    return ds

def evaluate():
    # prefer a non-empty model file; fall back to an alternate model if needed
    chosen = None
    # prefer non-empty .keras files, fallback to .h5 variants
    def good(p):
        return os.path.exists(p) and os.path.getsize(p) > 100
    if good(MODEL_PATH):
        chosen = MODEL_PATH
    elif good(MODEL_PATH_FALLBACK):
        chosen = MODEL_PATH_FALLBACK
    elif good(ALT_MODEL_PATH):
        chosen = ALT_MODEL_PATH
    elif good(ALT_MODEL_PATH_FALLBACK):
        chosen = ALT_MODEL_PATH_FALLBACK
    else:
        raise FileNotFoundError(
            "No valid model file found. Checked: \n"
            f"  {MODEL_PATH} ({os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH)} bytes)\n"
            f"  {MODEL_PATH_FALLBACK} ({os.path.exists(MODEL_PATH_FALLBACK) and os.path.getsize(MODEL_PATH_FALLBACK)} bytes)\n"
            f"  {ALT_MODEL_PATH} ({os.path.exists(ALT_MODEL_PATH) and os.path.getsize(ALT_MODEL_PATH)} bytes)\n"
            f"  {ALT_MODEL_PATH_FALLBACK} ({os.path.exists(ALT_MODEL_PATH_FALLBACK) and os.path.getsize(ALT_MODEL_PATH_FALLBACK)} bytes)\n"
        )
    try:
        model = tf.keras.models.load_model(chosen)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{chosen}': {e}")
    ds = build_dataset()
    inv_map = load_mapping()

    y_true = []
    y_pred = []
    probs = []
    for batch_imgs, batch_labels in ds:
        preds = model.predict(batch_imgs)
        pred_idx = np.argmax(preds, axis=1)
        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(pred_idx.tolist())
        probs.extend(preds.tolist())

    print('Total samples evaluated:', len(y_true))
    # map to names if mapping available
    if inv_map:
        target_names = [inv_map[i] for i in sorted(inv_map.keys())]
    else:
        target_names = [str(i) for i in sorted(set(y_true))]

    print('\nClassification report:')
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    print('\nConfusion matrix (rows=true, cols=pred):')
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

if __name__ == '__main__':
    evaluate()
