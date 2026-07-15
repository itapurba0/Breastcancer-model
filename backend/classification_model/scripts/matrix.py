# backend/classification_model/plot_metrics.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(__file__)

# 1) Confusion matrix: try to load cm.npy, otherwise use the provided matrix
cm_path = os.path.join(ROOT, "cm.npy")
if os.path.exists(cm_path):
    cm = np.load(cm_path)
else:
    # fallback to the matrix you provided
    cm = np.array([[87, 0, 0],
                   [3, 39, 0],
                   [1, 0, 25]])

class_map_path = os.path.join(ROOT, "class_indices.json")
if os.path.exists(class_map_path):
    try:
        m = json.load(open(class_map_path, "r"))
        # handle both formats: index->name or name->index
        if all(k.isdigit() for k in m.keys()):
            labels = [m[str(i)] for i in range(len(m))]
        else:
            # m is name->idx
            inv = {int(v): k for k, v in m.items()}
            labels = [inv[i] for i in sorted(inv.keys())]
    except Exception:
        labels = [str(i) for i in range(cm.shape[0])]
else:
    labels = [str(i) for i in range(cm.shape[0])]

# Plot confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (rows=true, cols=pred)")
out_cm = os.path.join(ROOT, "confusion_matrix3.png")
plt.tight_layout()
plt.savefig(out_cm, dpi=150)
plt.close()
print("Saved confusion matrix to", out_cm)

