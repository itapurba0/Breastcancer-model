import os
import shutil
import random


SRC_NORMAL_DIR = os.path.join(os.getcwd(), "data", "normal")
SRC_BENIGN_DIR = os.path.join(os.getcwd(), "data", "benign")
SRC_MALIGNANT_DIR = os.path.join(os.getcwd(), "data", "malignant")


OUT_ROOT = os.path.join(os.getcwd(), "data_prepared")
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}
CLASSES = {"normal": SRC_NORMAL_DIR, "benign": SRC_BENIGN_DIR, "malignant": SRC_MALIGNANT_DIR}
RANDOM_SEED = 42
EXTS = {".png", ".jpg", ".jpeg"}

def gather_files_recursive(src):
    items = []
    if not os.path.isdir(src):
        return items
    for root, _, files in os.walk(src):
        for fn in files:
            if os.path.splitext(fn.lower())[1] in EXTS:
                items.append(os.path.join(root, fn))
    return items

def make_split():
    random.seed(RANDOM_SEED)
    if os.path.exists(OUT_ROOT):
        print("Warning: data/ already exists — files may be overwritten or duplicated.")
    total_before = 0
    summary = {}
    for cls, src in CLASSES.items():
        if not os.path.isdir(src):
            print(f"Warning: source folder for class '{cls}' not found: {src}")
            summary[cls] = 0
            continue
        files = gather_files_recursive(src)
        n = len(files)
        total_before += n
        summary[cls] = n
        if n == 0:
            print(f"Warning: no images found for class '{cls}' in {src}")
        random.shuffle(files)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])
        train = files[:n_train]
        val = files[n_train:n_train + n_val]
        test = files[n_train + n_val:]
        subsets = {"train": train, "val": val, "test": test}
        for subset, filelist in subsets.items():
            out_dir = os.path.join(OUT_ROOT, subset, cls)
            os.makedirs(out_dir, exist_ok=True)
            for src_path in filelist:
                dst = os.path.join(out_dir, os.path.basename(src_path))
                if not os.path.exists(dst):
                    shutil.copy2(src_path, dst)
    print("Source counts per class:", summary)
    print("Total images found:", total_before)
    
    total_after = 0
    res_summary = {}
    for subset in ("train", "val", "test"):
        for cls in CLASSES.keys():
            p = os.path.join(OUT_ROOT, subset, cls)
            cnt = 0
            if os.path.isdir(p):
                cnt = sum(1 for _ in os.listdir(p) if os.path.splitext(_.lower())[1] in EXTS)
            res_summary[f"{subset}/{cls}"] = cnt
            total_after += cnt
    print("Created files under data/ (counts):")
    for k, v in res_summary.items():
        print(f"  {k}: {v}")
    print("Total copied:", total_after)
    print("Done. Created folders under:", OUT_ROOT)

if __name__ == "__main__":
    make_split()