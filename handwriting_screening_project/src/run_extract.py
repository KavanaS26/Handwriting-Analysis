"""Run standardization + line segmentation + feature extraction over both datasets."""
import os, sys, glob, time
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import standardize
from lines import segment_lines
from features import image_features, FEATURE_NAMES

from config import DATA, OUT
os.makedirs(f"{OUT}/std", exist_ok=True)
JOBS = [
    ("dyslexia", "dyslexic/no", 0), ("dyslexia", "dyslexic/yes", 1),
    ("dysgraphia", "DATASET DYSGRAPHIA HANDWRITING/Low Potential Dysgraphia", 0),
    ("dysgraphia", "DATASET DYSGRAPHIA HANDWRITING/Potential Dysgraphia", 1),
]
rows, t0 = [], time.time()
for task, folder, label in JOBS:
    files = sorted(glob.glob(f"{DATA}/{folder}/*"))
    for p in files:
        try:
            bw = standardize(p)
            lines = segment_lines(bw)
            feats = image_features(lines) if lines else None
            if feats is None:
                rows.append({"task": task, "label": label, "file": os.path.relpath(p, DATA), "ok": False}); continue
            tag = f"{task}_{label}_{os.path.basename(p).split('.')[0].replace(' ','_').replace('(','').replace(')','')}"
            cv2.imwrite(f"{OUT}/std/{tag}.png", 255 - bw)
            rows.append({"task": task, "label": label, "file": os.path.relpath(p, DATA), "ok": True, "std_w": bw.shape[1],
                         "std_h": bw.shape[0], "n_lines_seg": len(lines), **feats})
        except Exception as e:
            rows.append({"task": task, "label": label, "file": os.path.relpath(p, DATA), "ok": False, "err": str(e)})
    print(task, folder.split("/")[-1], len(files), "done", f"{time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/features.csv", index=False)
print(df.groupby(["task", "label"])["ok"].agg(["sum", "count"]))
