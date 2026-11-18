import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
sys.path.insert(0, os.path.dirname(__file__))
from config import DATA, OUT
df = pd.read_csv(f"{OUT}/features_clean.csv")
G = {
 "size/shape":  ["letter_height_cv", "size_ratio", "comps_per_line", "ink_density"],
 "baseline":    ["baseline_dev", "baseline_slope_abs", "line_height_cv", "n_lines"],
 "stroke":      ["stroke_mean", "stroke_cv"],
 "slant":       ["slant_var", "slant_mean_abs"],
 "spacing":     ["word_gap_cv", "word_gap_mean", "letter_gap_cv", "touching_ratio"],
}
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
mk = lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=200, random_state=0, class_weight="balanced"))
auc = lambda X, y: cross_val_score(mk(), X, y, cv=rcv, scoring="roc_auc").mean()
def meta(files):
    m = []
    for p in files:
        p = os.path.join(DATA, p)
        im = Image.open(p); a = np.array(im.convert("L"))
        m.append([im.size[0], im.size[1], im.size[0]/im.size[1], a.mean(), len(np.unique(a)),
                  {"L":1,"P":1,"RGB":3,"RGBA":4}.get(im.mode,0), os.path.getsize(p)/1024])
    return np.array(m)
allf = sum(G.values(), [])
for task in ["dyslexia", "dysgraphia"]:
    d = df[df.task == task]; y = d.label.values
    print(f"\n===== {task} (n={len(d)}) =====")
    print(f"all 16 features                               AUC {auc(d[allf].values, y):.3f}")
    print(f"RAW FILE METADATA ONLY (no handwriting used)  AUC {auc(meta(d.file), y):.3f}   <- shortcut check")
    print("one group ALONE  |  everything EXCEPT that group")
    for g, cols in G.items():
        rest = [c for c in allf if c not in cols]
        print(f"  {g:11s} alone {auc(d[cols].values, y):.3f}   |   without {auc(d[rest].values, y):.3f}")
