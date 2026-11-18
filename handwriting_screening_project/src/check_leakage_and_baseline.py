import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, cross_validate
from scipy.stats import mannwhitneyu
sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_NAMES

from config import DATA, OUT
df = pd.read_csv(f"{OUT}/features.csv"); df = df[df.ok].reset_index(drop=True)
df["source"] = (df.task == "dysgraphia").astype(int)
rf = lambda: RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")
cv = StratifiedKFold(5, shuffle=True, random_state=0)

# (a) RAW metadata: what a lazy model could use without looking at handwriting
meta = []
for p in df.file:
    p = os.path.join(DATA, p)
    im = Image.open(p); a = np.array(im.convert("L"))
    meta.append([im.size[0], im.size[1], im.size[0] / im.size[1], a.mean(), len(np.unique(a)),
                 {"L": 1, "P": 1, "RGB": 3, "RGBA": 4}.get(im.mode, 0), os.path.getsize(p) / 1024])
Xa = np.array(meta)
# (b) STANDARDIZED image pixels (downsampled canonical binary image)
def thumb(row):
    tag = f"{row.task}_{int(row.label)}_{os.path.basename(row.file).split('.')[0].replace(' ','_').replace('(','').replace(')','')}"
    g = cv2.imread(f"{OUT}/std/{tag}.png", 0)
    return cv2.resize(255 - g, (128, 32), interpolation=cv2.INTER_AREA).ravel() / 255.0
Xb = np.array([thumb(r) for r in df.itertuples()])
# (c) handcrafted features
Xc = df[FEATURE_NAMES].values
y = df.source.values
print("=== SOURCE-LEAKAGE CHECK: can a model tell dyslexia-set vs dysgraphia-set? (balanced acc; chance = 0.50) ===")
for name, X, mdl in [("(a) raw file metadata (size, aspect, brightness, mode...)", Xa, rf()),
                     ("(b) standardized pixels, RandomForest", Xb, rf()),
                     ("(b) standardized pixels, LogisticReg", Xb, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"))),
                     ("(c) 16 handcrafted features", Xc, make_pipeline(SimpleImputer(strategy="median"), rf()))]:
    s = cross_val_score(mdl, X, y, cv=cv, scoring="balanced_accuracy")
    print(f"{name:62s} {s.mean():.3f} ± {s.std():.3f}")

print("\n=== FEATURE SANITY: per-task baselines (5x5 repeated CV; class = positive label) ===")
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
for task in ["dyslexia", "dysgraphia"]:
    d = df[df.task == task]; X = d[FEATURE_NAMES].values; yy = d.label.values
    print(f"\n{task}: n={len(d)}  pos={int(yy.sum())}  neg={int((1-yy).sum())}")
    for nm, m in [("LogReg", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
                  ("RandomForest", make_pipeline(SimpleImputer(strategy="median"), rf()))]:
        r = cross_validate(m, X, yy, cv=rcv, scoring=["roc_auc", "balanced_accuracy", "f1"])
        print(f"  {nm:13s} AUC={r['test_roc_auc'].mean():.3f}±{r['test_roc_auc'].std():.3f}  balAcc={r['test_balanced_accuracy'].mean():.3f}  F1={r['test_f1'].mean():.3f}")
    # univariate effect: AUC-style effect from Mann-Whitney (0.5 = no difference)
    eff = []
    for f in FEATURE_NAMES:
        a, b = d[d.label == 1][f].dropna(), d[d.label == 0][f].dropna()
        if len(a) > 5 and len(b) > 5:
            u, pv = mannwhitneyu(a, b); eff.append((f, u / (len(a) * len(b)), pv, a.median(), b.median()))
    eff.sort(key=lambda t: -abs(t[1] - 0.5))
    print("  top features (P(pos>neg), p-value, median pos vs neg):")
    for f, e, pv, ma, mb in eff[:5]:
        print(f"    {f:20s} {e:.2f}  p={pv:.4f}  {ma:.3f} vs {mb:.3f}")
df.to_csv(f"{OUT}/features_clean.csv", index=False)
