import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
import sys, json; sys.path.insert(0, os.path.dirname(__file__))
from config import DATA, OUT
meta = pd.read_csv(f"{OUT}/dyslexia2_meta.csv")

def std_char(path):
    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (256, int(256 * g.shape[0] / g.shape[1]))) if g.shape[1] != 256 else g
    bg = cv2.GaussianBlur(g, (0, 0), 25)                        # flatten uneven lighting / paper tone
    flat = cv2.normalize(cv2.divide(g.astype(np.float32), bg.astype(np.float32) + 1, scale=200), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    flat = cv2.createCLAHE(2.0, (4, 4)).apply(flat)
    _, bw = cv2.threshold(cv2.GaussianBlur(flat, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, 4] >= 25
    bw = (keep[lab] * 255).astype(np.uint8)
    ys, xs = np.where(bw > 0)
    if len(ys) < 30: return None
    return bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

K = np.ones((3, 3)); K[1, 1] = 0
def feats(c):
    h, w = c.shape; ink = c > 0
    sk = skeletonize(ink); nb = convolve(sk.astype(int), K, mode="constant") * sk
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3); th = 2 * dt[sk]; th = th[th > 0]
    cnts, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = max(cnts, key=cv2.contourArea); hull = cv2.convexHull(big)
    ncomp = cv2.connectedComponents(c, connectivity=8)[0] - 1
    return dict(bbox_aspect=w / h, ink_ratio=ink.mean(), stroke_rel=th.mean() / max(h, w), stroke_cv=th.std() / th.mean(),
                skel_len=sk.sum() / np.hypot(h, w), endpoints=(nb == 1).sum(), junctions=(nb >= 3).sum(),
                solidity=cv2.contourArea(big) / (cv2.contourArea(hull) + 1e-6), n_comp=ncomp,
                skel_curv=sk.sum() / (np.count_nonzero(ink.any(0)) + np.count_nonzero(ink.any(1))))

rows, thumbs, keepi = [], [], []
for i, r in meta.iterrows():
    c = std_char(os.path.join(DATA, r.file))
    if c is None: continue
    rows.append(feats(c)); keepi.append(i)
    thumbs.append(cv2.resize(c, (24, 24), interpolation=cv2.INTER_AREA).ravel() / 255.0)
F = pd.DataFrame(rows); m = meta.loc[keepi].reset_index(drop=True)
print(f"usable characters after standardization: {len(F)}/{len(meta)}")
y = m.label.values; g = m.session.values; T = np.array(thumbs)
rf = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, random_state=0))
lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
rnd = StratifiedKFold(5, shuffle=True, random_state=0); grp = GroupKFold(5)
meta_cols = ["w","h","aspect","bright","contrast","R","G","B","warm","sharp","kb","ext"]
def auc(model, X, cv, grouped):
    return cross_val_score(model, X, y, cv=cv, scoring="roc_auc", **({"groups": g} if grouped else {}))
RES = {}
print("\nAUC (mean ± std over folds); 0.50 = no signal")
print(f"{'input':46s} {'random split':>14s} {'session-grouped':>17s}")
for nm, X, mdl in [("raw photo metadata (acquisition only)", m[meta_cols].values, rf),
                   ("standardized SHAPE features (10), RF", F.values, rf),
                   ("standardized 24x24 binary glyph, LogReg", T, lr)]:
    a, b = auc(mdl, X, rnd, False), auc(mdl, X, grp, True)
    print(f"{nm:46s} {a.mean():.3f}±{a.std():.3f}   {b.mean():.3f}±{b.std():.3f}")
    RES[nm] = dict(random=[float(a.mean()), float(a.std())], grouped=[float(b.mean()), float(b.std())])
F.assign(label=y, session=g, file=m.file).to_csv(f"{OUT}/dyslexia2_shape_features.csv", index=False)
# which shape features move?
from sklearn.metrics import roc_auc_score
print("\nsingle shape feature AUC (direction-free):", {c: round(max(roc_auc_score(y, F[c].fillna(F[c].median())), 1 - roc_auc_score(y, F[c].fillna(F[c].median()))), 2) for c in F.columns})

json.dump(RES, open(f"{OUT}/dyslexia2_results.json", "w"), indent=1)
