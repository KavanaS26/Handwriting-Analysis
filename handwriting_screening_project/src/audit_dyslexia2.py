import os, re, warnings, collections
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import sys; sys.path.insert(0, os.path.dirname(__file__))
from config import DATA, OUT
ROOT = os.path.join(DATA, "pakistan_dyslexia")
def session(name):
    n = os.path.splitext(name)[0]
    if re.match(r"\d{4}-\d{2}-\d{2} \d", n): return "date_" + n[:10]
    m = re.match(r"IMG_(\d{8})_(\d{6})", n)
    if m: return f"IMGdt_{m.group(1)}_{m.group(2)}"          # crops of one photo share this
    if n.startswith("IMG_"): return "IMG_camroll"
    if n.startswith("WhatsApp"): return "whatsapp"
    m = re.match(r"(\d+) \(", n)
    return f"batch_{m.group(1)}" if m else "other"
rows = []
for cls, lab in [("No", 0), ("Yes", 1)]:
    for f in sorted(os.listdir(f"{ROOT}/{cls}")):
        p = f"{ROOT}/{cls}/{f}"; rel = os.path.relpath(p, DATA); im = Image.open(p).convert("RGB"); a = np.array(im); g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        rows.append(dict(file=rel, label=lab, session=session(f), w=im.size[0], h=im.size[1], aspect=im.size[0]/im.size[1],
            bright=g.mean(), contrast=g.std(), R=a[...,0].mean(), G=a[...,1].mean(), B=a[...,2].mean(),
            sharp=cv2.Laplacian(g, cv2.CV_64F).var(), kb=os.path.getsize(p)/1024, ext=int(p.endswith(".jpg"))))
df = pd.DataFrame(rows); df["warm"] = df.R - df.B
df.to_csv(f"{OUT}/dyslexia2_meta.csv", index=False)
X = df[["w","h","aspect","bright","contrast","R","G","B","warm","sharp","kb","ext"]].values; y = df.label.values
print("=== class-pure sessions? (share of images whose capture-session key appears in only ONE class) ===")
ct = pd.crosstab(df.session, df.label); pure = ct[(ct[0] == 0) | (ct[1] == 0)]
print(f"{len(ct)} session keys; {len(pure)} are class-pure, covering {pure.values.sum()} of {len(df)} images ({pure.values.sum()/len(df):.0%})")
print(ct.sort_values(by=[0,1], ascending=False).head(12).to_string())
print("\n=== single-field AUC (0.5 = no signal) ===")
for c in ["w","h","aspect","bright","contrast","warm","sharp","kb","ext"]:
    a = roc_auc_score(y, df[c]); print(f"  {c:9s} {max(a,1-a):.3f}")
rf = RandomForestClassifier(n_estimators=300, random_state=0)
print("\n=== metadata-only classifier (no handwriting content) ===")
for nm, cv, kw in [("random split ", StratifiedKFold(5, shuffle=True, random_state=0), {}),
                   ("session-grouped split", GroupKFold(5), {"groups": df.session})]:
    s = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc", **kw); print(f"  {nm:22s} AUC {s.mean():.3f} ± {s.std():.3f}")
