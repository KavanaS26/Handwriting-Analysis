"""Statistical tests for the paper (run AFTER train.py).

1. Nadeau-Bengio corrected resampled t-test on the paired fold-level AUC difference
   (final handwriting model minus metadata-only baseline). Plain t-tests are invalid on repeated-CV
   folds because the training sets overlap; the correction inflates the variance by (1/k + n_test/n_train).
2. Percentile-bootstrap 95% CI of the out-of-fold AUC (resampling images; NOT children - no writer IDs exist).
Uses exactly the same splits as train.py (RepeatedStratifiedKFold 5x5, seed 42).
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from config import DATA, OUT
from features import FEATURE_NAMES

df = pd.read_csv(f"{OUT}/features_clean.csv")
def meta(files):
    m = []
    for p in files:
        p = os.path.join(DATA, p); im = Image.open(p); a = np.array(im.convert("L"))
        m.append([im.size[0], im.size[1], im.size[0] / im.size[1], a.mean(), a.std(), len(np.unique(a)),
                  {"L": 1, "P": 1, "RGB": 3, "RGBA": 4}.get(im.mode, 0), os.path.getsize(p) / 1024])
    return np.array(m)
rf_final = lambda: make_pipeline(SimpleImputer(strategy="median"),
    CalibratedClassifierCV(RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=0), method="sigmoid", cv=5))
rf_meta = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=0)

_p = f"{OUT}/stat_tests.json"
res = json.load(open(_p)) if os.path.exists(_p) else {}
for task in (sys.argv[1:] or ["dysgraphia", "dyslexia"]):
    d = df[df.task == task].reset_index(drop=True); X = d[FEATURE_NAMES].values; Xm = meta(d.file); y = d.label.values
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    a_f, a_m, ntr, nte = [], [], [], []
    for tr, te in cv.split(X, y):
        a_f.append(roc_auc_score(y[te], rf_final().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]))
        a_m.append(roc_auc_score(y[te], rf_meta().fit(Xm[tr], y[tr]).predict_proba(Xm[te])[:, 1]))
        ntr.append(len(tr)); nte.append(len(te))
    diff = np.array(a_f) - np.array(a_m); k = len(diff)
    var = diff.var(ddof=1); corr = 1 / k + np.mean(nte) / np.mean(ntr)
    t = diff.mean() / np.sqrt(corr * var); p = 2 * stats.t.sf(abs(t), df=k - 1)
    half = stats.t.ppf(0.975, k - 1) * np.sqrt(corr * var)
    oof = pd.read_csv(f"{OUT}/oof_{task}.csv"); rng = np.random.RandomState(0); bs = []
    for _ in range(2000):
        i = rng.randint(0, len(oof), len(oof))
        if len(np.unique(oof.label.values[i])) == 2: bs.append(roc_auc_score(oof.label.values[i], oof.oof_final.values[i]))
    bm = []
    om = oof.oof_metadata.values
    for _ in range(2000):
        i = rng.randint(0, len(oof), len(oof))
        if len(np.unique(oof.label.values[i])) == 2: bm.append(roc_auc_score(oof.label.values[i], om[i]))
    res[task] = dict(n_folds=k, mean_diff=float(diff.mean()), sd_diff=float(diff.std(ddof=1)),
        nadeau_bengio_t=float(t), df=k - 1, p_value=float(p),
        ci95_corrected=[float(diff.mean() - half), float(diff.mean() + half)],
        oof_auc_model=float(roc_auc_score(oof.label, oof.oof_final)), oof_auc_model_boot95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        oof_auc_metadata=float(roc_auc_score(oof.label, oof.oof_metadata)), oof_auc_metadata_boot95=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))])
    print(task, json.dumps(res[task], indent=1), flush=True)
    json.dump(res, open(_p, "w"), indent=1)
