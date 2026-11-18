"""Final system: two independent heads (dyslexia, dysgraphia) on shared handcrafted features.

Evaluation protocol (identical splits for all models, 5x5 repeated stratified CV):
  - final model      : calibrated Random Forest on 16 features
  - interpretable    : logistic regression on the same features
  - shortcut baseline: Random Forest on raw FILE METADATA only (no handwriting content)
  - negative control : Random Forest trained on label-shuffled data
A head is only credible if the final model clearly beats the shortcut baseline.
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, brier_score_loss, confusion_matrix
from sklearn.inspection import permutation_importance
sys.path.insert(0, os.path.dirname(__file__))
from config import DATA, OUT, MODELS
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
rf_plain = lambda: make_pipeline(SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=0))
lr = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
rf_meta = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=0)

def score(y, p, thr=0.5):
    pr = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
    return dict(auc=roc_auc_score(y, p), bal_acc=balanced_accuracy_score(y, pr), f1=f1_score(y, pr),
                sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1), brier=brier_score_loss(y, p))

summary, t0 = {}, time.time()
for task in ["dyslexia", "dysgraphia"]:
    d = df[df.task == task].reset_index(drop=True); X = d[FEATURE_NAMES].values; Xm = meta(d.file); y = d.label.values
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    names = ["final_RF_calibrated", "logreg", "metadata_only_RF", "label_shuffle_control"]
    res = {n: [] for n in names}; oof = np.zeros((len(y), 5)); cnt = np.zeros(len(y)); rng = np.random.RandomState(0)
    oof_meta = np.zeros(len(y)); oof_lr = np.zeros(len(y))
    for k, (tr, te) in enumerate(cv.split(X, y)):
        m1 = rf_final().fit(X[tr], y[tr]); p1 = m1.predict_proba(X[te])[:, 1]
        m2 = lr().fit(X[tr], y[tr]); p2 = m2.predict_proba(X[te])[:, 1]
        m3 = rf_meta().fit(Xm[tr], y[tr]); p3 = m3.predict_proba(Xm[te])[:, 1]
        m4 = rf_plain().fit(X[tr], rng.permutation(y[tr])); p4 = m4.predict_proba(X[te])[:, 1]
        for n, p in zip(names, [p1, p2, p3, p4]):
            s = score(y[te], p); s["fold"] = k; s["repeat"] = k // 5; res[n].append(s)
        oof[te, 0] += p1; oof_lr[te] += p2; oof_meta[te] += p3; cnt[te] += 1
    oof_final, oof_lr, oof_meta = oof[:, 0] / cnt, oof_lr / cnt, oof_meta / cnt
    tab = {n: pd.DataFrame(v) for n, v in res.items()}
    out = {n: {m: [float(t[m].mean()), float(t[m].std())] for m in ["auc", "bal_acc", "f1", "sens", "spec", "brier"]} for n, t in tab.items()}
    diff = tab["final_RF_calibrated"].auc.values - tab["metadata_only_RF"].auc.values
    out["paired_auc_diff_final_minus_metadata"] = dict(mean=float(diff.mean()), std=float(diff.std()), frac_folds_better=float((diff > 0).mean()),
                                                         ci95=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))])
    cm = confusion_matrix(y, (oof_final >= 0.5).astype(int)); out["confusion_oof_final"] = cm.tolist()
    # permutation importance (AUC drop) of the plain RF, test folds only
    imp = np.zeros(len(FEATURE_NAMES)); nf = 0
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=7).split(X, y):
        m = rf_plain().fit(X[tr], y[tr])
        pi = permutation_importance(m, X[te], y[te], scoring="roc_auc", n_repeats=5, random_state=0); imp += pi.importances_mean; nf += 1
    out["perm_importance"] = dict(sorted(zip(FEATURE_NAMES, (imp / nf).round(4).tolist()), key=lambda t: -t[1]))
    pd.DataFrame({"file": d.file, "label": y, "oof_final": oof_final, "oof_logreg": oof_lr, "oof_metadata": oof_meta}).to_csv(f"{OUT}/oof_{task}.csv", index=False)
    summary[task] = out
    # ---- final fit on all data + stats for the demo (OOD check, explanation) ----
    final = rf_final().fit(X, y)
    stats = {f: dict(p1=float(np.nanpercentile(X[:, i], 1)), p99=float(np.nanpercentile(X[:, i], 99)),
                     med0=float(np.nanmedian(X[y == 0, i])), med1=float(np.nanmedian(X[y == 1, i])),
                     all=np.sort(X[~np.isnan(X[:, i]), i]).tolist()) for i, f in enumerate(FEATURE_NAMES)}
    joblib.dump(dict(model=final, features=FEATURE_NAMES, stats=stats, n_train=int(len(y)), task=task,
                     metrics=out["final_RF_calibrated"], baseline_metadata_auc=out["metadata_only_RF"]["auc"][0]), f"{MODELS}/{task}_head.joblib")
    print(f"\n===== {task}  (n={len(y)}, pos={int(y.sum())})  [{time.time()-t0:.0f}s] =====")
    print(f"{'model':26s} {'AUC':>13s} {'BalAcc':>8s} {'F1':>6s} {'Sens':>6s} {'Spec':>6s} {'Brier':>7s}")
    for n in names:
        o = out[n]; print(f"{n:26s} {o['auc'][0]:.3f}±{o['auc'][1]:.3f} {o['bal_acc'][0]:8.3f} {o['f1'][0]:6.3f} {o['sens'][0]:6.3f} {o['spec'][0]:6.3f} {o['brier'][0]:7.3f}")
    dd = out["paired_auc_diff_final_minus_metadata"]
    print(f"paired AUC gain over metadata-only: {dd['mean']:+.3f} (95% range {dd['ci95'][0]:+.3f}..{dd['ci95'][1]:+.3f}); better in {dd['frac_folds_better']:.0%} of 25 folds")
    print("confusion (OOF, rows=true 0/1, cols=pred 0/1):", cm.tolist())
    print("top permutation importance:", list(out["perm_importance"].items())[:5])
json.dump(summary, open(f"{OUT}/metrics.json", "w"), indent=1)
