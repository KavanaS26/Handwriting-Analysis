import os, sys, json
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from config import OUT
FIG = f"{OUT}/figures"; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
M = json.load(open(f"{OUT}/metrics.json")); P = json.load(open(f"{OUT}/dyslexia2_results.json"))
# Fig 1: ROC curves, model vs shortcut baseline
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
for a, t, title in zip(ax, ["dysgraphia", "dyslexia"], ["Dysgraphia head (Mendeley, n=249)", "Dyslexia head (Kaggle, n=100) - exploratory"]):
    d = pd.read_csv(f"{OUT}/oof_{t}.csv")
    for col, lab, c, ls in [("oof_final", "Features, calibrated RF", "#1f77b4", "-"), ("oof_logreg", "Features, logistic reg.", "#2ca02c", "--"), ("oof_metadata", "File metadata only (shortcut)", "#d62728", ":")]:
        f, tp, _ = roc_curve(d.label, d[col]); a.plot(f, tp, color=c, ls=ls, lw=2, label=f"{lab} (AUC {roc_auc_score(d.label, d[col]):.2f})")
    a.plot([0, 1], [0, 1], color="grey", lw=1); a.set_title(title, fontsize=11); a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate"); a.legend(loc="lower right", fontsize=8.5)
plt.suptitle("Out-of-fold ROC (5x5 repeated CV, predictions averaged over repeats)", fontsize=10); plt.tight_layout(); plt.savefig(f"{FIG}/fig1_roc.png", dpi=170); plt.close()
# Fig 2: does the model beat the shortcut? across the three datasets
labels = ["Dysgraphia\n(Mendeley, lines,\nrandom CV)", "Dyslexia\n(Kaggle, pages,\nrandom CV)", "Dyslexia\n(Pakistan, characters,\nsession-grouped CV)"]
model = [M["dysgraphia"]["final_RF_calibrated"]["auc"], M["dyslexia"]["final_RF_calibrated"]["auc"], P["standardized SHAPE features (10), RF"]["grouped"]]
base = [M["dysgraphia"]["metadata_only_RF"]["auc"], M["dyslexia"]["metadata_only_RF"]["auc"], P["raw photo metadata (acquisition only)"]["grouped"]]
x = np.arange(3); w = 0.36; fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.bar(x - w / 2, [m[0] for m in model], w, yerr=[m[1] for m in model], color="#1f77b4", label="Handwriting features", capsize=3)
ax.bar(x + w / 2, [b[0] for b in base], w, yerr=[b[1] for b in base], color="#d62728", label="File metadata only (no handwriting)", capsize=3)
for i in range(3):
    ax.text(x[i] - w / 2, model[i][0] + model[i][1] + 0.01, f"{model[i][0]:.2f}", ha="center", fontsize=9); ax.text(x[i] + w / 2, base[i][0] + base[i][1] + 0.01, f"{base[i][0]:.2f}", ha="center", fontsize=9)
ax.axhline(0.5, color="grey", ls="--", lw=1); ax.set_ylim(0.4, 1.12); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9); ax.set_ylabel("ROC-AUC (mean +/- fold std)")
ax.set_title("Only the dysgraphia head clearly beats the shortcut baseline"); ax.legend(loc="upper right", fontsize=9, ncol=1); plt.tight_layout(); plt.savefig(f"{FIG}/fig2_vs_shortcut.png", dpi=170); plt.close()
# Fig 3: permutation importance
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, t in zip(ax, ["dysgraphia", "dyslexia"]):
    imp = list(M[t]["perm_importance"].items())[:8][::-1]; a.barh([k for k, _ in imp], [v for _, v in imp], color="#4c72b0"); a.set_title(f"{t.capitalize()} head"); a.set_xlabel("AUC drop when feature is shuffled")
plt.tight_layout(); plt.savefig(f"{FIG}/fig3_importance.png", dpi=170); plt.close()
print("figures:", sorted(os.listdir(FIG)))
