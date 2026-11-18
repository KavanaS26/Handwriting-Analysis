"""Fit an outlier detector on the pooled training features; store thresholds for the demo."""
import os, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(__file__))
from config import OUT, MODELS
from features import FEATURE_NAMES
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
df = pd.read_csv(f"{OUT}/features_clean.csv"); X = df[FEATURE_NAMES].values
imp = SimpleImputer(strategy="median").fit(X); sc = StandardScaler().fit(imp.transform(X)); Z = sc.transform(imp.transform(X))
iso = IsolationForest(n_estimators=500, random_state=0).fit(Z); s = iso.decision_function(Z)
bundle = dict(imputer=imp, scaler=sc, iforest=iso, refuse_below=float(np.percentile(s, 1)), warn_below=float(np.percentile(s, 5)))
joblib.dump(bundle, f"{MODELS}/ood.joblib")
print(f"outlier detector: refuse below {bundle['refuse_below']:.3f} (1st pct), warn below {bundle['warn_below']:.3f} (5th pct)")
