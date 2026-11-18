"""Inference: one handwriting image in -> standardized image, features, both scores, warnings.

Usage:  python src/predict.py path/to/image.jpg [--save-dir out_dir]
Both outputs are independent SCREENING-SUPPORT scores, not diagnoses.
"""
import os, sys, io, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, cv2, joblib
sys.path.insert(0, os.path.dirname(__file__))
from config import MODELS
from preprocess import standardize
from lines import segment_lines
from features import image_features, FEATURE_NAMES
from guards import component_stats, duplicate_glyph_fraction

_cache = {}
def load_heads():
    if not _cache:
        for t in ["dyslexia", "dysgraphia"]:
            _cache[t] = joblib.load(os.path.join(MODELS, f"{t}_head.joblib"))
        _cache["ood"] = joblib.load(os.path.join(MODELS, "ood.joblib"))
    return _cache

def _percentile(sorted_vals, v):
    return float(np.searchsorted(sorted_vals, v) / len(sorted_vals) * 100)

def predict(image, want_overlay=True):
    """image: file path, bytes, or file-like object."""
    heads = load_heads()
    if isinstance(image, (bytes, bytearray)):
        image = io.BytesIO(image)
    res = {"ok": False, "warnings": [], "features": {}, "heads": {}}
    try:
        bw = standardize(image)
    except Exception as e:
        res["error"] = f"Could not read the image ({e})."; return res
    lines = segment_lines(bw)
    res["n_lines"] = len(lines); res["std_shape"] = list(bw.shape)
    # ---- hard input guards: withhold scores for non-handwriting input ----
    if len(component_stats(bw)) < 8:
        res["error"] = "Too little writing found (looks like a single character, a mark or a blank page). Upload at least one line of handwriting (several words)."
        return res
    dup = duplicate_glyph_fraction(bw); res["duplicate_glyph_fraction"] = round(dup, 3)
    if dup > 0.35:
        res["error"] = "This looks like digitally typed text (identical letter shapes repeat). Upload a photo or scan of real handwriting."
        return res
    feats = image_features(lines) if lines else None
    if feats is None:
        res["error"] = ("No usable handwriting found. The system needs at least one line of handwriting "
                        "(several words), not a single character, a blank page or a very faint photo.")
        return res
    res["features"] = {k: (None if v != v else round(float(v), 4)) for k, v in feats.items()}
    x = np.array([[feats.get(f, np.nan) for f in FEATURE_NAMES]], float)
    o = heads["ood"]; osc = float(o["iforest"].decision_function(o["scaler"].transform(o["imputer"].transform(x)))[0])
    res["outlier_score"] = round(osc, 3)
    if osc < o["refuse_below"]:
        res["error"] = ("This image is too unlike the handwriting the system was built on (outlier score %.2f), e.g. noise, a photo of "
                        "something else, or an extreme crop. Scores are withheld." % osc)
        return res
    if osc < o["warn_below"]:
        res["warnings"].append("Unusual input (outlier score %.2f): treat the scores with extra caution." % osc)
    # ---- input sanity / out-of-distribution warnings ----
    ref = heads["dysgraphia"]["stats"]; ref2 = heads["dyslexia"]["stats"]
    out_of = []
    for i, f in enumerate(FEATURE_NAMES):
        v = x[0, i]
        if v != v: continue
        lo = min(ref[f]["p1"], ref2[f]["p1"]); hi = max(ref[f]["p99"], ref2[f]["p99"]); pad = 0.15 * (hi - lo + 1e-9)
        if v < lo - pad or v > hi + pad: out_of.append(f)
    res["out_of_range_features"] = out_of
    nan_frac = float(np.isnan(x).mean())
    if len(lines) < 2 and feats.get("comps_per_line", 0) < 8:
        res["warnings"].append("Very little writing detected; scores are unreliable. Use a line or more of handwriting.")
    if len(out_of) >= 4 or nan_frac > 0.35:
        res["warnings"].append(f"This image looks unlike the training data ({len(out_of)} features out of range, "
                               f"{nan_frac:.0%} missing). Treat the scores as meaningless.")
    res["warnings"].append("The system does not verify that the text is handwritten; scanned printed text is not detected.")
    if bw.shape[1] < 300:
        res["warnings"].append("Image is very small; features may be noisy.")
    # ---- per-head scores ----
    for t, meta in [("dysgraphia", heads["dysgraphia"]), ("dyslexia", heads["dyslexia"])]:
        p = float(meta["model"].predict_proba(x)[0, 1])
        st = meta["stats"]
        expl = []
        for i, f in enumerate(FEATURE_NAMES):
            v = x[0, i]
            if v != v: continue
            pct = _percentile(np.array(st[f]["all"]), v)
            expl.append(dict(feature=f, value=round(float(v), 4), percentile=round(pct), median_neg=round(st[f]["med0"], 3), median_pos=round(st[f]["med1"], 3)))
        res["heads"][t] = dict(probability=round(p, 3), explanation=expl, n_train=meta["n_train"],
                               cv_auc=round(meta["metrics"]["auc"][0], 3), metadata_baseline_auc=round(meta["baseline_metadata_auc"], 3),
                               validated=(t == "dysgraphia"))
    res["heads"]["dyslexia"]["status"] = ("EXPLORATORY: on its training data this head performs no better than a model that never sees the "
                                          "handwriting (file properties alone), so its score must not be read as evidence of dyslexia.")
    res["heads"]["dysgraphia"]["status"] = ("Screening-support score for 'potential dysgraphia' (Malay child handwriting, 249 images). "
                                            "Not validated on other languages or populations.")
    if want_overlay:
        rgb = cv2.cvtColor(255 - bw, cv2.COLOR_GRAY2BGR)
        for i, (a, b, c) in enumerate(lines):
            cv2.rectangle(rgb, (2, a), (rgb.shape[1] - 3, b), (0, 0, 220), 2)
            cv2.putText(rgb, str(i + 1), (6, a + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 0, 0), 2)
        res["overlay_bgr"] = rgb; res["standardized"] = 255 - bw
    res["ok"] = True
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("image"); ap.add_argument("--save-dir", default=None)
    a = ap.parse_args(); r = predict(a.image)
    if not r["ok"]:
        print("ERROR:", r.get("error")); sys.exit(1)
    print(f"lines detected: {r['n_lines']}")
    for t in ["dysgraphia", "dyslexia"]:
        h = r["heads"][t]; print(f"\n[{t}] score = {h['probability']:.2f}   (CV AUC {h['cv_auc']}, metadata-only baseline {h['metadata_baseline_auc']})\n  {h['status']}")
    for w in r["warnings"]: print("WARNING:", w)
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True); b = os.path.splitext(os.path.basename(a.image))[0]
        cv2.imwrite(f"{a.save_dir}/{b}_lines.png", r["overlay_bgr"]); json.dump({k: v for k, v in r.items() if not k.endswith("bgr") and k != "standardized"}, open(f"{a.save_dir}/{b}.json", "w"), indent=1)
