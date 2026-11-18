"""Step 4: language-independent handcrafted handwriting features.

All values are computed per text line on the standardized binary crop and
normalised by the median component height (a letter-size proxy), so they do
not depend on image resolution, zoom or language. No OCR is used.
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize

FEATURE_NAMES = [
    "n_lines", "line_height_cv",
    "letter_height_cv",      # variation of letter sizes            (dysgraphia)
    "baseline_dev",          # drift of letters off the baseline    (dysgraphia)
    "baseline_slope_abs",    # overall tilt of the baseline
    "stroke_mean", "stroke_cv",   # stroke thickness / pressure control
    "slant_var", "slant_mean_abs",  # slant consistency / mean slant
    "word_gap_cv", "word_gap_mean",  # inter-word spacing irregularity
    "letter_gap_cv",         # irregular spacing inside words
    "ink_density", "size_ratio", "comps_per_line", "touching_ratio",
]


def _components(crop, min_area=12):
    n, lab, st, cen = cv2.connectedComponentsWithStats(crop, connectivity=8)
    min_h = max(6, 0.12 * crop.shape[0])   # ignore specks far smaller than the line
    comps = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a >= min_area and h >= min_h:
            comps.append((x, y, w, h, a, i))
    return comps, lab


def _robust_baseline(xs, ys):
    """Linear baseline fit with one pass of outlier rejection. Returns (resid_std, slope)."""
    if len(xs) < 4:
        return np.nan, np.nan
    p = np.polyfit(xs, ys, 1)
    r = ys - np.polyval(p, xs)
    mad = np.median(np.abs(r - np.median(r))) + 1e-6
    ok = np.abs(r) < 3 * 1.4826 * mad
    if ok.sum() >= 4:
        p = np.polyfit(xs[ok], ys[ok], 1)
        r = ys[ok] - np.polyval(p, xs[ok])
    return float(np.std(r)), float(p[0])


def line_features(crop):
    comps, lab = _components(crop)
    if len(comps) < 3:
        return None
    hs = np.array([c[3] for c in comps], float)
    ws = np.array([c[2] for c in comps], float)
    med_h = np.median(hs)
    if med_h < 5:
        return None
    f = {}
    # letter-size variation (ignore dots / tiny marks)
    big = hs[hs >= 0.4 * med_h]
    f["letter_height_cv"] = float(np.std(big) / np.mean(big)) if len(big) > 2 else np.nan
    f["size_ratio"] = float(med_h / crop.shape[0])
    f["comps_per_line"] = float(len(comps))
    f["ink_density"] = float((crop > 0).mean())
    # baseline: bottoms of non-descender-ish, letter-sized components
    sel = [c for c in comps if c[3] >= 0.5 * med_h]
    xs = np.array([c[0] + c[2] / 2 for c in sel], float)
    bot = np.array([c[1] + c[3] for c in sel], float)
    sd, slope = _robust_baseline(xs, bot)
    f["baseline_dev"] = sd / med_h if not np.isnan(sd) else np.nan
    f["baseline_slope_abs"] = abs(slope) if not np.isnan(slope) else np.nan
    # stroke thickness via distance transform sampled on the skeleton
    dt = cv2.distanceTransform((crop > 0).astype(np.uint8), cv2.DIST_L2, 3)
    sk = skeletonize(crop > 0)
    th = 2 * dt[sk]
    th = th[th > 0]
    if len(th) > 10:
        f["stroke_mean"] = float(th.mean() / med_h)
        f["stroke_cv"] = float(th.std() / th.mean())
    else:
        f["stroke_mean"] = f["stroke_cv"] = np.nan
    # slant: principal-axis angle of tall components, measured from vertical
    angs = []
    for (x, y, w, h, a, i) in comps:
        if h >= 0.6 * med_h and h / max(w, 1) >= 1.3:
            m = cv2.moments((lab[y:y + h, x:x + w] == i).astype(np.uint8), binaryImage=True)
            if m["mu20"] + m["mu02"] > 0:
                th_ = 0.5 * np.degrees(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))
                dev = th_ - 90 if th_ > 0 else th_ + 90   # tilt from vertical, image coords
                if abs(dev) < 60:
                    angs.append(dev)
    if len(angs) >= 3:
        f["slant_var"] = float(np.var(angs))
        f["slant_mean_abs"] = float(abs(np.mean(angs)))
    else:
        f["slant_var"] = f["slant_mean_abs"] = np.nan
    # horizontal gaps: split into letter-gaps and word-gaps by size
    col = (crop > 0).any(axis=0).astype(int)
    d = np.diff(np.concatenate([[0], col, [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    gaps = (starts[1:] - ends[:-1]).astype(float) / med_h if len(starts) > 1 else np.array([])
    word = gaps[gaps > 0.55]
    letter = gaps[(gaps > 0) & (gaps <= 0.55)]
    f["word_gap_cv"] = float(word.std() / word.mean()) if len(word) >= 3 else np.nan
    f["word_gap_mean"] = float(word.mean()) if len(word) >= 1 else np.nan
    f["letter_gap_cv"] = float(letter.std() / letter.mean()) if len(letter) >= 4 and letter.mean() > 0 else np.nan
    # touching/overlapping letters: how much wider-than-expected the components are
    f["touching_ratio"] = float(np.mean(ws > 1.8 * med_h))
    return f


def image_features(lines):
    """Aggregate per-line features (mean across lines) into one vector per image."""
    per = [line_features(c) for (_, _, c) in lines]
    per = [p for p in per if p is not None]
    if not per:
        return None
    out = {"n_lines": float(len(per))}
    hs = np.array([c.shape[0] for (_, _, c) in lines], float)
    out["line_height_cv"] = float(hs.std() / hs.mean()) if len(hs) > 1 else 0.0
    for k in FEATURE_NAMES:
        if k in out:
            continue
        v = np.array([p.get(k, np.nan) for p in per], float)
        out[k] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    return out
