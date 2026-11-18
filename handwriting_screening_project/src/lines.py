"""Step 2: split a standardized binary page into text-line crops."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def _valley_runs(sm):
    """Text-line bands from peaks of the row profile; cut at the valleys between peaks."""
    peaks, _ = find_peaks(sm, distance=22, prominence=0.15 * sm.max())
    if len(peaks) < 2:
        return None
    cuts = [0] + [int(a + np.argmin(sm[a:b])) for a, b in zip(peaks[:-1], peaks[1:])] + [len(sm)]
    return [[cuts[i], cuts[i + 1]] for i in range(len(cuts) - 1)]


def segment_lines(bw, min_h=14):
    """Projection-profile line segmentation (valley-based, threshold fallback)."""
    prof = (bw > 0).sum(axis=1).astype(np.float64)
    if prof.max() == 0:
        return []
    sm = gaussian_filter1d(prof, sigma=max(2.0, bw.shape[1] / 300))
    runs = _valley_runs(sm)
    used_valley = runs is not None
    if runs is None:
        on = sm > 0.12 * sm.max()
        runs, start = [], None
        for i, v in enumerate(on):
            if v and start is None:
                start = i
            elif not v and start is not None:
                runs.append([start, i]); start = None
        if start is not None:
            runs.append([start, len(on)])
    # merge runs separated by tiny gaps (i-dots, accents)
    merged = []
    for r in runs:
        if (not used_valley) and merged and r[0] - merged[-1][1] < 6:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    out = []
    for y0, y1 in merged:
        # expand a little so ascenders/descenders aren't clipped
        base = y0
        crop = bw[y0:y1]
        rows = np.where((crop > 0).any(axis=1))[0]
        if len(rows) == 0:
            continue
        a, b = base + rows.min(), base + rows.max() + 1
        crop = bw[a:b]
        ink = (crop > 0).sum()
        if (b - a) >= min_h and ink > 80:
            ys, xs = np.where(crop > 0)
            crop = crop[:, xs.min():xs.max() + 1]
            out.append((a, b, crop))
    return out
