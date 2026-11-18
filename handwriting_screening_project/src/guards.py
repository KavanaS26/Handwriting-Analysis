"""Input guards: reject / warn on images that are not handwriting lines."""
import numpy as np, cv2

def component_stats(bw):
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    st = st[1:]; ok = (st[:, 4] >= 30) & (st[:, 3] >= 8)
    return st[ok]

def duplicate_glyph_fraction(bw):
    """Share of letter-sized components whose (width, height, area) exactly repeats.
    Digitally typed text repeats glyph shapes pixel-for-pixel; handwriting almost never does."""
    st = component_stats(bw)
    if len(st) < 12: return 0.0
    med = np.median(st[:, 3]); st = st[st[:, 3] >= 0.5 * med]
    keys = [tuple(r[[2, 3, 4]]) for r in st]
    from collections import Counter
    c = Counter(keys)
    return float(sum(v for v in c.values() if v >= 2) / len(keys))
