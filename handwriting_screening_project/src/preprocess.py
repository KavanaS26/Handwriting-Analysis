"""Step 2-3: standardize any handwriting image into one canonical format.

Canonical format: binary uint8 image, INK = 255, background = 0, ruled lines
removed, deskewed, cropped to ink, working width = 1000 px.
Works for photos/scans (dyslexia set) and pre-binarized white-on-black
line images (dysgraphia set) alike, so source-specific cues are removed.
"""
import cv2
import numpy as np
from PIL import Image

WORK_W = 1000


def load_gray(path):
    """Load any image as uint8 gray, compositing transparency over white."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.array(im.convert("L"))


def fix_polarity(gray):
    """Make ink dark on light paper. Decide from the border/median brightness."""
    b = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    bg_level = np.median(np.concatenate([b, gray.ravel()[::7]]))
    return 255 - gray if bg_level < 127 else gray


def resize_width(gray, w=WORK_W):
    h0, w0 = gray.shape
    scale = w / w0
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(gray, (w, max(8, int(round(h0 * scale)))), interpolation=interp)


def binarize(gray):
    """CLAHE -> Gaussian blur -> Otsu (inverse so ink = 255)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def remove_ruled_lines(bw):
    """Remove long horizontal ruled/underline strokes (paper, not handwriting)."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, bw.shape[1] // 12), 1))
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    # long vertical rules (page margins, box edges)
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(60, bw.shape[0] // 4)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kv)
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    return cv2.bitwise_and(bw, cv2.bitwise_not(cv2.bitwise_or(lines, vert)))


def clean_noise(bw, min_area=12):
    """Morphological opening + drop tiny specks."""
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    keep = np.zeros(n, bool)
    area = stats[1:, cv2.CC_STAT_AREA]
    w, h = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]
    flat = (h <= 7) & (w >= 14)            # leftover dashes of ruled lines
    keep[1:] = (area >= min_area) & ~flat
    return (keep[lab] * 255).astype(np.uint8)


def estimate_skew(bw, search=6.0, step=0.5):
    """Projection-profile deskew: the angle that makes row sums most peaky."""
    small = cv2.resize(bw, (bw.shape[1] // 2, bw.shape[0] // 2), interpolation=cv2.INTER_AREA)
    h, w = small.shape
    best, best_a = -1, 0.0
    for a in np.arange(-search, search + 1e-6, step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(small, M, (w, h), flags=cv2.INTER_NEAREST)
        s = np.var(r.sum(axis=1).astype(np.float64))
        if s > best:
            best, best_a = s, a
    return best_a


def deskew(bw):
    a = estimate_skew(bw)
    if abs(a) < 0.25:
        return bw, 0.0
    h, w = bw.shape
    pad = 20
    big = cv2.copyMakeBorder(bw, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    H, W = big.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), a, 1.0)
    out = cv2.warpAffine(big, M, (W, H), flags=cv2.INTER_LINEAR)
    _, out = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)
    return out, a


def crop_to_ink(bw, margin=10):
    ys, xs = np.where(bw > 0)
    if len(ys) == 0:
        return bw
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    y0, x0 = max(0, y0 - margin), max(0, x0 - margin)
    return bw[y0:y1 + margin + 1, x0:x1 + margin + 1]


def standardize(path, return_steps=False):
    g0 = load_gray(path)
    g1 = fix_polarity(g0)
    g2 = resize_width(g1)
    bw = binarize(g2)
    bw = remove_ruled_lines(bw)
    bw = clean_noise(bw)
    bw, angle = deskew(bw)
    bw = clean_noise(crop_to_ink(bw))
    bw = crop_to_ink(bw)
    if return_steps:
        return bw, {"raw": g0, "polar": g1, "resized": g2, "angle": angle}
    return bw
