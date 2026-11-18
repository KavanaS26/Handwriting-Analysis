"""Bad inputs must be refused, not scored. Run after training:  python src/stress_test.py"""
import io, os, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from PIL import Image, ImageDraw, ImageFont
from predict import predict
rng = np.random.RandomState(0); T = {}
T["blank page"] = Image.new("RGB", (900, 600), "white")
T["random noise"] = Image.fromarray(rng.randint(0, 255, (500, 700, 3), dtype=np.uint8))
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
except Exception: font = ImageFont.load_default()
im = Image.new("RGB", (1000, 360), "white"); d = ImageDraw.Draw(im)
for i, t in enumerate(["The quick brown fox jumps over the lazy dog.", "Printed text from a computer is not handwriting at all.", "Neat evenly spaced letters in a regular typeface here."]):
    d.text((30, 30 + i * 100), t, fill="black", font=font)
T["typed text (digital)"] = im; T["tiny 80x40 image"] = Image.new("RGB", (80, 40), "white")
bad = 0
for n, img in T.items():
    b = io.BytesIO(); img.save(b, "PNG"); r = predict(b.getvalue(), want_overlay=False)
    print(f"{n:24s} {'REFUSED' if not r['ok'] else 'SCORED  <-- PROBLEM'}"); bad += r["ok"]
print("PASS" if bad == 0 else f"FAIL: {bad} bad inputs were scored")
