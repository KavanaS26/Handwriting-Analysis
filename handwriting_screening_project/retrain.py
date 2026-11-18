"""Rebuild everything from the images in data/:  python retrain.py   (about 6-8 minutes)
extract features -> clean -> train both heads -> outlier detector -> statistics -> figures."""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(ROOT, "src")
steps = ["run_extract.py", "check_leakage_and_baseline.py", "train.py", "build_ood.py", "stat_tests.py", "make_figures.py"]
for i, s in enumerate(steps, 1):
    print(f"\n=== [{i}/{len(steps)}] {s} ===", flush=True)
    if subprocess.call([sys.executable, os.path.join(S, s)], cwd=ROOT) != 0:
        sys.exit(f"Step failed: {s}")
print("\nDone. New models are in models/, results in out/.  Next: python check_setup.py")
