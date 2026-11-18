"""Run this first:  python check_setup.py
Checks libraries, dataset folders, trained models, and runs one real prediction."""
import os, sys, importlib
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(ROOT, "src"))
ok = True
def say(good, msg):
    global ok; ok = ok and good; print(("[ OK ] " if good else "[FAIL] ") + msg)

print(f"Python {sys.version.split()[0]}")
for mod in ["numpy", "pandas", "scipy", "sklearn", "skimage", "cv2", "PIL", "matplotlib", "joblib", "streamlit"]:
    try:
        m = importlib.import_module(mod); say(True, f"{mod} {getattr(m, '__version__', '')}")
    except Exception as e:
        say(False, f"{mod} missing -> run: pip install -r requirements.txt")

data = os.environ.get("HW_DATA", os.path.join(ROOT, "data"))
for sub, n in [("dyslexic/no", 50), ("dyslexic/yes", 50), ("DATASET DYSGRAPHIA HANDWRITING/Low Potential Dysgraphia", 135),
               ("DATASET DYSGRAPHIA HANDWRITING/Potential Dysgraphia", 114)]:
    p = os.path.join(data, sub); c = len(os.listdir(p)) if os.path.isdir(p) else 0
    say(c > 0, f"data/{sub}: {c} images (expected {n})  [needed only for retraining]")

models = os.path.join(ROOT, "models")
try:
    import joblib
    for f in ["dysgraphia_head.joblib", "dyslexia_head.joblib", "ood.joblib"]:
        joblib.load(os.path.join(models, f)); say(True, f"models/{f} loads")
    from predict import predict
    p = os.path.join(data, "dyslexic", "yes"); img = os.path.join(p, sorted(os.listdir(p))[0])
    r = predict(open(img, "rb").read())
    say(r["ok"], "test prediction: " + (", ".join(f"{k}={v['probability']:.2f}" for k, v in r["heads"].items()) if r["ok"] else str(r.get("error"))))
except Exception as e:
    say(False, f"models/prediction problem: {e}\n       -> fix: run  python retrain.py")
print("\nALL GOOD - run the app with:  streamlit run app.py" if ok else "\nSomething failed - see [FAIL] lines above.")
