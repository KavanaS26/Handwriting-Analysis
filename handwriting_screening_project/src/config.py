"""Central paths. Set HW_DATA to the folder that holds the three datasets:
   <HW_DATA>/dyslexic/{no,yes}/                                   (Kaggle dyslexia)
   <HW_DATA>/DATASET DYSGRAPHIA HANDWRITING/{Low Potential Dysgraphia,Potential Dysgraphia}/   (Mendeley dysgraphia)
   <HW_DATA>/pakistan_dyslexia/{No,Yes}/                          (Mendeley character-level dyslexia, optional)
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("HW_DATA", os.path.join(ROOT, "data"))
OUT = os.path.join(ROOT, "out")
MODELS = os.path.join(ROOT, "models")
for _d in (OUT, MODELS):
    os.makedirs(_d, exist_ok=True)
