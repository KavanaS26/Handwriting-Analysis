# Handwriting screening-support prototype

Research prototype: upload a handwriting image and get two independent, **exploratory** scores
(dysgraphia and dyslexia-labelled). It is **not a diagnostic tool** and must not be used to make
decisions about a child or any person.

## Run it (Windows)
1. Install Python 3.11+ from python.org (tick **Add Python to PATH**).
2. Double-click **`run_windows.bat`**. It creates a virtual environment, installs the pinned
   libraries, checks the setup and opens the app at http://localhost:8501.

## Run it (macOS / Linux)
`sh run_mac_linux.sh`

## Manual commands
```
python -m venv venv
venv\Scripts\activate            (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python check_setup.py            # verifies libraries, data, models, one real prediction
streamlit run app.py             # web demo
python src/predict.py my_handwriting.jpg    # command-line prediction
```
Use an image with **at least one full line of real handwriting** (several words). Blank pages,
noise, typed text and tiny images are refused on purpose.

## Re-create the models from the data
The datasets are in `data/`. One command rebuilds everything (about 6-8 minutes):
```
python retrain.py
```
It runs, in order: `run_extract.py` (images -> 16 features), `check_leakage_and_baseline.py`
(cleaning), `train.py` (both heads + cross-validation vs. metadata baseline and shuffle control),
`build_ood.py` (refuse-bad-input detector), `stat_tests.py` (Nadeau-Bengio corrected t-test,
bootstrap CIs) and `make_figures.py`. Results and logs go to `out/`, models to `models/`.
Back up `models/` first if you want to keep the current ones.

## Folder layout
```
app.py                Streamlit demo          retrain.py       rebuild all models/results
check_setup.py        environment check       requirements.txt pinned library versions
src/                  pipeline code           models/          trained heads + outlier detector
data/                 the datasets            out/             features, logs, metrics, figures
```
Data layout expected under `data/` (or set the `HW_DATA` environment variable):
`dyslexic/{no,yes}`, `DATASET DYSGRAPHIA HANDWRITING/{Low Potential Dysgraphia,Potential Dysgraphia}`,
and optionally `pakistan_dyslexia/{No,Yes}` (only for the audit scripts; not included).

## Verified results (re-run from the images in `data/`)
| Task | Handwriting model AUC | Metadata-only AUC | Paired difference | Corrected p |
|---|---|---|---|---|
| Dysgraphia (249 images) | 0.942 +/- 0.027 | 0.823 +/- 0.067 | +0.119 | 0.001 |
| Dyslexia-labelled (100 images) | 0.986 +/- 0.024 | 0.978 +/- 0.022 | +0.008 | 0.60 |

The dysgraphia head clearly beats the metadata shortcut; the dyslexia head does not, so its score is
exploratory only. Splits are image-level (no child IDs exist), so real-world performance on unseen
children is likely lower. The paper in the parent folder describes the study in full.

## Known limitations
- Streamlit's browser interface depends on the installed Streamlit version (1.67.0 is pinned).
- Scanned *printed* text is not reliably refused; about 1% of genuine handwriting is refused in-sample.
- Datasets are small and population-specific; the dysgraphia set is Malay handwriting.
- Check the licence of each dataset before redistributing `data/`.
