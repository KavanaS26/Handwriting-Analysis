"""Streamlit demo.  Run:  streamlit run app.py     (needs: pip install streamlit)"""
import io, os, sys
import numpy as np, pandas as pd, streamlit as st
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from predict import predict


def show_image(img, **kw):
    """Full-width image that works on both old and new Streamlit versions."""
    try:
        st.image(img, width="stretch", **kw)
    except Exception:
        st.image(img, use_container_width=True, **kw)


def show_table(df):
    try:
        st.dataframe(df, hide_index=True, width="stretch")
    except Exception:
        show_table(df)

st.set_page_config(page_title="Handwriting screening-support (research prototype)", layout="wide")
st.title("Handwriting screening-support prototype")
st.error("Research prototype. These are screening-support scores, NOT diagnoses. Do not use them to make decisions about a child or any person.")

with st.sidebar:
    st.header("How to use")
    st.write("Upload a photo or scan of **at least one line of real handwriting** (several words). "
             "Single characters, blank pages, typed text and non-handwriting images are refused.")
    st.header("What the two scores mean")
    st.write("**Dysgraphia score**: trained on 249 Malay school-child line images (potential vs low potential dysgraphia). "
             "Beats a file-metadata shortcut baseline, but is untested on other languages and populations.")
    st.write("**Dyslexia score (exploratory)**: trained on 100 English images. On its own data it does *not* beat a baseline that never sees the handwriting, "
             "so it must not be read as evidence of dyslexia.")
    st.write("The two scores are independent. No training image carries both labels, so co-occurrence is not modelled.")

up = st.file_uploader("Handwriting image", type=["jpg", "jpeg", "png"])
if up is not None:
    data = up.getvalue()
    res = predict(data)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Your image")
        show_image(Image.open(io.BytesIO(data)).convert("RGB"))
    if not res["ok"]:
        st.warning(res.get("error", "Could not process this image."))
    else:
        with c2:
            st.subheader("Standardized, with detected text lines")
            show_image(res["overlay_bgr"][:, :, ::-1])
            st.caption(f"{res['n_lines']} line(s) detected. Ruled lines removed, ink normalized, deskewed.")
        for w in res["warnings"]:
            st.info(w)
        a, b = st.columns(2)
        for col, key, title in [(a, "dysgraphia", "Dysgraphia score"), (b, "dyslexia", "Dyslexia score (EXPLORATORY)")]:
            h = res["heads"][key]
            with col:
                st.subheader(title)
                st.metric("Model score (0-1)", f"{h['probability']:.2f}")
                st.progress(min(max(h["probability"], 0.0), 1.0))
                (st.success if h["validated"] else st.warning)(h["status"])
                st.caption(f"Cross-validated AUC {h['cv_auc']}; metadata-only baseline AUC {h['metadata_baseline_auc']}; trained on {h['n_train']} images.")
                df = pd.DataFrame(h["explanation"])
                df.columns = ["feature", "your value", "percentile in training set", "median (class 0)", "median (class 1)"]
                with st.expander("Extracted features vs training data"):
                    show_table(df)
