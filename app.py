import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UTI Prediction Tool",
    page_icon="🔬",
    layout="centered"
)

# ── Load model, feature list, and encoding map ─────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join("outputs", "models", "model.pkl"))

    with open(os.path.join("outputs", "selected_features.json"), "r") as f:
        features = json.load(f)

    with open(os.path.join("outputs", "encoding_map.json"), "r") as f:
        encoding_map = json.load(f)

    return model, features, encoding_map

model, selected_features, encoding_map = load_model()

# ── Feature metadata ───────────────────────────────────────────────────────────
# Options match the exact raw string values from training data
# Order matches encoding_map so the user sees meaningful labels
FEATURE_META = {
    "abx": {
        "label"  : "Were antibiotics prescribed?",
        "options": ["No", "Yes"]
    },
    "ua_wbc": {
        "label"  : "Urine WBC (White Blood Cells)",
        "options": ["negative", "small", "moderate", "large", "other", "not_reported"]
    },
    "ANTIBIOTICS": {
        "label"  : "Antibiotic drug class administered?",
        "options": ["No", "Yes"]
    },
    "ua_bacteria": {
        "label"  : "Urine Bacteria",
        "options": ["none", "few", "moderate", "marked", "many", "not_reported"]
    },
    "ua_ph": {
        "label"  : "Urine pH",
        "options": ["5.0", "5.5", "6.0", "6.5", "7.0", "7.5",
                    "8.0", "8.5", "9.0", "not_reported", "other"]
    },
    "ua_leuk": {
        "label"  : "Leukocyte Esterase (dipstick)",
        "options": ["negative", "small", "moderate", "large", "other", "not_reported"]
    },
    "ua_clarity": {
        "label"  : "Urine Clarity",
        "options": ["clear", "not_clear", "not_reported"]
    },
    "gender": {
        "label"  : "Patient Gender",
        "options": ["Female", "Male", "not_reported"]
    },
    "ANTIARTHRITICS": {
        "label"  : "Antiarthritic medication (current)?",
        "options": ["No", "Yes"]
    },
    "ua_nitrite": {
        "label"  : "Urine Nitrite (dipstick)",
        "options": ["negative", "positive", "other", "not_reported"]
    },
    "ANTIASTHMATICS": {
        "label"  : "Antiasthmatic medication (current)?",
        "options": ["No", "Yes"]
    },
    "Anion_Gap": {
        "label"  : "Anion Gap (blood chemistry bin 1–5)",
        "options": ["1", "2", "3", "4", "5", "not_reported"]
    },
    "ua_urobili": {
        "label"  : "Urine Urobilinogen",
        "options": ["negative", "positive", "not_reported"]
    },
    "dysuria": {
        "label"  : "Dysuria (painful urination)?",
        "options": ["0", "1", "not_reported"]
    },
    "ANALGESICS": {
        "label"  : "Analgesic medication (current)?",
        "options": ["No", "Yes"]
    },
}

# ── Encode inputs using the fixed encoding map ─────────────────────────────────
def encode_inputs(raw_inputs: dict) -> pd.DataFrame:
    """
    Looks up each dropdown string value in the encoding map
    and returns the exact integer the model was trained on.
    Falls back to 0 and warns if a value isn't found.
    """
    encoded = {}
    for feature, value in raw_inputs.items():
        feature_map = encoding_map[feature]      # {"No": 0, "Yes": 1, ...}
        code = feature_map.get(str(value), None)
        if code is None:
            st.warning(f"Unrecognised value '{value}' for '{feature}' — defaulting to 0")
            code = 0
        encoded[feature] = int(code)

    return pd.DataFrame([encoded])[selected_features]

# ── App header ─────────────────────────────────────────────────────────────────
st.title("🔬 UTI Prediction Tool")
st.markdown("""
This tool uses a machine learning model trained on the Taylor et al. (2018)
emergency department dataset to predict the likelihood of a urinary tract
infection based on clinical inputs.

> **Note:** This tool is for research and educational purposes only.  
> It is not a substitute for clinical judgement or professional medical advice.
""")

st.divider()

# ── Input form ─────────────────────────────────────────────────────────────────
st.subheader("Patient Clinical Inputs")
st.markdown("Complete all fields and click **Predict** to generate a UTI risk assessment.")

raw_inputs = {}

col1, col2 = st.columns(2)
feature_list = list(FEATURE_META.keys())
mid = len(feature_list) // 2

with col1:
    for feature in feature_list[:mid]:
        meta = FEATURE_META[feature]
        raw_inputs[feature] = st.selectbox(
            label=meta["label"],
            options=meta["options"],
            key=feature
        )

with col2:
    for feature in feature_list[mid:]:
        meta = FEATURE_META[feature]
        raw_inputs[feature] = st.selectbox(
            label=meta["label"],
            options=meta["options"],
            key=feature
        )

st.divider()

# ── Prediction ─────────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True):

    input_df    = encode_inputs(raw_inputs)
    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("### 🔴 UTI Likely")
        st.metric(label="Estimated UTI Probability", value=f"{probability * 100:.1f}%")
        st.markdown("The model predicts a **high likelihood of UTI** based on the provided inputs.")
    else:
        st.success("### 🟢 UTI Unlikely")
        st.metric(label="Estimated UTI Probability", value=f"{probability * 100:.1f}%")
        st.markdown("The model predicts a **low likelihood of UTI** based on the provided inputs.")

    # ── SHAP waterfall plot ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Feature Contribution Breakdown")
    st.markdown("""
    The chart below shows how each input pushed the prediction  
    toward **(red → UTI likely)** or away from **(blue → UTI unlikely)** a positive result.
    """)

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values        = shap_values[0],
                base_values   = explainer.expected_value,
                data          = input_df.iloc[0],
                feature_names = selected_features
            ),
            show=False
        )
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.warning(f"SHAP plot could not be generated: {e}")

    # ── Input summaries ────────────────────────────────────────────────────────
    with st.expander("Show encoded values sent to model"):
        st.dataframe(input_df, use_container_width=True)

    with st.expander("Show raw input values"):
        st.dataframe(
            pd.DataFrame(raw_inputs, index=["Value"]).T,
            use_container_width=True
        )