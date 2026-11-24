import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Water Potability Predictor", page_icon="💧", layout="wide")


# Must match the quantiles you used to build df_quantile_capped
LOWER_Q, UPPER_Q = 0.01, 0.99


@st.cache_resource
def load_artifacts():
    with open("random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("dataset.pkl", "rb") as f:
        df = pickle.load(f)
    return model, df


try:
    model, df = load_artifacts()
except FileNotFoundError:
    st.error("Put random_forest_model.pkl and dataset.pkl in the same folder as app.py.")
    st.stop()

# Use the exact feature names/order the model was trained on
if hasattr(model, "feature_names_in_"):
    features = list(model.feature_names_in_)
else:
    features = [c for c in df.columns if c != "Potability"]

# Preprocessing info derived from the dataset
medians = df[features].median()
lower = df[features].quantile(LOWER_Q)
upper = df[features].quantile(UPPER_Q)
importances = dict(zip(features, model.feature_importances_))

UNITS = {
    "ph": "pH",
    "Hardness": "mg/L",
    "Solids": "ppm (TDS)",
    "Chloramines": "ppm",
    "Sulfate": "mg/L",
    "Conductivity": "µS/cm",
    "Organic_carbon": "ppm",
    "Trihalomethanes": "µg/L",
    "Turbidity": "NTU",
}


def preprocess(X: pd.DataFrame) -> pd.DataFrame:
    """Same steps as training: median-impute, then cap to training bounds."""
    X = X[features].astype(float).fillna(medians)
    return X.clip(lower, upper, axis=1)


def predict(X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(preprocess(X))[:, 1]


st.title("💧 Water Potability Predictor")
st.caption("Random Forest model that estimates the probability that a water sample is safe to drink.")

threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                              help="Probability at or above this value is classified as potable.")
st.sidebar.markdown("---")
st.sidebar.info("This is a statistical model, not a lab test. Never use it as the only "
                "basis for deciding whether water is safe to drink.")

tab_single, tab_batch, tab_model = st.tabs(["Single sample", "Batch (CSV)", "About the model"])

# ---------------------------------------------------------------- single
with tab_single:
    st.subheader("Enter water measurements")
    st.caption("Tick “Unknown” for a missing value; it will be filled with the training median.")

    values = {}
    cols = st.columns(3)
    for i, f in enumerate(features):
        with cols[i % 3]:
            unknown = st.checkbox("Unknown", key=f"unk_{f}")
            lo, hi = float(lower[f]), float(upper[f])
            values[f] = np.nan if unknown else st.number_input(
                f"{f} ({UNITS.get(f, '')})",
                value=float(round(medians[f], 3)),
                min_value=min(lo, 0.0), max_value=hi * 3,
                format="%.4f", disabled=unknown, key=f"val_{f}",
            )

    if st.button("Predict", type="primary"):
        p = float(predict(pd.DataFrame([values]))[0])
        potable = p >= threshold

        c1, c2 = st.columns([1, 2])
        c1.metric("P(potable)", f"{p:.1%}")
        if potable:
            c2.success(f"Predicted: **Potable** (probability {p:.1%} ≥ threshold {threshold:.0%})")
        else:
            c2.error(f"Predicted: **Not potable** (probability {p:.1%} < threshold {threshold:.0%})")
        st.progress(min(max(p, 0.0), 1.0))

        raw = pd.Series(values)
        if raw.isna().any():
            st.warning("Filled with median: " + ", ".join(raw[raw.isna()].index))
        clipped = [f for f in features
                   if not np.isnan(values[f]) and not (lower[f] <= values[f] <= upper[f])]
        if clipped:
            st.warning("Outside the training range, so capped: " + ", ".join(clipped))

# ---------------------------------------------------------------- batch
with tab_batch:
    st.subheader("Predict many samples")
    st.write("Upload a CSV with these columns: " + ", ".join(f"`{f}`" for f in features)
             + ". An optional `Potability` column is used only for comparison.")
    template = pd.DataFrame([medians.round(3)], columns=features)
    st.download_button("Download CSV template", template.to_csv(index=False),
                       "template.csv", "text/csv")

    up = st.file_uploader("Upload CSV", type="csv")
    if up:
        data = pd.read_csv(up)
        missing = [f for f in features if f not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            proba = predict(data)
            out = data.copy()
            out["P(potable)"] = proba.round(4)
            out["Predicted"] = (proba >= threshold).astype(int)

            m1, m2, m3 = st.columns(3)
            m1.metric("Samples", len(out))
            m2.metric("Predicted potable", int(out["Predicted"].sum()))
            if "Potability" in out.columns:
                m3.metric("Agreement with Potability",
                          f"{(out['Predicted'] == out['Potability']).mean():.1%}")

            st.dataframe(out, use_container_width=True)
            st.download_button("Download predictions", out.to_csv(index=False),
                               "predictions.csv", "text/csv")

# ---------------------------------------------------------------- model
with tab_model:
    st.subheader("Feature importance")
    imp = pd.Series(importances).sort_values()
    st.bar_chart(imp, horizontal=True)
    st.subheader("Model settings")
    st.json({k: v for k, v in model.get_params().items()
             if k in ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
                      "max_features", "class_weight", "ccp_alpha")})
    st.subheader("Training data ranges (used for capping)")
    st.dataframe(pd.DataFrame({"median": medians, "lower cap": lower, "upper cap": upper}).round(3))