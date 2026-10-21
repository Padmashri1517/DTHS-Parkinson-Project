import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from model import FuzzyKNN, prepare_dataset, evaluate_models, FEATURE_COLUMNS, TARGET_COLUMN

st.set_page_config(page_title="DTHS | Parkinson Screening Prototype", page_icon="🩺", layout="wide")
st.title("Digital Twin-Based Healthcare System (DTHS)")
st.caption("Educational prototype • Voice-feature-based Parkinson's screening support")

st.warning(
    "Research/academic demonstration only. This tool does not diagnose Parkinson's disease "
    "and must not be used to make medical decisions. Consult a qualified clinician."
)

with st.sidebar:
    st.header("Project navigation")
    page = st.radio("Choose a page", ["Overview", "Train & Evaluate", "Patient Twin Demo"])
    st.markdown("---")
    st.caption("Dataset expected: UCI Parkinson's voice-feature CSV (for example, parkinsons.data).")

def load_uploaded_data():
    uploaded = st.file_uploader("Upload a Parkinson's voice-feature CSV", type=["csv"])
    if uploaded is not None:
        try:
            return pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
    return None

if page == "Overview":
    st.subheader("System workflow")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Layer 1", "End User", "Patient / clinician")
    c2.metric("Layer 2", "ML Cloud", "Feature processing")
    c3.metric("Classifier", "Fuzzy k-NN", "Membership-based")
    c4.metric("Layer 3", "Care support", "Non-diagnostic guidance")
    st.markdown("""
    **What this prototype demonstrates**
    1. Load a public dataset containing already-extracted voice measurements.
    2. Prepare the features and train a fuzzy k-nearest-neighbour classifier.
    3. Compare its performance with standard KNN and SVM baselines.
    4. Display a demo patient profile, model output, and a local session history.

    **Important limitation:** The classic UCI `parkinsons.data` file contains extracted numerical
    features, not raw audio. This prototype does not turn a microphone recording into those
    features. Raw-audio feature extraction is a separate, more advanced module.
    """)
    st.info("Start with the Train & Evaluate page. Upload your CSV there to check the column names and run the experiment.")

elif page == "Train & Evaluate":
    st.subheader("1. Upload dataset")
    df = load_uploaded_data()
    if df is not None:
        st.write("Preview", df.head())
        st.write("Shape:", df.shape)
        st.write("Columns:", ", ".join(map(str, df.columns)))
        try:
            X, y, feature_cols = prepare_dataset(df)
            st.success(f"Dataset prepared: {len(X)} rows, {len(feature_cols)} usable features.")
            with st.spinner("Training and evaluating models..."):
                results, model, scaler, X_clean, y_clean = evaluate_models(X, y)
            st.subheader("2. Results")
            st.dataframe(results, use_container_width=True)
            st.bar_chart(results.set_index("Model")[["Accuracy", "F1-score", "MCC"]])
            st.caption("Metrics are calculated on a held-out test split. Results may differ from the paper due to dataset/split and implementation differences.")
            st.session_state["trained_model"] = model
            st.session_state["scaler"] = scaler
            st.session_state["feature_cols"] = feature_cols
            st.session_state["data_X"] = X_clean
            st.session_state["data_y"] = y_clean
            st.session_state["last_results"] = results
            st.success("Model saved for this browser session. Open Patient Twin Demo.")
        except Exception as exc:
            st.error(f"Dataset preparation/training failed: {exc}")
            st.markdown("Check that the CSV has a target column named `status` (0/1) and numerical voice features.")
    else:
        st.markdown("""
        **Where to get a starter dataset**
        - Search for **UCI Parkinson's Disease Data Set**.
        - Download the file commonly named `parkinsons.data`.
        - Upload it here. Do not upload any real patient's identifiable health data.
        """)
        st.code("Typical target column: status\nTypical identifier column: name (removed from features)")

elif page == "Patient Twin Demo":
    st.subheader("Patient digital-twin demonstration")
    if "trained_model" not in st.session_state:
        st.info("Train a model first in the Train & Evaluate page, then return here.")
    else:
        model = st.session_state["trained_model"]
        scaler = st.session_state["scaler"]
        feature_cols = st.session_state["feature_cols"]
        X_ref = st.session_state["data_X"]
        y_ref = st.session_state["data_y"]
        st.write("Enter values from one row of your dataset, or use the demo sample below.")
        row_idx = st.number_input("Demo row index", min_value=0, max_value=max(0, len(X_ref)-1), value=0, step=1)
        sample = X_ref.iloc[int(row_idx)].copy()
        with st.expander("Edit voice-feature values", expanded=True):
            values = {}
            cols = st.columns(3)
            for i, feature in enumerate(feature_cols):
                with cols[i % 3]:
                    values[feature] = st.number_input(
                        feature, value=float(sample[feature]), format="%.6f", key=f"f_{feature}"
                    )
        if st.button("Run demo prediction", type="primary"):
            new_x = pd.DataFrame([[values[f] for f in feature_cols]], columns=feature_cols)
            scaled = scaler.transform(new_x)
            probabilities = model.predict_proba(scaled)[0]
            pred = int(np.argmax(probabilities))
            label = "Parkinson's-associated class (status=1)" if pred == 1 else "Non-Parkinson's-associated class (status=0)"
            st.subheader("Model output")
            st.metric("Predicted dataset class", str(pred))
            st.write(label)
            st.write("Class membership scores:", {str(i): round(float(p), 4) for i, p in enumerate(probabilities)})
            st.caption("The meaning of status labels depends on the dataset documentation. This output is not a medical diagnosis.")
            record = {
                "Demo row": int(row_idx),
                "Predicted class": pred,
                "Class 0 membership": round(float(probabilities[0]), 4),
                "Class 1 membership": round(float(probabilities[1]), 4),
            }
            history = st.session_state.get("history", [])
            history.append(record)
            st.session_state["history"] = history
        if st.session_state.get("history"):
            st.subheader("Session prediction history")
            st.dataframe(pd.DataFrame(st.session_state["history"]), use_container_width=True)
            st.caption("History is kept only in the current Streamlit session; it is not saved to a hospital database.")
