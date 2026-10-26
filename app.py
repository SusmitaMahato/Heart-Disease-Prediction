from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.predict import predict_risk

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "heart.csv"

st.set_page_config(
    page_title="HeartWise | Heart Health",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #f7f9fc; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
.hero {
    padding: 24px; border-radius: 18px;
    background: linear-gradient(120deg, #7f1d35, #dc5266);
    color: white; margin-bottom: 20px;
}
.hero h1 { color: white; }
.small-note { color: #606b7a; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>❤️ HeartWise</h1>
    <p>Your heart health risk prediction prototype</p>
    <p>Explore health indicators and an educational ML prediction.</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Risk Prediction", "Data Explorer", "About"]
)


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


if page == "Home":
    st.subheader("Welcome to HeartWise")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### What can you do?")
        st.write(
            "- Enter health indicators.\n"
            "- Explore the sample dataset.\n"
            "- View the model's estimated probability.\n"
            "- Understand the limitations of ML predictions."
        )

    with col2:
        st.markdown("### How it works")
        st.write(
            "Dataset → Data cleaning → Model training → "
            "Evaluation → Prediction"
        )

    st.info(
        "This is an educational prototype. It cannot diagnose "
        "heart disease or replace professional medical advice."
    )

elif page == "Risk Prediction":
    st.subheader("Heart Disease Risk Prediction")
    st.write("Enter the requested values below.")

    with st.form("prediction_form"):
        st.markdown("#### Basic information")
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input(
                "Age (years)", min_value=18, max_value=100, value=45
            )
        with c2:
            sex = st.selectbox(
                "Sex (dataset coding)",
                options=[1, 0],
                format_func=lambda x: "Male (1)" if x == 1 else "Female (0)"
            )
        with c3:
            cp = st.selectbox(
                "Chest pain type",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Typical angina", 2: "Atypical angina",
                    3: "Non-anginal pain", 4: "Asymptomatic"
                }[x]
            )

        st.markdown("#### Clinical measurements")
        c1, c2, c3 = st.columns(3)

        with c1:
            trestbps = st.number_input(
                "Resting blood pressure (mm Hg)",
                min_value=70, max_value=250, value=120
            )
        with c2:
            chol = st.number_input(
                "Serum cholesterol (mg/dL)",
                min_value=80, max_value=700, value=200
            )
        with c3:
            fbs = st.selectbox(
                "Fasting blood sugar > 120 mg/dL",
                options=[0, 1],
                format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            restecg = st.selectbox("Resting ECG code", [0, 1, 2])
        with c2:
            thalach = st.number_input(
                "Maximum heart rate achieved",
                min_value=50, max_value=250, value=150
            )
        with c3:
            exang = st.selectbox(
                "Exercise-induced angina",
                options=[0, 1],
                format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
            )

        st.markdown("#### Additional measurements")
        c1, c2, c3 = st.columns(3)
        with c1:
            oldpeak = st.number_input(
                "ST depression (oldpeak)",
                min_value=0.0, max_value=10.0, value=1.0, step=0.1
            )
        with c2:
            slope = st.selectbox("ST segment slope code", [1, 2, 3])
        with c3:
            ca = st.selectbox("Major vessels (ca)", [0, 1, 2, 3])

        thal = st.selectbox(
            "Thalassemia code (thal)",
            options=[3, 6, 7],
            format_func=lambda x: {
                3: "Normal (3)", 6: "Fixed defect (6)",
                7: "Reversible defect (7)"
            }[x]
        )

        submitted = st.form_submit_button(
            "Analyze health indicators",
            type="primary",
            use_container_width=True
        )

    if submitted:
        input_data = {
            "age": age, "sex": sex, "cp": cp,
            "trestbps": trestbps, "chol": chol, "fbs": fbs,
            "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak,
            "slope": slope, "ca": ca, "thal": thal
        }

        try:
            result = predict_risk(input_data)
            probability = result["probability"]

            st.markdown("---")
            st.subheader("Model output")
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Estimated positive-class probability",
                    f"{probability:.1%}"
                )
            with col2:
                st.metric(
                    "Model classification",
                    "Positive class" if result["prediction"] == 1
                    else "Negative class"
                )

            st.progress(probability)
            st.warning(
                "This probability is not your personal clinical risk "
                "and is not a diagnosis. The model uses a small historical "
                "dataset and may be inaccurate. Consult a qualified "
                "healthcare professional for medical assessment."
            )

        except FileNotFoundError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Prediction failed: {error}")

elif page == "Data Explorer":
    st.subheader("Heart Disease Dataset Explorer")
    df = load_dataset()

    if df is None:
        st.warning(
            "Dataset not found. Run "
            "`python -m src.download_data` first."
        )
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Records", len(df))
        col2.metric("Features", len(df.columns) - 1)
        col3.metric("Positive-class records", int(df["target"].sum()))

        st.markdown("#### Dataset preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("#### Target distribution")
        counts = (
            df["target"]
            .map({0: "Negative class", 1: "Positive class"})
            .value_counts()
            .rename_axis("Class")
            .reset_index(name="Count")
        )
        fig = px.bar(
            counts, x="Class", y="Count", color="Class",
            title="Distribution of target classes"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Age distribution")
        fig_age = px.histogram(
            df, x="age", color="target", nbins=20,
            title="Age distribution by target class",
            labels={"target": "Dataset target"}
        )
        st.plotly_chart(fig_age, use_container_width=True)

elif page == "About":
    st.subheader("About HeartWise")
    st.write(
        "HeartWise is a student machine-learning project demonstrating "
        "data preprocessing, model training, evaluation, and interactive prediction."
    )
    st.markdown("#### Dataset")
    st.write(
        "UCI Cleveland Heart Disease dataset. The target is simplified to "
        "0 (original target equals 0) and 1 (original target is greater than 0)."
    )
    st.markdown("#### Limitations")
    st.write(
        "- The dataset is small and historical.\n"
        "- Results may not generalize to other populations.\n"
        "- Model probabilities are not calibrated clinical risks.\n"
        "- This prototype is not intended for clinical decisions."
    )

st.markdown("---")
st.markdown(
    '<p class="small-note">HeartWise is an educational project, '
    'not a medical diagnostic service.</p>',
    unsafe_allow_html=True
)
