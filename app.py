
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI-Based Diabetes Risk Screening",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("diabetes_screening_xgb_pipeline.pkl")
    features = joblib.load("screening_features.pkl")
    return model, features


xgb_model, screening_features = load_model()

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🩺 AI-Based Diabetes Risk Screening")

st.markdown(
    """
    ### Explainable Clinical Decision Support Prototype

    This research prototype uses machine learning to estimate
    diabetes screening risk from non-laboratory health and
    lifestyle indicators.
    """
)

st.info(
    "Research Prototype | Model: XGBoost | Explainability: SHAP"
)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("System Information")

    st.write("**Application:** Diabetes Risk Screening")
    st.write("**Model:** XGBoost")
    st.write("**Explainability:** SHAP")
    st.write("**Mode:** Screening")
    st.write("**Status:** Research Prototype")

    st.divider()

    st.caption(
        "Laboratory biomarkers such as HbA1c and fasting glucose "
        "are reserved for the subsequent clinical decision-support "
        "stage and are not used in this screening model."
    )

# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------

st.header("Patient Information")

col1, col2, col3 = st.columns(3)

with col1:

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )

    waist = st.number_input(
        "Waist Circumference (cm)",
        min_value=40.0,
        max_value=180.0,
        value=85.0,
        step=1.0
    )

    systolic = st.number_input(
        "Systolic Blood Pressure (mmHg)",
        min_value=70.0,
        max_value=250.0,
        value=120.0,
        step=1.0
    )

    diastolic = st.number_input(
        "Diastolic Blood Pressure (mmHg)",
        min_value=40.0,
        max_value=150.0,
        value=80.0,
        step=1.0
    )


with col2:

    family_history = st.selectbox(
        "Family History of Diabetes",
        ["No", "Yes"]
    )

    hypertension = st.selectbox(
        "Hypertension",
        ["No", "Yes"]
    )

    physical_activity = st.selectbox(
        "Physical Activity",
        ["Low", "Moderate", "High"]
    )

    smoking = st.selectbox(
        "Smoking",
        ["No", "Yes"]
    )


with col3:

    alcohol = st.selectbox(
        "Alcohol Consumption",
        ["No", "Yes"]
    )

    obesity = st.selectbox(
        "Obesity",
        ["No", "Yes"]
    )

    pcos = st.selectbox(
        "PCOS",
        ["No", "Yes"]
    )

    gestational_diabetes = st.selectbox(
        "Gestational Diabetes",
        ["No", "Yes"]
    )

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

st.divider()

predict_button = st.button(
    "🔍 Predict Diabetes Risk",
    type="primary",
    use_container_width=True
)

if predict_button:

    # Create input dataframe
    input_data = pd.DataFrame([{
        "BMI": bmi,
        "Waist_Circumference": waist,
        "Blood_Pressure_Systolic": systolic,
        "Blood_Pressure_Diastolic": diastolic,
        "Family_History_of_Diabetes": family_history,
        "Hypertension": hypertension,
        "Physical_Activity": physical_activity,
        "Smoking": smoking,
        "Alcohol_Consumption": alcohol,
        "Obesity": obesity,
        "PCOS": pcos,
        "Gestational_Diabetes": gestational_diabetes
    }])

    # Ensure exact feature order
    input_data = input_data[screening_features]

    # Prediction
    prediction = xgb_model.predict(input_data)[0]

    probability = xgb_model.predict_proba(input_data)[0][1]

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    st.header("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:

            st.error("⚠️ Screening Result: Positive")

        else:

            st.success("✅ Screening Result: Negative")

    with result_col2:

        st.metric(
            "Model-Estimated Probability",
            f"{probability * 100:.2f}%"
        )

    # -----------------------------------------------------
    # RISK STRATIFICATION
    # -----------------------------------------------------

    if probability < 0.30:

    risk_level = "Low"

elif probability < 0.85:

    risk_level = "Moderate"

else:

    risk_level = "High"

    st.subheader("Risk Stratification")

    if risk_level == "Low":

        st.success("🟢 Low model-estimated screening risk")

    elif risk_level == "Moderate":

        st.warning("🟡 Moderate model-estimated screening risk")

    else:

        st.error("🔴 High model-estimated screening risk")

    # -----------------------------------------------------
    # NEXT STEP
    # -----------------------------------------------------

    st.subheader("Suggested Next Step")

    if prediction == 1:

        st.write(
            "The screening model indicates an elevated likelihood "
            "of diabetes. Further clinical evaluation and appropriate "
            "laboratory testing should be considered."
        )

    else:

        st.write(
            "The screening model does not indicate elevated diabetes "
            "risk based on the entered screening features. Routine "
            "health monitoring and healthy lifestyle practices are "
            "still recommended."
        )

    # -----------------------------------------------------
    # SHAP EXPLANATION
    # -----------------------------------------------------

    st.divider()

    st.header("🔎 Why did the model make this prediction?")

    st.write(
        "SHAP (SHapley Additive exPlanations) is used to examine "
        "how individual input features contributed to the model's "
        "prediction."
    )

    try:

        preprocessor = xgb_model.named_steps["preprocessor"]
        estimator = xgb_model.named_steps["model"]

        transformed_input = preprocessor.transform(input_data)

        if hasattr(transformed_input, "toarray"):
            transformed_input = transformed_input.toarray()

        feature_names = preprocessor.get_feature_names_out()

        input_shap = pd.DataFrame(
            transformed_input,
            columns=feature_names
        )

        explainer = shap.TreeExplainer(estimator)

        shap_values = explainer.shap_values(input_shap)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):

            shap_values_plot = shap_values[1]

        else:

            shap_values_plot = shap_values

            if len(np.shape(shap_values_plot)) == 3:
                shap_values_plot = shap_values_plot[:, :, 1]

        fig, ax = plt.subplots(figsize=(10, 5))

        shap.summary_plot(
            shap_values_plot,
            input_shap,
            plot_type="bar",
            show=False
        )

        plt.tight_layout()

        st.pyplot(fig)

        st.caption(
            "Higher absolute SHAP values indicate a stronger "
            "contribution to the model prediction. SHAP values "
            "describe model behavior and should not be interpreted "
            "as causal medical effects."
        )

    except Exception as e:

        st.warning(
            "SHAP explanation could not be generated for this "
            "prediction."
        )

# ---------------------------------------------------------
# DISCLAIMER
# ---------------------------------------------------------

st.divider()

st.caption(
    "⚠️ This application is a research prototype and is not "
    "a substitute for professional medical diagnosis. Model "
    "outputs should not be used as the sole basis for medical "
    "decisions."
)
