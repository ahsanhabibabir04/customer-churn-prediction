
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path("models/logistic_regression.joblib")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered",
)

st.title("Customer Churn Prediction")
st.write(
    "Predict whether a telecom customer is likely to leave "
    "the service using a machine-learning model."
)

st.info(
    "This is an educational portfolio project using a public "
    "sample dataset. Predictions are estimates, not guarantees."
)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run: python -m src.train_model")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.header("Customer Information")

with st.form("customer_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen?", [0, 1])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    if phone_service == "Yes":
        multiple_lines = st.selectbox(
            "Multiple Lines", ["Yes", "No"]
        )
    else:
        multiple_lines = "No phone service"
        st.caption("Multiple Lines: No phone service")

    internet_service = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )

    if internet_service == "No":
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"
        st.caption("Internet add-on services: No internet service")
    else:
        online_security = st.selectbox(
            "Online Security", ["Yes", "No"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No"]
        )

    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0,
    )
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=float(monthly_charges * tenure),
        step=10.0,
    )

    submitted = st.form_submit_button("Predict Churn")


if submitted:
    customer = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ]
    )

    prediction = int(model.predict(customer)[0])
    probability = float(model.predict_proba(customer)[0][1])

    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning("The model predicts that this customer may churn.")
    else:
        st.success("The model predicts that this customer may stay.")

    st.metric("Estimated Churn Probability", f"{probability:.1%}")
    st.progress(probability)

    st.caption(
        "This probability is a model estimate, not a guarantee "
        "of the customer's future behavior."
    )
