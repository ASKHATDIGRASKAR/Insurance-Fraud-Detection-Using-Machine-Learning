import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import tensorflow as tf
FEATURES = [
    "Age",
    "Sex",
    "VehiclePrice",
    "AccidentArea",
    "Fault",
    "PastClaims"
]

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Load models
model = joblib.load("model/fraud_model.pkl")
anomaly = joblib.load("model/anomaly_model.pkl")

st.title("🚗 Insurance Fraud Detection AI Dashboard")

st.markdown(
"""
<style>
.stApp{
background: linear-gradient(135deg,#1f1c2c,#928dab);
color:white;
}
</style>
""",
unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("Claim Input")

age = st.sidebar.slider("Age",18,80)
sex = st.sidebar.selectbox("Sex",["Male","Female"])
vehicle_price = st.sidebar.number_input("Vehicle Price")
accident_area = st.sidebar.selectbox("Accident Area",["Urban","Rural"])
fault = st.sidebar.selectbox("Fault",["Policy Holder","Third Party"])
past_claims = st.sidebar.slider("Past Claims",0,5)

sex = 1 if sex=="Male" else 0
accident_area = 1 if accident_area=="Urban" else 0
fault = 1 if fault=="Policy Holder" else 0

features = np.array([[age,sex,vehicle_price,accident_area,fault,past_claims]])

col1,col2,col3 = st.columns(3)

col1.metric("Age",age)
col2.metric("Past Claims",past_claims)
col3.metric("Vehicle Price",vehicle_price)

st.divider()

if st.button("Analyze Claim"):

    pred = model.predict(features)
    prob = model.predict_proba(features)[0][1]*100
    anomaly_score = anomaly.predict(features)

    if pred[0]==1:
        st.error(f"⚠ Fraud Detected | Probability: {prob:.2f}%")
    else:
        st.success(f"✅ Genuine Claim | Fraud Probability: {prob:.2f}%")

    if anomaly_score[0]==-1:
        st.warning("🧠 Anomaly Detection: Suspicious Pattern")

    st.progress(int(prob))

# Upload CSV
st.header("📁 Bulk Fraud Detection")

file = st.file_uploader("Upload Claims CSV")

if file:

    df = pd.read_csv(file)

    st.write("Uploaded Dataset Preview")
    st.dataframe(df.head())

    # Check missing columns
    missing_cols = [col for col in FEATURES if col not in df.columns]

    if missing_cols:

        st.error("Dataset not compatible with the model")

        st.write("Missing columns:")
        st.write(missing_cols)

        st.info("Required dataset format:")

        example_df = pd.DataFrame(columns=FEATURES)

        st.dataframe(example_df)

        st.stop()

    # Keep only required features
    df = df[FEATURES]

    # Encode categorical variables
    df["Sex"] = df["Sex"].map({"Male":1,"Female":0})
    df["AccidentArea"] = df["AccidentArea"].map({"Urban":1,"Rural":0})
    df["Fault"] = df["Fault"].map({"Policy Holder":1,"Third Party":0})

    # Prediction
    preds = model.predict(df)
    probs = model.predict_proba(df)[:,1]

    df["Fraud Prediction"] = preds
    df["Fraud Probability"] = probs

    st.subheader("Fraud Detection Results")
    st.dataframe(df)

if file:

    df = pd.read_csv(file)

    # Required columns
    required_cols = [
        "Age",
        "Sex",
        "VehiclePrice",
        "AccidentArea",
        "Fault",
        "PastClaims"
    ]

    # Keep only required columns
    df = df[required_cols]

    # Encode categorical columns
    df["Sex"] = df["Sex"].map({"Male":1,"Female":0})
    df["AccidentArea"] = df["AccidentArea"].map({"Urban":1,"Rural":0})
    df["Fault"] = df["Fault"].map({"Policy Holder":1,"Third Party":0})

    preds = model.predict(df)
    probs = model.predict_proba(df)[:,1]

    df["Fraud Prediction"] = preds
    df["Fraud Probability"] = probs

    st.dataframe(df)
# Explainable AI
st.header("🤖 Model Explanation")

importance = model.feature_importances_

features_names = [
"Age",
"Sex",
"VehiclePrice",
"AccidentArea",
"Fault",
"PastClaims"
]

imp_df = pd.DataFrame({
"Feature":features_names,
"Importance":importance
})

fig = px.bar(imp_df,x="Feature",y="Importance")
st.plotly_chart(fig,use_container_width=True)

import plotly.graph_objects as go

st.header("🔥 Fraud Heatmap")

if file:

    heatmap_data = df.groupby(["AccidentArea"])["Fraud Probability"].mean().reset_index()

    fig = px.density_heatmap(
        heatmap_data,
        x="AccidentArea",
        y="Fraud Probability",
        color_continuous_scale="reds"
    )

    st.plotly_chart(fig, use_container_width=True)
    import tensorflow as tf

dl_model = tf.keras.models.load_model("model/deep_fraud_model.h5")

dl_prob = dl_model.predict(features)[0][0] * 100

st.info(f"🧠 Deep Learning Fraud Score: {dl_prob:.2f}%")
st.header("📊 Fraud Analytics")

fig = px.pie(
    df,
    names="Fraud Prediction",
    title="Fraud vs Genuine Claims"
)

st.plotly_chart(fig, use_container_width=True)

fig = px.histogram(
    df,
    x="Age",
    color="Fraud Prediction",
    title="Fraud Distribution by Age"
)

st.plotly_chart(fig, use_container_width=True)

fig = px.scatter(
    df,
    x="VehiclePrice",
    y="Fraud Probability",
    color="Fraud Prediction",
    title="Vehicle Price vs Fraud Risk"
)

st.plotly_chart(fig, use_container_width=True)