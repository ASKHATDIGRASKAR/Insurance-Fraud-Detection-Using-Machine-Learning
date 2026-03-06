# 🚗 Insurance Fraud Detection Using Machine Learning

An AI-powered dashboard that detects fraudulent insurance claims using **Machine Learning, Deep Learning, and Anomaly Detection**.

This project demonstrates how data science can help insurance companies identify suspicious claims and reduce fraud losses.

---

# 📌 Project Overview

Insurance fraud costs companies billions every year. Detecting fraud manually is slow and inefficient.

This project builds an **AI fraud detection system** that:

- Analyzes insurance claim data
- Predicts if a claim is fraudulent
- Detects anomalies in claim behavior
- Visualizes fraud analytics in a dashboard

The application includes a **Streamlit dashboard** where users can input claim data or upload a dataset to detect fraud automatically.

---

# ⚙️ Technologies Used

- Python
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Plotly
- Joblib

---

# 🧠 Machine Learning Models

This project uses multiple AI techniques:

### Random Forest Classifier
Used for the main fraud detection model.

### Deep Learning Model
A neural network that learns patterns in claim data.

### Isolation Forest
Detects anomalies and suspicious patterns in claims.

---

# 📂 Project Structure


insurance-fraud-detection
│
├── dataset/
│ └── insurance_claims.csv
│
├── model/
│ ├── fraud_model.pkl
│ ├── anomaly_model.pkl
│ └── deep_fraud_model.h5
│
├── train_model.py
├── app.py
├── requirements.txt
└── README.md


---

# 📊 Features

✔ Fraud claim prediction  
✔ Deep learning fraud scoring  
✔ Anomaly detection  
✔ Interactive dashboard  
✔ CSV batch fraud detection  
✔ Fraud analytics charts  
✔ Dataset validation  
✔ Download sample dataset

---

# 📁 Dataset Format

The model expects the following columns:


Age, Sex, VehiclePrice, AccidentArea, Fault, PastClaims


Example:


Age,Sex,VehiclePrice,AccidentArea,Fault,PastClaims
25,Male,20000,Urban,Policy Holder,1
40,Female,15000,Rural,Third Party,0
30,Male,22000,Urban,Policy Holder,2


---

# 🚀 Installation

Clone the repository:


git clone 


Navigate to the project folder:


cd insurance-fraud-detection


Install dependencies:


pip install -r requirements.txt


---

# 🏋️ Train the Model

Run the training script:


python train_model.py


This will generate the trained models inside the **model/** folder.

---

# ▶ Run the Dashboard

Start the Streamlit application:


streamlit run app.py


Open the dashboard in your browser:


http://localhost:8501


---

# 📊 Dashboard Preview

The dashboard allows users to:

- Enter claim information
- Predict fraud probability
- Upload claim datasets
- Detect high-risk claims
- Visualize fraud analytics

---

# 🔮 Future Improvements

- SHAP explainable AI
- Fraud risk gauge visualization
- Real-time fraud detection API
- Larger fraud dataset training
- Deployment using Docker or cloud services

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository  
2. Create a new branch  
3. Submit a pull request  

---

# 📜 License

This project is licensed under the MIT License.

