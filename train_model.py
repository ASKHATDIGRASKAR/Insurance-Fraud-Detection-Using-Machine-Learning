import os
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Create model folder
os.makedirs("model", exist_ok=True)

# Load dataset
data = pd.read_csv("dataset/insurance_claims.csv")

# Encode categorical columns
le = LabelEncoder()
data["Sex"] = le.fit_transform(data["Sex"])
data["AccidentArea"] = le.fit_transform(data["AccidentArea"])
data["Fault"] = le.fit_transform(data["Fault"])

# Features and target
X = data.drop("Fraud", axis=1)
y = data["Fraud"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Random Forest Model
# ----------------------------
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

joblib.dump(rf_model, "model/fraud_model.pkl")

# ----------------------------
# Anomaly Detection Model
# ----------------------------
anomaly_model = IsolationForest(contamination=0.1)
anomaly_model.fit(X)

joblib.dump(anomaly_model, "model/anomaly_model.pkl")

# ----------------------------
# Deep Learning Model
# ----------------------------
dl_model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

dl_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

dl_model.fit(X_train, y_train, epochs=20, batch_size=8)

dl_model.save("model/deep_fraud_model.h5")

print("✅ All models trained and saved!")