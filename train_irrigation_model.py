# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("irrigation_dataset.csv")

print("Dataset loaded:", df.shape)

# -----------------------------
# Features
# -----------------------------
X = df.drop("irrigation_mm", axis=1)
y = df["irrigation_mm"]

# -----------------------------
# Categorical Columns
# -----------------------------
categorical_cols = ["soil", "crop"]

# -----------------------------
# Numeric Columns
# -----------------------------
numeric_cols = [
    "infiltration",
    "temperature",
    "humidity",
    "rainfall",
    "ndvi"
]

# -----------------------------
# Preprocessing
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

# -----------------------------
# Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
pipeline.fit(X_train, y_train)

# -----------------------------
# Accuracy
# -----------------------------
score = pipeline.score(X_test, y_test)

print("Model R² score:", score)

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(pipeline, "irrigation_model.pkl")

print("Model saved → irrigation_model.pkl")
