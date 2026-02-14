# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your dataset (CSV you prepare from farmer inputs)
df = pd.read_csv("ml/yield_dataset.csv")
print("Columns found in dataset:", df.columns.tolist())


X = df.drop("yield", axis=1)
y = df["yield"]

cat_cols = ["soil_type", "fertilizer_type", "crop_stage", "stress_level"]

num_cols = [
    "fertilizer_kg",
    "irrigation_count",
    "pesticide_sprays",
    "avg_temp",
    "rainfall",
    "humidity",
    "wind_speed",
    "ndvi"
]



preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
)

pipe = Pipeline([
    ("prep", preprocess),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipe.fit(X_train, y_train)

joblib.dump(pipe, "models/yield_model.joblib")

print("✅ Yield model trained & saved")
