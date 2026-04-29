# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import joblib
import os

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("data/raw/train.csv")

print("Dataset Shape:", df.shape)

# -----------------------------
# 2. SELECT IMPORTANT FEATURES ONLY
# -----------------------------
selected_features = [
    "GrLivArea",
    "OverallQual",
    "YearBuilt",
    "FullBath",
    "HalfBath"
]

df = df[selected_features + ["SalePrice"]]

# -----------------------------
# 3. HANDLE MISSING VALUES
# -----------------------------
df = df.dropna()

# -----------------------------
# 4. DEFINE FEATURES & TARGET
# -----------------------------
X = df[selected_features]
y = df["SalePrice"]

# -----------------------------
# 5. TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 6. MODEL TRAINING
# -----------------------------

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Random Forest (MAIN MODEL)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# -----------------------------
# 7. EVALUATION FUNCTION
# -----------------------------
def evaluate(model, X_test, y_test, name):
    pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    
    print(f"\n{name} Performance:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)
    
    return pred

# Evaluate models
lr_pred = evaluate(lr, X_test, y_test, "Linear Regression")
rf_pred = evaluate(rf, X_test, y_test, "Random Forest")

# -----------------------------
# 8. SAVE MODEL
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/house_price_model.pkl")

# -----------------------------
# 9. VISUALIZATION
# -----------------------------
os.makedirs("outputs", exist_ok=True)

# Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, rf_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.savefig("outputs/actual_vs_predicted.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.show()

# -----------------------------
# 10. SAMPLE PREDICTION
# -----------------------------
sample = X_test.iloc[0:1]
prediction = rf.predict(sample)

print("\nSample Prediction:")
print("Predicted Price:", prediction[0])