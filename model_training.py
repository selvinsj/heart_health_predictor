import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

# Load the dataset
df = pd.read_csv("heart_data_my.csv")

# Prepare features and target
X = df.drop(columns=['hearthealth_risk_percentage', 'suggestions'])  # Features
y = df['hearthealth_risk_percentage']  # Target

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Initialize XGBoost regressor
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}%")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Save model
joblib.dump(model, "model/xgboost_heart_model.pkl")
print("\nModel saved to model/xgboost_heart_model.pkl")

# Feature importance
importance = model.feature_importances_
print("\nFeature Importance:")
for name, score in zip(X.columns, importance):
    print(f"{name}: {score:.3f}")  # Fixed incorrect formatting
