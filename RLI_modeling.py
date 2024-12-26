import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from joblib import dump, load

# Parameters
file_path = "Filtered_Left_Joined_Dataset.csv"
chunksize = 10000  # Adjust chunk size based on system memory
columns_to_drop = ['RenewFlag', 'CnclMo', 'NotCancFlag']
target_column = 'PostalCode.Count'

# Demographic columns used for modeling
demographic_columns = ["ESTIMATE HOUSEHOLDS TOTAL", "ESTIMATE HOUSEHOLDS - MEAN INCOME (DOLLARS)", "ESTIMATE MALE TOTAL POPULATION", "ESTIMATE FEMALE TOTAL POPULATION"]

# Load and preprocess data
X_full = pd.DataFrame()
y_full = pd.Series(dtype='float64')

for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
    chunk = chunk.drop(columns=columns_to_drop, errors='ignore')
    X_full = pd.concat([X_full, chunk[demographic_columns]], ignore_index=True)
    y_full = pd.concat([y_full, chunk[target_column]], ignore_index=True)

# Handle missing values
X_full.fillna(X_full.mean(), inplace=True)

# Log transform the target
y_full_log = np.log1p(y_full)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full_log, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
dump(scaler, 'scaler.joblib')

# Model training: Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_scaled, y_train)

rf_preds_log = rf_model.predict(X_test_scaled)
rf_preds = np.expm1(rf_preds_log)  # Inverse transform predictions

print("\nRandom Forest Performance:")
print(f"Mean Squared Error: {mean_squared_error(np.expm1(y_test), rf_preds)}")
print(f"Mean Absolute Error: {mean_absolute_error(np.expm1(y_test), rf_preds)}")
print(f"R^2 Score: {r2_score(np.expm1(y_test), rf_preds)}")

# Hyperparameter tuning: XGBoost
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

print("Best Parameters:", grid_search.best_params_)
best_xgb = grid_search.best_estimator_

xgb_preds_log = best_xgb.predict(X_test_scaled)
xgb_preds = np.expm1(xgb_preds_log)

print("\nXGBoost Performance:")
print(f"Mean Squared Error: {mean_squared_error(np.expm1(y_test), xgb_preds)}")
print(f"Mean Absolute Error: {mean_absolute_error(np.expm1(y_test), xgb_preds)}")
print(f"R^2 Score: {r2_score(np.expm1(y_test), xgb_preds)}")

# Cross-validation
cv_scores = cross_val_score(best_xgb, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"Cross-Validation R^2 Scores: {cv_scores}")
print(f"Mean Cross-Validation R^2: {cv_scores.mean()}")

# Feature importance visualization
xgb_importances = pd.DataFrame({
    'Feature': demographic_columns,
    'Importance': best_xgb.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot top features
plt.figure(figsize=(10, 6))
plt.barh(xgb_importances['Feature'], xgb_importances['Importance'])
plt.gca().invert_yaxis()
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Stacking model
stacking_model = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(random_state=42)),
        ('xgb', best_xgb)
    ],
    final_estimator=LinearRegression()
)

stacking_model.fit(X_train_scaled, y_train)
stack_preds_log = stacking_model.predict(X_test_scaled)
stack_preds = np.expm1(stack_preds_log)

print("\nStacking Model Performance:")
print(f"Mean Squared Error: {mean_squared_error(np.expm1(y_test), stack_preds)}")
print(f"Mean Absolute Error: {mean_absolute_error(np.expm1(y_test), stack_preds)}")
print(f"R^2 Score: {r2_score(np.expm1(y_test), stack_preds)}")

# Save models
dump(rf_model, 'random_forest_model.joblib')
dump(best_xgb, 'xgboost_model.joblib')
dump(stacking_model, 'stacking_model.joblib')
print("Models saved successfully!")

# Process new data for predictions
input_file = 'Merged_USCensus_Data.csv'
output_file = 'predictions.csv'
header_written = False

loaded_model = load('stacking_model.joblib')

for chunk in pd.read_csv(input_file, chunksize=chunksize):
    chunk.columns = chunk.columns.str.strip().str.upper()
    zip_codes = chunk['ZIP_CODE'] if 'ZIP_CODE' in chunk.columns else None

    data_to_scale = chunk[demographic_columns].reindex(columns=demographic_columns, fill_value=0)
    X_chunk_scaled = scaler.transform(data_to_scale)

    y_pred_chunk = loaded_model.predict(X_chunk_scaled)

    predictions = pd.DataFrame({
        'ZIP_CODE': zip_codes.values if zip_codes is not None else [None] * len(y_pred_chunk),
        'PREDICTED_TARGET': np.expm1(y_pred_chunk)
    })

    predictions.to_csv(output_file, mode='a', index=False, header=not header_written)
    header_written = True

print("Predictions saved successfully!")
