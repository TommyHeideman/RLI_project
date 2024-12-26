<!DOCTYPE html>
<html>
<head>
    <title>Policy Growth Opportunity Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            color: #333;
        }
        h1, h2, h3 {
            color: #0056b3;
        }
        ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        code {
            background: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
        }
        .impact {
            background: #e7f5ff;
            padding: 10px;
            border-left: 4px solid #0056b3;
            margin: 10px 0;
        }
        img {
            max-width: 100%;
            height: auto;
            margin: 10px 0;
        }
    </style>
</head>
<body>

<h1>Policy Growth Opportunity Analysis</h1>

<div class="impact">
    <h2>Why This Project Matters</h2>
    <p>This project identifies untapped opportunities for policy growth by analyzing RLI insurance data alongside U.S. Census demographics. Using predictive modeling, we pinpointed ZIP codes with the highest potential for new policies based on income and demographic factors.</p>
</div>

<h2>Project Overview</h2>
<p>The goal was to count existing policies in each ZIP code and leverage Census data (income and sex demographics) to train a machine learning model predicting areas of high growth potential. The results were visualized for actionable insights.</p>
<ul>
    <li><strong>Impact:</strong> Empowered stakeholders to optimize policy distribution strategies and target high-opportunity regions.</li>
    <li><strong>Technologies Used:</strong> Python, R, Power BI, XGBoost, Random Forest, Stacking Regressor.</li>
</ul>

<h2>Key Steps</h2>
<ol>
    <li>Cleaned and merged insurance and demographic datasets.</li>
    <li>Trained predictive models to forecast policy counts in unexplored regions.</li>
    <li>Enhanced predictions by integrating geographic data (state and city).</li>
    <li>Created interactive Power BI dashboards for stakeholder use.</li>
</ol>

<hr>

<h2>1. Data Cleaning</h2>
<ul>
    <li><strong>Standardized:</strong> Reformatted ZIP codes for consistency.</li>
    <li><strong>Joined:</strong> Merged RLI insurance data with U.S. Census demographics by ZIP code.</li>
    <li><strong>Output:</strong> A consolidated dataset for modeling and analysis.</li>
</ul>

<h2>2. Machine Learning</h2>
<ul>
    <li><strong>Models Used:</strong>
        <ul>
            <li>Random Forest: Baseline performance benchmark.</li>
            <li>XGBoost: Fine-tuned for high predictive accuracy.</li>
            <li>Stacking Regressor: Combined models for superior results.</li>
        </ul>
    </li>
    <li><strong>Key Metrics:</strong>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>MSE</th>
                    <th>MAE</th>
                    <th>R² Score</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Random Forest</td>
                    <td>2118.23</td>
                    <td>18.33</td>
                    <td>0.694</td>
                </tr>
                <tr>
                    <td>XGBoost</td>
                    <td>1413.67</td>
                    <td>15.04</td>
                    <td>0.796</td>
                </tr>
                <tr>
                    <td>Stacking Regressor</td>
                    <td>1376.02</td>
                    <td>14.94</td>
                    <td>0.801</td>
                </tr>
            </tbody>
        </table>
    </li>
</ul>

<h3>Python Modeling Code</h3>
<pre><code>
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
file_path = r"Filtered_Left_Joined_Dataset.csv"
chunksize = 10000
columns_to_drop = ['RenewFlag', 'CnclMo', 'NotCancFlag']
demographic_columns = [...]
# (list truncated for brevity, include full list in project repository)

# Load and preprocess data
X_full = pd.DataFrame()
y_full = pd.Series(dtype='float64')

for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
    chunk = chunk.drop(columns=columns_to_drop, errors='ignore')
    X_full = pd.concat([X_full, chunk[demographic_columns]], ignore_index=True)
    y_full = pd.concat([y_full, chunk['PostalCode.Count']], ignore_index=True)

# Handle missing values
X_full.fillna(X_full.mean(), inplace=True)

# Log transform the target
y_full_log = np.log1p(y_full)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full_log, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training: Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_scaled, y_train)

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

best_xgb = grid_search.best_estimator_

# Stacking model
stacking_model = StackingRegressor(
    estimators=[
        ('rf', rf_model),
        ('xgb', best_xgb)
    ],
    final_estimator=LinearRegression()
)
stacking_model.fit(X_train_scaled, y_train)

# Save models
dump(rf_model, 'random_forest_model.joblib')
dump(best_xgb, 'xgboost_model.joblib')
dump(stacking_model, 'stacking_model.joblib')
</code></pre>

<h2>3. Visualization and Insights</h2>
<p>Predictions were visualized in an interactive Power BI dashboard, showcasing ZIP codes with high growth potential. The visualization enabled easy exploration by geographic and demographic filters.</p>
<img src="https://github.com/TommyHeideman/<RLI_project>/raw/main/PowerBI_map.png" 
     alt="Power BI map showing model predictions and comparison to RLI policy counts">

<h2>4. Tools and Technologies</h2>
<ul>
    <li><strong>Python:</strong> Data cleaning, feature engineering, and machine learning with libraries like Pandas, Scikit-learn, and XGBoost.</li>
    <li><strong>R:</strong> Data merging and formatting with libraries such as <code>dplyr</code> and <code>zipcodeR</code>.</li>
    <li><strong>Power BI:</strong> Created interactive dashboards for data-driven decision-making.</li>
</ul>

<h2>5. Key Deliverables</h2>
<ul>
    <li><strong>Cleaned Datasets:</strong> Prepared for further analysis or integration.</li>
    <li><strong>Predictive Models:</strong> Saved in joblib format for reproducibility.</li>
    <li><strong>Interactive Dashboard:</strong> Delivered actionable insights to stakeholders.</li>
    <li><strong>Final Report:</strong> For more details, refer to the <a href="final_report_document_link">Final Project Report</a>.</li>
</ul>

<hr>

<h2>How to Run</h2>
<ol>
    <li>Install dependencies:
        <pre><code>pip install pandas numpy scikit-learn xgboost matplotlib joblib</code></pre>
    </li>
    <li>Download the dataset and scripts from the repository.</li>
    <li>Run preprocessing and modeling scripts, then visualize results using Power BI.</li>
</ol>

<h2>Contact</h2>
<p>If you have any questions or would like to discuss this project further, feel free to connect with me on LinkedIn or email me directly.</p>

</body>
</html>
