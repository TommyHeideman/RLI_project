Predictive Modeling and Data Cleaning Project
Project Overview
This project involves cleaning, processing, and modeling two datasets:

RLI Data: Insurance-related data containing ZIP codes and demographic information.
US Census Data: Population and household estimates by ZIP code.
Key Steps
Clean and merge datasets.
Train machine learning models to predict target variables.
Enhance predictions by adding geographic information (state and city) based on ZIP codes.
1. Data Cleaning Steps
Input Files
IOWA-RLI PUP Data.csv: Original RLI data.
USCensus_AgeandSex.csv: Census demographic data.
Step 1: Reading and Standardizing RLI Data
Clean column names to snake_case.
Standardize spacing in the producer_entity column:
r
Copy code
library(dplyr)
library(stringr)

RLI_original_data <- RLI_original_data %>%
  mutate(producer_entity = str_squish(producer_entity))
Step 2: Formatting ZIP Codes
Ensure ZIP codes are 5-digit strings with leading zeros:
r
Copy code
RLI_original_data <- RLI_original_data %>%
  mutate(postal_code = str_pad(as.character(postal_code), width = 5, pad = "0"))

census_data <- census_data %>%
  mutate(ZIP_CODE = str_pad(as.character(ZIP_CODE), width = 5, pad = "0"))
Step 3: Merging RLI and Census Data
Merge RLI data with Census data on ZIP codes using a left join:
r
Copy code
merged_data <- RLI_original_data %>%
  left_join(census_data, by = c("postal_code" = "ZIP_CODE"))
Save the resulting dataset as:
RLI_Census_MergedData.csv
2. Machine Learning Modeling
Input File
Filtered_Left_Joined_Dataset.csv: Preprocessed dataset for modeling.
Step 1: Data Preprocessing
Drop irrelevant columns:
arduino
Copy code
'RenewFlag', 'CnclMo', 'NotCancFlag'
Log-transform the target variable to normalize skewness:
python
Copy code
y_full_log = np.log1p(y_full)
Step 2: Model Training
Models Trained:

Random Forest: Baseline model.
XGBoost: Hyperparameter-tuned model using Grid Search.
Best Parameters:
json
Copy code
{'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300, 'subsample': 1.0}
Stacking Regressor: Combined Random Forest and XGBoost with Linear Regression as the final estimator.
Step 3: Model Performance
<table> <tr> <th>Model</th> <th>MSE</th> <th>MAE</th> <th>R² Score</th> </tr> <tr> <td>Random Forest</td> <td>2118.23</td> <td>18.33</td> <td>0.694</td> </tr> <tr> <td>XGBoost (Best Params)</td> <td>1413.67</td> <td>15.04</td> <td>0.796</td> </tr> <tr> <td>Stacking Model</td> <td>1376.02</td> <td>14.94</td> <td>0.801</td> </tr> </table>
Step 4: Cross-Validation
XGBoost Cross-Validation R² Scores:

csharp
Copy code
[0.8188, 0.8110, 0.8128, 0.8147, 0.8137]
Mean: 0.8142
Step 5: Saving the Models
The trained models were saved using joblib:

python
Copy code
from joblib import dump

dump(rf_model, 'random_forest_modelV3.joblib')
dump(best_xgb, 'xgboost_modelV3.joblib')
dump(stacking_model, 'stacking_modelV3.joblib')
3. Predictions on New Data
Input File
Merged_USCensus_Data.csv: Preprocessed data with demographic features.
Steps
Standardize columns to match training data.
Make predictions using the XGBoost model:
python
Copy code
data_to_scale = chunk.reindex(columns=expected_columns, fill_value=0)
predictions = loaded_model.predict(X_chunk_scaled)
Output File
predictions.csv: Contains the predicted results:
ZIP_CODE	PREDICTED_TARGET
12345	543.21
4. Enhancing Predictions with Geographic Information
Objective
Merge predictions with state and city details using the zipcodeR library.

Steps
Format ZIP Codes:
r
Copy code
data <- data %>%
  mutate(ZIP_CODE = sprintf("%05d", as.numeric(ZIP_CODE))) %>%
  mutate(ZIP_CODE = as.factor(ZIP_CODE))
Load ZIP Code Database:
r
Copy code
zip_details <- zipcodeR::zip_code_db %>%
  mutate(zipcode = sprintf("%05s", as.character(zipcode))) %>%
  mutate(zipcode = as.factor(zipcode))
Left Join to Add State/City:
r
Copy code
data_with_states_regions <- data %>%
  left_join(zip_details, by = c("ZIP_CODE" = "zipcode"))
Save Final Output:
The enhanced predictions are saved to:
Data_with_States_and_Regions.xlsx
5. Execution Steps
Python Workflow
Train the model:

bash
Copy code
python train_model.py
Generate predictions:

bash
Copy code
python predict_model.py
R Workflow
Run the R script to enhance predictions with geographic information:

r
Copy code
source("merge_with_zipcodes.R")
6. Dependencies
Python Libraries
Install the required libraries:

bash
Copy code
pip install pandas numpy scikit-learn xgboost matplotlib joblib
R Packages
Install the required packages:

r
Copy code
install.packages(c("zipcodeR", "readxl", "writexl", "dplyr"))
7. Notes
The project efficiently processes large datasets using chunk-based operations.
ZIP codes are formatted to ensure successful merging across all steps.
Power BI adjustments aligned the datasets before modeling.
