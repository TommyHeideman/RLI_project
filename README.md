# RLI_project

1. Project Overview
This project involves cleaning, processing, and modeling two datasets:

RLI Data: Insurance-related data containing ZIP codes and demographic information.
US Census Data: Population and household estimates by ZIP code.
Key Steps:
Clean and merge datasets.
Train machine learning models to predict target variables.
Enhance predictions by adding geographic information (state and city) based on ZIP codes.
2. Data Cleaning Steps
Input Files
IOWA-RLI PUP Data.csv: Original RLI data.
USCensus_AgeandSex.csv: Census demographic data.
Step 1: Reading and Standardizing RLI Data
Cleaned column names to snake_case for consistency using the janitor package.
Standardized the producer_entity column to ensure consistent spacing:
R
Copy code
library(dplyr)
library(stringr)

RLI_original_data <- RLI_original_data %>%
  mutate(producer_entity = str_squish(producer_entity))
Step 2: Formatting ZIP Codes
Ensured the postal_code column (RLI) and ZIP_CODE column (Census data):
Were formatted as 5-digit strings with leading zeros.
Converted to character types:
R
Copy code
RLI_original_data <- RLI_original_data %>%
  mutate(postal_code = str_pad(as.character(postal_code), width = 5, pad = "0"))

census_data <- census_data %>%
  mutate(ZIP_CODE = str_pad(as.character(ZIP_CODE), width = 5, pad = "0"))
Step 3: Left Join Between RLI and Census Data
Merged RLI data with Census data on ZIP codes using a left join:

R
Copy code
merged_data <- RLI_original_data %>%
  left_join(census_data, by = c("postal_code" = "ZIP_CODE"))
The resulting dataset was saved to:

RLI_Census_MergedData.csv
3. Machine Learning Modeling
Input File
Filtered_Left_Joined_Dataset.csv: Preprocessed dataset for modeling.
Step 1: Data Preprocessing
Dropped irrelevant columns: 'RenewFlag', 'CnclMo', 'NotCancFlag'.
Selected demographic features for modeling.
Handled missing values by imputing column means.
Log-transformed the target variable (PostalCode.Count) to reduce skewness:
python
Copy code
y_full_log = np.log1p(y_full)
Step 2: Model Training
Models Trained
Random Forest: Baseline model.
XGBoost:
Hyperparameters tuned using GridSearchCV.
Best parameters:
arduino
Copy code
{'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300, 'subsample': 1.0}
Stacking Regressor:
Combined Random Forest and XGBoost with Linear Regression as the final estimator.
Step 3: Model Performance
Model	MSE	MAE	R² Score
Random Forest	2118.23	18.33	0.694
XGBoost (Best Params)	1413.67	15.04	0.796
Stacking Model	1376.02	14.94	0.801
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
dump(rf_model, 'random_forest_modelV3.joblib')
dump(best_xgb, 'xgboost_modelV3.joblib')
dump(stacking_model, 'stacking_modelV3.joblib')
4. Predictions on New Data
Input File
Merged_USCensus_Data.csv: Preprocessed data with demographic features.
Steps:
Loaded the saved XGBoost model and scaler.
Processed data in chunks to handle large files efficiently.
Standardized columns to match model input and made predictions:
python
Copy code
data_to_scale = chunk.reindex(columns=expected_columns, fill_value=0)
predictions = loaded_model.predict(X_chunk_scaled)
Output File
Predictions saved to predictions.csv with:
ZIP_CODE
PREDICTED_TARGET
5. Enhancing Predictions with Geographic Information
Objective
Merge the predictions with state and city details using the zipcodeR library.

Input File
Predictions_FinalV2.xlsx: Predictions with ZIP codes.
Steps Performed
Load Data and Format ZIP Codes:

Ensure leading zeros are preserved and ZIP codes are converted to a factor:
R
Copy code
data <- data %>%
  mutate(ZIP_CODE = sprintf("%05d", as.numeric(ZIP_CODE))) %>%
  mutate(ZIP_CODE = as.factor(ZIP_CODE))
Load ZIP Code Database:

Used the zipcodeR library:
R
Copy code
zip_details <- zipcodeR::zip_code_db %>%
  mutate(zipcode = sprintf("%05s", as.character(zipcode))) %>%
  mutate(zipcode = as.factor(zipcode))
Left Join with ZIP Code Details:

Merged predictions with state and city information:
R
Copy code
data_with_states_regions <- data %>%
  left_join(zip_details, by = c("ZIP_CODE" = "zipcode"))
Check for Missing Data:

Identified rows with missing state or region information.
Save Final Data:

Saved the enhanced predictions to:
Data_with_States_and_Regions.xlsx
6. Execution Steps
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
Merge predictions with ZIP code data:
Run the provided R script in RStudio.
Output:
Enhanced predictions saved to Data_with_States_and_Regions.xlsx.
7. Dependencies
Python Libraries
pandas
numpy
scikit-learn
xgboost
joblib
matplotlib
Install via:

bash
Copy code
pip install pandas numpy scikit-learn xgboost matplotlib joblib
R Packages
zipcodeR
readxl
writexl
dplyr
Install in R:

R
Copy code
install.packages(c("zipcodeR", "readxl", "writexl", "dplyr"))
