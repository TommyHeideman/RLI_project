# Predictive Modeling and Data Cleaning Project

## Project Overview
This project involves cleaning, processing, and modeling two datasets:
- **RLI Data**: Insurance-related data containing ZIP codes and demographic information.
- **US Census Data**: Population and household estimates by ZIP code.

### Key Steps
1. **Clean and merge datasets**.
   - Run the R script to merge the census data onto the RLI data (unused for modeling but included in the Power BI report).
   - Run the Python script to join the census datasets together.
   - Use Excel and `COUNTIF` functions between `Merged_USCensus_Data.csv` and RLI data to get postal code counts. Rename this file to `Filtered_Left_Joined_Dataset` and save as a CSV.
2. **Train machine learning models** to predict target variables.
   - Run the Python script using an IDE that supports `.ipynb` and the given dataset.
3. **Enhance predictions** by adding geographic information (state and city) based on ZIP codes.

---

## 1. Data Cleaning Steps

### I. R Script
Run in R-Studio and change file paths as needed.

#### **Input Files**
- **`IOWA-RLI PUP Data.csv`**: Original RLI data.
- **`USCensus_AgeandSex.csv`**: Census demographic data.

#### **Step 1: Reading and Standardizing RLI Data**
- Clean column names to snake_case.
- Standardize spacing in the `producer_entity` column:

```r
library(dplyr)
library(stringr)

RLI_original_data <- RLI_original_data %>%
  mutate(producer_entity = str_squish(producer_entity))
```

#### **Step 2: Formatting ZIP Codes**
Ensure ZIP codes are 5-digit strings with leading zeros:

```r
RLI_original_data <- RLI_original_data %>%
  mutate(postal_code = str_pad(as.character(postal_code), width = 5, pad = "0"))

census_data <- census_data %>%
  mutate(ZIP_CODE = str_pad(as.character(ZIP_CODE), width = 5, pad = "0"))
```

#### **Step 3: Merging RLI and Census Data**
Merge RLI data with Census data on ZIP codes using a left join:

```r
merged_data <- RLI_original_data %>%
  left_join(census_data, by = c("postal_code" = "ZIP_CODE"))
```

Save the resulting dataset as:
- `RLI_combined_Census_Sex.csv`

### II. Python Script: CensusCombined.ipynb
Run in Python IDE, and configure file paths.

#### **Input Files**
- `USCensus_Income.csv`: Census demographic data.
- `USCensus_AgeandSex.csv`: Census demographic data.

#### **Output File**
- `Merged_USCensus_Data.csv`
   - Use Excel `COUNTIF` functions to match postal codes and save as `Filtered_Left_Joined_Dataset.csv`.

---

## 2. Machine Learning Modeling

### Script: CapstoneModel.ipynb
Run in preferred Python IDE (e.g., Jupyter) and set file paths.

#### **Input File**
- `Filtered_Left_Joined_Dataset.csv`: Preprocessed dataset for modeling.

### **Step 1: Data Preprocessing**
- Drop irrelevant columns: `'RenewFlag', 'CnclMo', 'NotCancFlag'`
- Log-transform the target variable:
```python
y_full_log = np.log1p(y_full)
```

### **Step 2: Model Training**
Trained models include:
1. **Random Forest**: Baseline model.
2. **XGBoost**: Hyperparameter-tuned model using Grid Search.
3. **Stacking Regressor**: Combines Random Forest and XGBoost with Linear Regression as the final estimator.

### **Step 3: Model Performance**
| Model            | MSE     | MAE    | R² Score |
|------------------|---------|--------|-----------|
| Random Forest    | 2118.23 | 18.33  | 0.694     |
| XGBoost          | 1413.67 | 15.04  | 0.796     |
| Stacking Model   | 1376.02 | 14.94  | 0.801     |

### **Step 4: Cross-Validation**
XGBoost Cross-Validation R² Scores:
```
[0.8188, 0.8110, 0.8128, 0.8147, 0.8137]
Mean: 0.8142
```

### **Step 5: Saving the Models**
Models saved as:
- `random_forest_modelV3.joblib`
- `xgboost_modelV3.joblib`
- `stacking_modelV3.joblib`

---

## 3. Predictions on New Data

### **Input File**
- `Merged_USCensus_Data.csv`: Preprocessed data with demographic features.

### **Steps**
1. Standardize columns to match training data.
2. Make predictions using the XGBoost model:
```python
data_to_scale = chunk.reindex(columns=expected_columns, fill_value=0)
predictions = loaded_model.predict(X_chunk_scaled)
```

### **Output File**
- `predictions.csv`: Contains predicted results.
   - Add a difference column: `RLI Count - Predicted Outcome`.

---

## 4. Enhancing Predictions with Geographic Information

### Objective
Merge predictions with state and city details using the `zipcodeR` library.

### **Steps**
1. Format ZIP Codes:
```r
data <- data %>%
  mutate(ZIP_CODE = sprintf("%05d", as.numeric(ZIP_CODE))) %>%
  mutate(ZIP_CODE = as.factor(ZIP_CODE))
```
2. Load ZIP Code Database:
```r
zip_details <- zipcodeR::zip_code_db %>%
  mutate(zipcode = sprintf("%05s", as.character(zipcode))) %>%
  mutate(zipcode = as.factor(zipcode))
```
3. Left Join to Add State/City:
```r
data_with_states_regions <- data %>%
  left_join(zip_details, by = c("ZIP_CODE" = "zipcode"))
```
4. Save Final Output:
- `Data_with_States_and_Regions.xlsx`

---

## 5. Integration into Power BI
The final predictions, with enhanced geographic details, were integrated into Power BI for visualization. This enables stakeholders to analyze predictions alongside state and city-level insights directly in the Power BI dashboard.

---

## 6. Dependencies

### **Python Libraries**
Install required libraries:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib joblib
```

### **R Packages**
Install required packages:
```r
install.packages(c("zipcodeR", "readxl", "writexl", "dplyr"))
```

---

## 7. Notes
- The project efficiently processes large datasets using chunk-based operations.
- ZIP codes are formatted to ensure successful merging.
- Power BI adjustments aligned the datasets before modeling.
