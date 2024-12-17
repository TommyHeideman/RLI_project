# Predictive Modeling and Data Cleaning Project
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
