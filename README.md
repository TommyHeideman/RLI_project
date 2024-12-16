<!DOCTYPE html>
<html>
<head>
    <title>Predictive Modeling and Data Cleaning Project</title>
</head>
<body>

<h1>Predictive Modeling and Data Cleaning Project</h1>

<h2>Project Overview</h2>
<p>This project involves cleaning, processing, and modeling two datasets:</p>
<ul>
    <li><strong>RLI Data:</strong> Insurance-related data containing ZIP codes and demographic information.</li>
    <li><strong>US Census Data:</strong> Population and household estimates by ZIP code.</li>
</ul>
<h3>Key Steps</h3>
<ol>
    <li>Clean and merge datasets.</li>
    <li>Train machine learning models to predict target variables.</li>
    <li>Enhance predictions by adding geographic information (state and city) based on ZIP codes.</li>
</ol>

<hr>

<h2>1. Data Cleaning Steps</h2>

<h3>Input Files</h3>
<ul>
    <li><strong>IOWA-RLI PUP Data.csv:</strong> Original RLI data.</li>
    <li><strong>USCensus_AgeandSex.csv:</strong> Census demographic data.</li>
</ul>

<h3>Step 1: Reading and Standardizing RLI Data</h3>
<ul>
    <li>Clean column names to <strong>snake_case</strong>.</li>
    <li>Standardize spacing in the <code>producer_entity</code> column:</li>
</ul>
<pre>
<code>
library(dplyr)
library(stringr)

RLI_original_data <- RLI_original_data %>%
  mutate(producer_entity = str_squish(producer_entity))
</code>
</pre>

<h3>Step 2: Formatting ZIP Codes</h3>
<ul>
    <li>Ensure ZIP codes are <strong>5-digit strings</strong> with leading zeros:</li>
</ul>
<pre>
<code>
RLI_original_data <- RLI_original_data %>%
  mutate(postal_code = str_pad(as.character(postal_code), width = 5, pad = "0"))

census_data <- census_data %>%
  mutate(ZIP_CODE = str_pad(as.character(ZIP_CODE), width = 5, pad = "0"))
</code>
</pre>

<h3>Step 3: Merging RLI and Census Data</h3>
<ul>
    <li>Merge RLI data with Census data on ZIP codes using a <strong>left join</strong>:</li>
</ul>
<pre>
<code>
merged_data <- RLI_original_data %>%
  left_join(census_data, by = c("postal_code" = "ZIP_CODE"))
</code>
</pre>
<p>Save the resulting dataset as:</p>
<ul>
    <li><strong>RLI_combined_Census_Sex.csv</strong></li>
</ul>

<hr>

<h2>2. Machine Learning Modeling</h2>

<h3>Input File</h3>
<ul>
    <li><strong>Filtered_Left_Joined_Dataset.csv:</strong> Preprocessed dataset for modeling.</li>
</ul>

<h3>Step 1: Data Preprocessing</h3>
<ul>
    <li>Drop irrelevant columns:
        <pre><code>'RenewFlag', 'CnclMo', 'NotCancFlag'</code></pre>
    </li>
    <li>Log-transform the target variable to normalize skewness:
        <pre><code>
y_full_log = np.log1p(y_full)
</code></pre>
    </li>
</ul>

<h3>Step 2: Model Training</h3>
<ul>
    <li><strong>Models Trained:</strong></li>
    <ol>
        <li><strong>Random Forest:</strong> Baseline model.</li>
        <li><strong>XGBoost:</strong> Hyperparameter-tuned model using Grid Search.</li>
        <li><strong>Stacking Regressor:</strong> Combines Random Forest and XGBoost with Linear Regression as the final estimator.</li>
    </ol>
</ul>

<h3>Step 3: Model Performance</h3>
<table border="1">
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
            <td>Stacking Model</td>
            <td>1376.02</td>
            <td>14.94</td>
            <td>0.801</td>
        </tr>
    </tbody>
</table>

<h3>Step 4: Cross-Validation</h3>
<p>XGBoost Cross-Validation R² Scores:</p>
<pre>
<code>[0.8188, 0.8110, 0.8128, 0.8147, 0.8137]
Mean: 0.8142
</code>
</pre>

<h3>Step 5: Saving the Models</h3>
<p>The trained models were saved as:</p>
<ul>
    <li><strong>random_forest_modelV3.joblib</strong></li>
    <li><strong>xgboost_modelV3.joblib</strong></li>
    <li><strong>stacking_modelV3.joblib</strong></li>
</ul>

<hr>

<h2>3. Predictions on New Data</h2>

<h3>Input File</h3>
<ul>
    <li><strong>Merged_USCensus_Data.csv:</strong> Preprocessed data with demographic features.</li>
</ul>

<h3>Steps</h3>
<ol>
    <li>Standardize columns to match training data.</li>
    <li>Make predictions using the XGBoost model:</li>
</ol>
<pre>
<code>
data_to_scale = chunk.reindex(columns=expected_columns, fill_value=0)
predictions = loaded_model.predict(X_chunk_scaled)
</code>
</pre>

<h3>Output File</h3>
<ul>
    <li><strong>predictions.csv</strong>: Contains the predicted results.</li>
</ul>

<hr>

<h2>4. Enhancing Predictions with Geographic Information</h2>

<h3>Objective</h3>
<p>Merge predictions with <strong>state</strong> and <strong>city</strong> details using the <code>zipcodeR</code> library.</p>

<h3>Steps</h3>
<ol>
    <li>Format ZIP Codes:</li>
</ol>
<pre>
<code>
data <- data %>%
  mutate(ZIP_CODE = sprintf("%05d", as.numeric(ZIP_CODE))) %>%
  mutate(ZIP_CODE = as.factor(ZIP_CODE))
</code>
</pre>

<ol start="2">
    <li>Load ZIP Code Database:</li>
</ol>
<pre>
<code>
zip_details <- zipcodeR::zip_code_db %>%
  mutate(zipcode = sprintf("%05s", as.character(zipcode))) %>%
  mutate(zipcode = as.factor(zipcode))
</code>
</pre>

<ol start="3">
    <li>Left Join to Add State/City:</li>
</ol>
<pre>
<code>
data_with_states_regions <- data %>%
  left_join(zip_details, by = c("ZIP_CODE" = "zipcode"))
</code>
</pre>

<ol start="4">
    <li>Save Final Output:</li>
</ol>
<p>The enhanced predictions are saved to:</p>
<ul>
    <li><strong>Data_with_States_and_Regions.xlsx</strong></li>
</ul>

<hr>

<h2>5. Integration into Power BI</h2>
<p>The final predictions, along with enhanced geographic information, were added back into the Power BI report for further visualization and reporting. This allowed stakeholders to analyze predictions alongside state and city-level insights directly in the Power BI dashboard.</p>

<hr>

<h2>6. Dependencies</h2>

<h3>Python Libraries</h3>
<p>Install the required libraries:</p>
<pre>
<code>
pip install pandas numpy scikit-learn xgboost matplotlib joblib
</code>
</pre>

<h3>R Packages</h3>
<p>Install the required packages:</p>
<pre>
<code>
install.packages(c("zipcodeR", "readxl", "writexl", "dplyr"))
</code>
</pre>

<hr>

<h2>7. Notes</h2>
<ul>
    <li>The project efficiently processes large datasets using chunk-based operations.</li>
    <li>ZIP codes are formatted to ensure successful merging across all steps.</li>
    <li>Power BI adjustments aligned the datasets before modeling.</li>
</ul>

</body>
</html>
