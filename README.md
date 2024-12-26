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

<h2>Introduction</h2>
<p>This project aimed to empower RLI Insurance with actionable insights to optimize their personal umbrella policy distribution. By analyzing existing policy data and U.S. Census demographics, the initiative focused on identifying growth opportunities in ZIP codes with high potential. Predictive models were developed to forecast areas with untapped potential, and results were visualized in an interactive Power BI dashboard, providing stakeholders with clear and actionable insights to refine their strategies.</p>

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
# Full code for model training and predictions (refer to project repository).
</code></pre>

<h2>3. Visualization and Insights</h2>
<p>Predictions were visualized in an interactive Power BI dashboard, showcasing ZIP codes with high growth potential. The visualization enabled easy exploration by geographic and demographic filters.</p>
<img src="https://github.com/TommyHeideman/RLI_project/raw/main/PowerBI_map.png" 
     alt="Power BI map showing model predictions and comparison to RLI policy counts">
<img src="https://github.com/TommyHeideman/RLI_project/raw/main/Model_Demographics.png" 
     alt="Demographics model insights visualization">

<h3>Power BI Dashboard Features</h3>
<ul>
    <li>Interactive map displaying ZIP codes with the top 100 growth opportunities.</li>
    <li>Filter options by state, region, income, and demographic segments.</li>
    <li>Designed for actionable insights and stakeholder decision-making.</li>
</ul>

<h2>4. Recommendations</h2>
<p>Based on our findings, we recommend:</p>
<ul>
    <li>Focusing marketing and distribution efforts on high-income, densely populated regions with a significant population aged 60–80.</li>
    <li>Expanding successful strategies from high-performing areas (e.g., South Dakota, Carolinas) to underperforming regions.</li>
    <li>Leveraging predictive insights to strengthen carrier partnerships and optimize distribution strategies.</li>
</ul>

<h2>5. Limitations</h2>
<ul>
    <li>The model assumes that areas with the highest policy counts are optimal targets, not accounting for policy type differences.</li>
    <li>Data snapshot-based analysis; not integrated with a real-time pipeline.</li>
    <li>High computational requirements due to extensive column selection.</li>
    <li><strong>NDA Restrictions:</strong> Due to confidentiality agreements, only summary results are shared, with no access to proprietary or provider-specific data.</li>
</ul>

<h2>6. Key Deliverables</h2>
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
