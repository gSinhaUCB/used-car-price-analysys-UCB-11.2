# used-car-price-analysys-UCB-11.2
CRISP-DM analysis predicting used car values for dealership inventory strategy

# Used Car Price Analysis & Valuation Engine

An end-to-end machine learning and data science project developed using the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** framework. This project analyzes over **400,000 historical used car listings** across the United States to uncover key vehicle depreciation drivers and build predictive pricing models for dealership inventory management.

---

## Executive Summary & Business Objective

Used car dealerships face significant financial risk due to inaccurate trade-in appraisals, mispriced inventory, and equity drag on slow-moving vehicles. 

Core Objective: Build a robust, scalable predictive regression pipeline to identify what vehicle attributes drive market value and provide dealership managers with actionable sourcing, acquisition, and pricing guidelines.  

Provide actionable recommendations for used car dealerships to: 

1.  Maximize Profit Margins: Focus inventory acquisition on vehicles that hold long-term value and command price premiums.

2.  Minimize Turn Time: Avoid high-depreciation inventory that stalls on the lot and requires heavy markdowns.

3.  Data-Driven Pricing: Price vehicles accurately based on measurable feature rather than intuition alone


---

## CRISP-DM Methodology

This project strictly follows the 6-phase CRISP-DM framework:

1. *Business Understanding:* Define dealership objectives—maximize profit margins, minimize lot turn time, and standardize trade-in valuations using data.

2. *Data Understanding:* Explore ~426,000 raw vehicle listings containing numerical (price, year, odometer) and categorical (make, model, condition, title status, fuel, drive) attributes.

3. *Data Preparation:* Clean outliers, impute missing values, engineer the `age` feature, and encode high/low-cardinality categorical variables.

4. *Modeling:* Train and tune multiple regularized regression models (Linear, Ridge, Lasso) using *5-Fold Cross-Validation*.

5. *Evaluation:* Assess model performance (RMSE, R^2) on held-out test data and evaluate business utility.

6. *Deployment Strategy:* Translate standardized model coefficients into actionable rules for inventory managers.

---

## Data Preparation Pipeline

To maintain logical progression and prevent data leakage, the data pipeline is structured across 5 clean, sequential steps:

1. *Target Variable "price":*  Filter out extreme outliers: e.g., keep prices between $1,000 and $100,000.

2. *Feature Engineering "odometer / Age":*
    - Filter out unrealistic mileage: e.g., odometer between 1,000 and 300,000 miles.
    - Create an age feature: age = current_year - year.

3. *Handling Missing Values:*
    - Drop columns with massive missingness if they don't add value (e.g., VIN, size).
    - Impute or group categorical nulls (e.g., fill condition, cylinders, drive with 'unknown').

4. *Categorical Encoding:*
    - Use One-Hot Encoding for lower-cardinality features (fuel, transmission, drive, title_status).
    - Use Target Encoding for high-cardinality features like model or manufacturer.

5. *Final Type Fix & Pre-Modeling Check*

    - Clean up any leftover non-numeric metadata columns (like raw region or posting_date) and ensure all numbers are finite integers or floats.(Added after my modeling code failed in initial pass)


 ## Data Modeling Pipelines for Linear Regression, Ridge, Lasso

1) Define Features and Target dataframes
2) Create training and test data

Try 3 models and see which one works best.  
 - Linear Regression
 - Ridge Regression 
 - Lasso Regression 

 To evaluate what factors make a car more or less expensive, I will compare these three distinct models using 5-Fold Cross-Validation:

 **Observations**

1.  The baseline Linear Regression, Ridge, and Lasso models yielded nearly identical RMSE scores during cross-validation. This indicates that given the large sample size of our dataset, standard OLS regression did not suffer from extreme overfitting, and default regularization penalties did not significantly alter model weights.

2.  Let us see how the models would perform with a wider range of alpha.

Hyperparameter tuning across a wide range of alpha values ([0.1,100000]) revealed that Ridge Regression achieved its optimal cross-validation RMSE (7,463.33) at =10.0, while Lasso Regression performed best at =0.1.   Because the dataset contains over 300,000 samples across roughly 25 features, standard Linear Regression does not suffer from high variance or severe overfitting. Consequently, low regularization penalties (alpha <= 10) produce RMSE improvements that differ by less than a fraction of a cent. When penalties are increased dramatically (alpha >= 1,000), model performance for Lasso degrades significantly due to underfitting, confirming that weak regularization is optimal for this dataset.






## Business Recommendations for Dealerships

Based on empirical model findings, we recommend four concrete inventory management rules:

1. *Target the "Sweet Spot" Inventory:* Prioritize acquiring 3–6 year-old 4WD/AWD trucks and crossover SUVs under 80,000 miles. They offer the best balance of lower acquisition cost, high retail demand, and manageable depreciation.

2. *Adjust Diesel Sourcing Thresholds:* Maintain higher mileage acquisition limits for heavy-duty diesel work trucks (up to 150,000 miles), as secondary market demand remains resilient.

3. *Enforce Title Penalties:* Avoid salvage or rebuilt title vehicles; severe market markdowns (30%–50%) eat into profit margins.


---

## Tech Stack & Libraries

* Python Version: `3.10+`

* Data Manipulation: `pandas`, `numpy`, `datetime`

* Machine Learning: `scikit-learn` 
        (`Pipeline`, `StandardScaler`, `LinearRegression`, `Ridge`, `Lasso`, `GridSearchCV`, `cross_validate`)

* Categorical Encoding: `category_encoders` (`TargetEncoder`)

* Data Visualization: `matplotlib`, `seaborn`


## How to Run This Project

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/used-car-price-analysis.git](https://github.com/your-username/used-car-price-analysis.git)
   cd used-car-price-analysis

   