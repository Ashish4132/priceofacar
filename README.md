Sure, I can help you create a README file based on the information you've provided.

## What drives the price of a car?

This project aims to identify the key factors influencing used car prices, providing valuable insights for a used car dealership to optimize their inventory and pricing strategies.

### Project Overview

This application explores a large dataset of 426,000 used car listings from Kaggle, a subset of an original 3 million record dataset. The primary goal is to understand what makes a used car more or less expensive, ultimately providing clear, data-driven recommendations to a used car dealership client.

### CRISP-DM Framework

The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework, ensuring a structured approach to solving the business problem.

#### Business Understanding

**Business Problem:** Identify the key drivers of used car prices to help a used car dealership understand consumer value and fine-tune their inventory.

**Data Problem Definition:** This task can be reframed as a supervised regression problem. We aim to build a predictive model that estimates the `price` of a used car based on its various attributes (e.g., `year`, `manufacturer`, `odometer`, `condition`, `fuel`, `transmission`, etc.). The objective is to identify the features with the strongest correlation to car prices and quantify their impact.

#### Data Understanding

The dataset `vehicles.csv` contains information on 426,880 used cars across 18 attributes.

**Initial Data Exploration Steps:**

1.  **Load the dataset:** `df = pd.read_csv("data/vehicles.csv")`
2.  **Inspect shape and initial rows:** `df.shape`, `df.head()`
3.  **Check data types and non-null counts:** `df.info()`
4.  **Descriptive statistics for numerical columns:** `df.describe()`
5.  **Identify missing values:** `df.isnull().sum()` and calculate `missing_percentage`.

**Key Observations from Data Understanding:**

  * **Dataset Size:** (426880, 18)
  * **Missing Values:**
      * Significant missing values in `condition` (40.79%), `cylinders` (41.62%), `VIN` (37.73%), `drive` (30.59%), `size` (71.77%), `type` (21.75%), and `paint_color` (30.50%).
      * Minor missing values in `year`, `manufacturer`, `model`, `fuel`, `odometer`, `title_status`, `transmission`.
  * **Numerical Features Statistics:**
      * `price`: Wide range, with a minimum of 0 and a maximum of 3.7 billion. This indicates potential outliers and data entry errors that need cleaning.
      * `year`: Ranges from 1900 to 2022.
      * `odometer`: Ranges from 0 to 10 million, also suggesting outliers.
  * **Target Variable (`price`) Issues:** Contains zero values, which are not realistic for car prices and will be removed during data preparation.

**Visualizations from Data Understanding:**

  * **Log-Transformed Car Prices:** A right-skewed distribution, which became more normal after `np.log1p` transformation, indicating that price is better analyzed on a logarithmic scale.
  * **Odometer Readings:** A right-skewed distribution, with many cars having lower mileage and a tail extending to very high odometer readings.
  * **Car Age:** A distribution showing a concentration of newer cars (lower age), with a tail for older vehicles.

#### Data Preparation

This phase focuses on cleaning, engineering new features, transforming data, and preparing the dataset for modeling.

**Steps Taken:**

1.  **Handle Missing Data:** Rows with any missing values were dropped (`df.dropna(inplace=True)`).
2.  **Remove Duplicates:** Duplicate rows were removed (`df.drop_duplicates(inplace=True)`).
3.  **Filter Price Outliers:** Prices were filtered to a reasonable range: `1000 <= price <= 150000`. This addresses the unrealistic min/max values observed initially.
4.  **Filter Odometer Outliers:** Odometer readings were capped at `400000` to remove extreme outliers.
5.  **Feature Engineering:**
      * `age`: Calculated as `2025 - year` to represent the age of the car, and the original `year` column was dropped.
      * `log_price`: A log transformation of the `price` column was created (`np.log1p(df['price'])`) for better distribution and model performance.

**Visualizations from Data Preparation:**

  * **Price Distribution:** Box plot, violin plot, and histogram confirm that after cleaning, the majority of used car demand falls within the \\$5,000 to \\$25,000 range.
  * **Price Categories by Year:** Histograms for low, middle, and high-priced cars show that most demand is for cars with a model year of 2008 or newer.
  * **Odometer Distribution:** Box plot for odometer after filtering confirms a more reasonable range, with highest demand for cars between 50,000 to 150,000 miles.

#### EDA: Visualizing Relationships Between Features and Price

  * **Price and Year:** Scatter plot shows a positive correlation; newer cars generally have higher prices.
  * **Price and Odometer:** Scatter plot shows a negative correlation; cars with higher mileage generally have lower prices.
  * **Correlation Matrix (Price, Year, Odometer):**
    ```
              price      year  odometer
    price   1.000000  0.343463 -0.445633
    year    0.343463  1.000000 -0.285684
    odometer -0.445633 -0.285684  1.000000
    ```
    This confirms the observed correlations: `year` has a positive correlation with `price`, and `odometer` has a negative correlation.
  * **Price and Manufacturer (Brand):** Box plot reveals significant price differences across manufacturers, with luxury brands (e.g., Audi, Mercedes-Benz, BMW) commanding higher prices.
  * **Price and Cylinders:** Box plot indicates that cars with more cylinders tend to have higher prices.
  * **Price and Condition:** Box plot shows a clear relationship; "new" or "excellent" conditions correlate with higher prices.
  * **Price and Fuel:** Violin plot of `log_price` vs. `fuel` shows variations, with some fuel types (e.g., electric, diesel) potentially having different price distributions.
  * **Price and Title Status:** Box plot suggests that "lien" and "clean" titles are associated with higher prices.
  * **Price and Transmission:** Violin plot of `log_price` vs. `transmission` indicates that automatic transmissions generally correlate with higher average prices compared to manual.
  * **Price and Type:** Box plot demonstrates that certain car types (e.g., trucks, offroad, pickups) tend to have significantly higher prices than others (e.g., sedans, hatchbacks, minivans).
  * **Price and Drive:** Box plot indicates that 4WD (Four-wheel drive) cars generally have higher prices than FWD (Front-wheel drive).

#### Modeling

Several regression models were built to predict car prices.

**Pre-modeling Steps:**

1.  **Feature Selection:** The features used for modeling are `condition`, `cylinders`, `fuel`, `transmission`, `drive`, `type`, `paint_color`, `manufacturer`, `title_status`, `odometer`, and the engineered `age` (dropping `year`). `id` and `VIN` were excluded as they are unique identifiers and `region` and `state` were excluded to simplify the model and focus on car attributes.
2.  **Target and Features:** `X = df.drop(columns=['price'])`, `y = df['price']`.
3.  **One-Hot Encoding:** Categorical features were One-Hot Encoded using `pd.get_dummies` with `drop_first=True` to mitigate multicollinearity.
4.  **Train-Test Split:** Data was split into training and testing sets (`test_size=0.2`, `random_state=2`).
5.  **Error Metrics Function:** A custom function `error_metrics` was defined to consistently calculate MAE, MSE, RMSE, and R-squared for both training and testing sets.

**Models Implemented:**

1.  **Simple Linear Regression (with Polynomial Features):**

      * Pipeline: `StandardScaler()`, `PolynomialFeatures(degree=2)`, `LinearRegression()`.
      * **Results:**
          * `Train_MAE`: $3084.38
          * `Train_MSE`: 24750126.93
          * `Train_RMSE`: $4974.95
          * `Train_Score` (R2): 0.86
          * `Test_MAE`: $3516.13
          * `Test_MSE`: 45013570.7
          * `Test_RMSE`: $6709.22
          * `Test_Score` (R2): 0.74

2.  **Ridge Regression (with Polynomial Features):**

      * Pipeline: `StandardScaler()`, `PolynomialFeatures(degree=2)`, `Ridge()`.
      * **Results:**
          * `Train_MAE`: $3084.68
          * `Train_MSE`: 24750509.98
          * `Train_RMSE`: $4974.99
          * `Train_Score` (R2): 0.86
          * `Test_MAE`: $3479.23
          * `Test_MSE`: 39190833.5
          * `Test_RMSE`: $6260.26
          * `Test_Score` (R2): 0.77

3.  **Lasso Regression:**

      * Alpha tuned using `GridSearchCV` (best alpha: 0.464).
      * **Results:**
          * `Mean Squared Error on Training data`: 24750868.96
          * `Mean Squared Error on Test data`: 39192931.32
      * **Top 5 Features by Absolute Coefficient Value (Lasso):**
          * `manufacturer_tesla`: 22961.32
          * `manufacturer_mercedes-benz`: 12154.51
          * `manufacturer_ram`: 11956.40
          * `manufacturer_audi`: 11776.49
          * `manufacturer_bmw`: 11116.74
      * **Actual vs Predicted Plots:** Scatter plots show a strong linear relationship between actual and predicted prices for both training and test data, indicating a good model fit.

#### Evaluation

**Model Performance Summary:**

  * **Linear Regression:** Achieved an R-squared of 0.86 on training data and 0.74 on test data. The drop suggests some overfitting.
  * **Ridge Regression:** Achieved an R-squared of 0.86 on training data and 0.77 on test data. The smaller gap between train and test R-squared (0.09) compared to Linear Regression (0.12) indicates that Ridge regularization effectively reduced overfitting and improved generalization to unseen data.
  * **Lasso Regression:** Tuned to an optimal alpha of \~0.46. While MSEs are similar to Ridge, Lasso identified key features.

**Overall Assessment:**

The Ridge Regression model appears to be the most robust among the tested models, providing a good balance between fitting the training data and generalizing to new, unseen data, as evidenced by its higher test R-squared (0.77) and lower RMSE compared to the unregularized Linear Regression. The Lasso model also performs well and provides interpretability by identifying important features.

The models demonstrate a strong ability to predict used car prices, explaining up to 77% of the variance. The Mean Absolute Error (MAE) for the best model is around $3,479, which is a practical and acceptable level of error for a used car dealership.

**Recommendations for Client (Used Car Dealership):**

Based on our analysis, here are the key drivers of used car prices and recommendations for your inventory and sales strategies:

1.  **High Predictive Accuracy:** Our model explains 77% of used car price variation, providing a strong understanding of market value. This means you can rely on the insights to make informed decisions.
2.  **Reliable Price Estimates:** On average, our model's predictions are within $3,479 of the actual price. This offers a practical and precise tool for setting competitive prices for your inventory.
3.  **Age & Mileage are Key:**
      * **Newer cars with lower odometer readings consistently command higher prices.** Prioritize acquiring and marketing vehicles that are relatively new and have less mileage. These will likely sell faster and at a better profit margin.
      * **Consumer demand is highest for cars with model years from 2008 onwards.** Focus on this age range.
      * **Odometer readings between 50,000 and 150,000 miles show the highest demand.** Cars with odometers higher than 250,000 miles have significantly lower demand.
4.  **Brand Matters:** **Luxury brands like Audi, Mercedes-Benz, BMW, and Tesla significantly boost a car's value.** Stocking a good selection of these brands, especially in good condition, can attract higher-paying customers.
5.  **Condition is Crucial:** **Vehicles in "new" or "excellent" condition fetch premium prices.** Reconditioning vehicles to improve their condition (e.g., detailing, minor repairs) is a worthwhile investment that can substantially increase their market value.
6.  **Automatic is Preferred:** **Cars with automatic transmissions are generally more valuable to consumers.** While manual cars exist, the broader market prefers automatics, so prioritize these in your inventory.
7.  **Vehicle Type Impacts Price:** **Trucks, off-road vehicles, and pickups tend to have higher prices** compared to sedans, hatchbacks, wagons, and minivans. Consider diversifying your inventory to include these higher-value vehicle types if your market allows.
8.  **Title Status:** Cars with "lien" or "clean" titles generally command higher prices.
9.  **Drive Type:** 4WD vehicles tend to have higher prices than FWD, but the difference is not as substantial as other factors.

#### Deployment

This report serves as the deployment of our findings to your client, a used car dealership. The insights provided are actionable and can be integrated into your inventory management, pricing strategies, and marketing efforts to optimize sales and profitability.

**Future Considerations/Next Steps:**

  * **Explore feature interactions:** Investigating how combinations of features (e.g., age and manufacturer) affect price.
  * **More advanced models:** Experimenting with gradient boosting models (XGBoost, LightGBM) for potentially higher accuracy.
  * **Hyperparameter tuning:** More extensive tuning of the selected models.
  * **Geographical Analysis:** Incorporating `region` and `state` to see if there are local market price variations.
  * **Time-series analysis:** If historical pricing data with timestamps were available, analyzing price trends over time.

-----
