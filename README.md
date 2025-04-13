# Health Insurance Premium Prediction with Machine Learning Using Python

---

![Project Banner](https://github.com/Uzo-Hill/Health-Insurance-Premium-Prediction-using-Machine-Learning/blob/main/projectimageintro.PNG)

## Objective
The goal of this project was to predict health insurance premiums based on various personal characteristics such as age, gender, BMI, number of children, smoking habits, and region of residence.

## Skills & Tools Used:
-  **Python libraries**: `pandas`, `numpy`, `seaborn`, `scikit-learn`
- **Data analysis**, **feature engineering**, and **model training**, **Linear Regression**
- **Cross-validation** for model validation


## Data Collection
The dataset for this project was sourced from [Kaggle.com]([https://www.kaggle.com/mirichoi0218/insurance](https://www.kaggle.com/code/shubhamptrivedi/health-insurance-price-predict-linear-regression).  
It contains **1,338 entries** and the following **7 columns**:
- **age**: Age of the individual.
- **sex**: Gender (male/female).
- **bmi**: Body Mass Index, a measure of body fat.
- **children**: Number of children covered by the insurance.
- **smoker**: Smoking status (yes/no).
- **region**: Geographic region of residence.
- **charges**: Annual insurance premium charged.

---

![DataFrame](https://github.com/Uzo-Hill/Health-Insurance-Premium-Prediction-using-Machine-Learning/blob/main/Dataset.PNG) 


## Exploratory Data Analysis (EDA)
- A **heatmap** was used to visualize correlations. It revealed that **smoker** status had the strongest positive correlation with **charges** (0.79), followed by **age** (0.30) and **BMI**.
- The target variable **charges** showed a **right-skewed distribution**, which was corrected using a **logarithmic transformation**.

---

![CorrelationMatrix](https://github.com/Uzo-Hill/Health-Insurance-Premium-Prediction-using-Machine-Learning/blob/main/CorrMatrix.png)

---

![Distribution of Smokers and Non smokers by Gender](https://github.com/Uzo-Hill/Health-Insurance-Premium-Prediction-using-Machine-Learning/blob/main/SmokersVSNonsmokers.png)

## Data Preprocessing
- Categorical features (**sex**, **smoker**) were converted to numeric values using `map()` functions.
- A duplicate row was detected and removed.
- **Logarithmic transformation** was applied to the **charges** column to reduce skewedness.

---

![Log Transformation of Features](https://github.com/Uzo-Hill/Health-Insurance-Premium-Prediction-using-Machine-Learning/blob/main/LogTransform.png)


## Feature Selection
- The **region** column was excluded from the model since it was evenly distributed and not highly correlated with **charges**.
- After feature selection, the dataset was split into **training** and **testing sets**.

## Model Training
Three machine learning models were trained:
1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**

---

```python
  
  # Linear Regression
  lr_model = LinearRegression()
  lr_model.fit(X_train, y_train)
  lr_pred_log = lr_model.predict(X_test)

  # Decision Tree Regressor

  dt_model = DecisionTreeRegressor(random_state=42)
  dt_model.fit(X_train, y_train)
  dt_pred_log = dt_model.predict(X_test)

  # Random Forest Regressor

  rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
  rf_model.fit(X_train, y_train)
  rf_pred_log = rf_model.predict(X_test)

  ```

## Model Evaluation
The evaluation results of the models were as follows:

- **Linear Regression**:
  - **MSE**: 52,784,508
  - **R²**: 0.7127
- **Decision Tree Regressor**:
  - **MSE**: 41,933,264
  - **R²**: 0.7718
- **Random Forest Regressor**:
  - **MSE**: 19,916,972
  - **R²**: 0.8916
 
  ---

  ![Actual Vs Predicted Insurance Premium Chareges](https://github.com/Uzo-Hill/Health-Insurance-Premium-Prediction-using-Machine-Learning/blob/main/Actual%20Vs%20Predicted%20Premium%20Insurance%20Charges.png)



## Feature Importance Analysis
- The **feature importance chart** showed that **smoker status** was the most important feature, followed by **BMI** and **age**.
- **Children** and **sex** had minimal impact on the prediction.

---
![Feature Importance Analysis](https://github.com/Uzo-Hill/Health-Insurance-Premium-Prediction-using-Machine-Learning/blob/main/FeatureImportanceAnalysis.png)



## Model Validation
- The **Random Forest Regressor** was validated using **5-fold cross-validation**, resulting in **cross-validated R² scores**:  
  `[0.7319, 0.8136, 0.7550, 0.7877, 0.7699]`, with a **mean R² score** of **0.7716**.

## Conclusion
The **Random Forest Regressor** model proved to be the best in predicting health insurance premiums. It effectively captured the underlying patterns of the data, achieving a high **R² score** and low **MSE**, making it a suitable model for predicting insurance premiums.

## Recommendations
- **Hyperparameter Tuning**: Using `GridSearchCV` or `RandomizedSearchCV` to fine-tune the Random Forest model.
- **Deployment**: Deploying the model into a production environment to allow real-time predictions of insurance premiums.

---

*Created by Uzoh C. Hillary*
