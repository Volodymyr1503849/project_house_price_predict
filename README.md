# House Price Prediction
<p align="left"> 
</a>   <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> 
</a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://pytorch.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> 

## Project Objective:
The goal of this project is to predict house prices based on various features such as location, size, number of bedrooms, and other relevant attributes. This prediction can help potential buyers and sellers make informed decisions in the real estate market.

## Data and Preprocessing:

- ## Dataset: 
  The project uses a dataset containing information about houses, including both categorical (e.g., neighborhood, condition) and numerical features (e.g., square footage, number of bathrooms).
  ![Alt Text](./Screenshot%202024-11-01%20144254.png)
- ## Handling Missing Values: 
  Missing data was addressed by filling numerical features with their mean values and categorical features with their mode.
  ```python
  for column in data:
    if data[column].dtype in ["int64", "float64"]:
        if data[column].isnull().any():
            data[column] = data[column].fillna(data[column].mean())
    else:
        if data[column].isnull().any():
            data[column] = data[column].fillna(data[column].mode()[0])
            ```
- ## Feature Encoding: 
  Categorical variables were encoded using techniques such as One-Hot Encoding to convert them into a format suitable for machine learning algorithms.
  ```python
  data = pd.get_dummies(data, drop_first=True)
  ```
## Model and Evaluation:

- ## Model Choice: 
  The project initially implemented Linear Regression but faced issues with performance metrics. 
  ```
  Validation Mean Squared Error: 2641638813.2241855
  ```
  To improve accuracy, a Random Forest model was chosen due to its capability to capture complex relationships in the data.
  ```python
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)
  y_val_pred = model.predict(X_val)
  ```
- ## Evaluation Metrics: 
  The performance of the model was evaluated using Mean Squared Error (MSE) and R-squared values to assess how well the model predicts house prices.
  ```python
  val_mse = mean_squared_error(y_val, y_val_pred)
  print(f'Validation Mean Squared Error: {val_mse}')
  Validation Mean Squared Error: 821760492.6202474
  ```
## Results:
The Random Forest model demonstrated a significant improvement in prediction accuracy compared to the initial linear regression model, with a lower MSE indicating better performance in forecasting house prices.
```python
test_predictions = pd.DataFrame({'PredictedSalePrice': y_test_pred})
for i in test_predictions.iterrows():
    print(i)
```
```
(0, PredictedSalePrice    117202.33
Name: 0, dtype: float64)
(1, PredictedSalePrice    153011.0
Name: 1, dtype: float64)
(2, PredictedSalePrice    173201.86
Name: 2, dtype: float64)
(3, PredictedSalePrice    186180.9
Name: 3, dtype: float64)
(4, PredictedSalePrice    223479.5
```
![Alt text](./Screenshot%202024-11-01%20145514.png)


## Conclusion:
This project illustrates the application of machine learning techniques in predicting real estate prices. It emphasizes the importance of data preprocessing, feature selection, and model evaluation in achieving reliable predictive outcomes.
