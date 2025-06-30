from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ----- Framework -----

california = fetch_california_housing()

x = pd.DataFrame(data=california.data, columns=california.feature_names)
print(x.shape)
x

y = california.target
print(y.shape)
y

# ---- Train Model -----

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

reg_model = LinearRegression()

reg_model.fit(x_train, y_train)

print('Coefficients:', reg_model.coef_)
print('Intercept:', reg_model.intercept_)

# ----- Visualise ----- 

y_pred = reg_model.predict(x_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Home Prices")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
features = x.columns
coefficients = reg_model.coef_

plt.barh(features, coefficients)
plt.xlabel("Coefficient Value")
plt.title("Feature Importance")
plt.grid(True)
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, edgecolor='k')
plt.xlabel("Prediction Error")
plt.title("Distribution of Residuals")
plt.grid(True)
plt.show()

# ----- Feature Weights ----- 

features = california.feature_names

feature_coef = list(zip(features, reg_model.coef_))
sorted_feature_coef = sorted(feature_coef, key = lambda x: abs(x[1]), reverse=True)

for feature, coef in sorted_feature_coef:
    print(f"{feature}: {format(coef, '.3f')}") 

# ----- Algorithm (MSE) -----

mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# compare vectoirised implementation with non-vectorised implementation

def my_MSE(y_true, y_pred):
  mse = np.mean((y_true - y_pred)**2)
  return mse

def my_MSE_2(y_true, y_pred):
    n = len(y_true)
    mse = 0
    for i in range(n):
        mse += (y_true[i] - y_pred[i]) ** 2
    mse /= n
    return mse

print(my_MSE(y_true=y_test, y_pred=y_pred))
print(my_MSE_2(y_true=y_test, y_pred=y_pred))

# ----- Normalisation -----

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

reg_model.fit(x_train_scaled, y_train)
y_pred_scaled = reg_model.predict(x_test_scaled)

mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error (with normalization):', mse)