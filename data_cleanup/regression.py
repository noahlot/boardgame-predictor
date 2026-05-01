import dataset_cleanup as cleanup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import PredictionErrorDisplay
# import models
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

X, y = cleanup.load_engineer_data('data_cleanup/boardgames.csv')
X_train, X_test, y_train, y_test = cleanup.split_and_preprocess(X, y)

ridge_reg = Ridge(solver='saga', random_state=42)
linear_reg = LinearRegression()
forest = RandomForestRegressor(min_samples_leaf=10, max_depth=15, max_features='log2', random_state=42)

# fit and test models
ridge_reg.fit(X_train, y_train)
linear_reg.fit(X_train, y_train)
forest.fit(X_train, y_train)

ridge_pred = ridge_reg.predict(X_test)
linear_pred = linear_reg.predict(X_test)
forest_pred = forest.predict(X_test)
# get coefficients and metrics (r2, mean squared error) for each model
print("\n\n\nRidge Regression:\n")
mse_ridge = mean_squared_error(y_test, ridge_pred)
r2_ridge = r2_score(y_test, ridge_pred)
coeffs_ridge = pd.Series(ridge_reg.coef_, index=X.columns)
print(f"Coefficients:\n{coeffs_ridge}")
print(f"Mean Squared Error: {mse_ridge}")
print(f"R-squared Score: {r2_ridge}")
display_ridge = PredictionErrorDisplay(y_true=y_test, y_pred=ridge_pred)
display_ridge.plot()
plt.show()

print("\nLinear Regression:\n")
mse_linear = mean_squared_error(y_test, linear_pred)
r2_linear = r2_score(y_test, linear_pred)
coeffs_linear = pd.Series(linear_reg.coef_, index=X.columns)
print(f"Coefficients:\n{coeffs_linear}")
print(f"Mean Squared Error: {mse_linear}")
print(f"R-squared Score: {r2_linear}")
display_linear = PredictionErrorDisplay(y_true=y_test, y_pred=linear_pred)
display_linear.plot()
plt.show()

print("\nRandom Forest:\n")
mse_forest = mean_squared_error(y_test, forest_pred)
r2_forest = r2_score(y_test, forest_pred)
importances = pd.Series(forest.feature_importances_, index=X.columns)
print(f"Feature Importances:\n{importances}")
print(f"Mean Squared Error: {mse_forest}")
print(f"R-squared Score: {r2_forest}")
display_forest = PredictionErrorDisplay(y_true=y_test, y_pred=forest_pred)
display_forest.plot()
plt.show()