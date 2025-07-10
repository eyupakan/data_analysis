# import libraries
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


# load dataset (house price prediction)
data = fetch_california_housing(as_frame=True)
X = data.data[["MedInc", "AveRooms", "HouseAge"]]
y = data.target  # target veriable


# data train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# define model
ada_reg = AdaBoostRegressor(n_estimators=50, learning_rate=1, random_state=42)

# model training
ada_reg.fit(X_train, y_train)


# testing
y_pred = ada_reg.predict(X_test)

# evaluation: mse, rmse, r2_score
print(f"Adaboost regresyon (mse): {mean_squared_error(y_test,y_pred)}")
print(f"Adaboost regresyon (rmse): {root_mean_squared_error(y_test,y_pred)}")
print(f"Adaboost regresyon (R2): {r2_score(y_test,y_pred)}")

"""
Adaboost regresyon (mse): 0.6938730047196126
Adaboost regresyon (rmse): 0.8329903989360337
Adaboost regresyon (R2): 0.4713510081796194
"""
