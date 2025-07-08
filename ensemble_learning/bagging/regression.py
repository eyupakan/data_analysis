# import libraries
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# load dataset: ev fiyat tahmini
housing = fetch_california_housing()
X = housing.data  # features
y = housing.target  # hedef değişken


# data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# define models: bagging, random forrest, extra trees regressors
models = {
    "Bagging Regressor": BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        n_estimators=100,  # ağaç sayısı
        max_features=0.8,
        max_samples=0.8,
        random_state=42,
    ),
    "Random Forrest Regressor": RandomForestRegressor(
        n_estimators=100, max_depth=15, min_samples_split=5, random_state=42
    ),
    "Extra Trees Regressor": ExtraTreesRegressor(
        n_estimators=100, max_depth=15, min_samples_split=5, random_state=42
    ),
}

# training and testing
results = {}
predictions = {}

for name, model in tqdm(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"Mse": mse, "R2": r2}
    predictions[name] = y_pred

result_df = pd.DataFrame(results).T
print(result_df)

# visualize results

# tahmin vs gerçek değerler
plt.figure()
for i, (name, y_pred) in enumerate(predictions.items(), 1):
    plt.subplot(1, 3, i)
    plt.scatter(y_test, y_pred, alpha=0.5, label=name)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.title(f"{name} Gerçek vs Tahmin")
    plt.xlabel("Gerçek")
    plt.ylabel("Tahmin")
    plt.legend()
plt.tight_layout()
plt.show()

# residuals yani hatalar
plt.figure()
for i, (name, y_pred) in enumerate(predictions.items(), 1):
    residuals = y_test - y_pred
    plt.subplot(1, 3, i)
    plt.scatter(y_pred, residuals, alpha=0.5, label=name)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.title(f"{name} Gerçek vs Tahmin")
    plt.xlabel("Gerçek")
    plt.ylabel("Tahmin")
    plt.legend()
plt.tight_layout()
plt.show()

# feature importance
feature_names = housing.feature_names
plt.figure()

for i, (name, model) in enumerate(models.items(), 1):
    # feature importance elde et
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "estimators_") and hasattr(
        model.estimators_[0], "feature_importances_"
    ):
        # Tüm estimator'lardan feature importance al, eksikse 0 ile doldur
        all_importances = []
        for est in model.estimators_:
            fi = getattr(est, "feature_importances_", None)
            if fi is not None and len(fi) == X.shape[1]:
                all_importances.append(fi)
        if all_importances:
            importance = np.mean(all_importances, axis=0)
        else:
            continue  # hiçbir importance verisi yoksa geç
    else:
        continue  # model feature importance desteklemiyorsa geç

    # sıralama
    sorted_idx = np.argsort(importance)[::-1]

    # grafik
    plt.subplot(1, 3, i)
    plt.bar(range(X.shape[1]), importance[sorted_idx], label=name)
    plt.xticks(range(X.shape[1]), np.array(feature_names)[sorted_idx], rotation=45)
    plt.title(f"{name} Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.legend()

plt.tight_layout()
plt.show()
