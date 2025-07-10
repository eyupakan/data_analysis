# import libraries
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt

# load dataset (diabetes)
diabetes = load_diabetes()
X = diabetes.data  # features
y = diabetes.target  # target veriable

# data train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# define model
gb_reg = GradientBoostingRegressor(
    n_estimators=200,  # ağaç sayısı
    learning_rate=0.01,  # öğrenme oranı
    max_depth=5,  # ağaç derinliği
    subsample=0.8,  # verinin %80'i ile eğitim gerçekleştir
    min_samples_split=5,  # node bölünmesi için gerekli min örnek sayısı
    min_samples_leaf=4,  # yapraktaki min örnek sayısı
    validation_fraction=0.1,  # erken durdurma için validation set
    n_iter_no_change=5,  # 5 iterasyonda iyileşme yoksa durdur
    random_state=42,
)

# model training
gb_reg.fit(X_train, y_train)

# testing
y_pred = gb_reg.predict(X_test)

# evaluation: rmse, r2 score
print(f"GBR MSE : {root_mean_squared_error(y_test,y_pred)}")
print(f"GBR R2 SCORE: {r2_score(y_test,y_pred)}")

# residuals
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(0, color="r")
plt.title("GBR - Hata Dağılımı")
plt.xlabel("Tahmin edilen değerler")
plt.ylabel("Hata")
plt.show()
"""
GBR MSE : 57.14887663632432
GBR R2 SCORE: 0.3949966832471167
"""
