# import libraries
from sklearn.ensemble import BaggingClassifier  # torbalama modeli
from sklearn.tree import (
    DecisionTreeClassifier,
)  # torbalama içinde kullanılacak weak learner
from sklearn.datasets import load_iris  # kullanılacak veri seti
from sklearn.model_selection import train_test_split  # train test split fonksiyonu
from sklearn.metrics import accuracy_score  # doğruluk metriği


# load dataset (iris)
iris = load_iris()
X = iris.data  # features
y = iris.target  # target veriable


# data train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# define base model: decision tree
base_model = DecisionTreeClassifier(random_state=42)


# create bagging model
bagging_model = BaggingClassifier(
    estimator=base_model,  # temel model: karar ağacı
    n_estimators=10,  # kullanılacak model sayısı
    max_samples=0.8,  # her modelin kullanacağı örnek oranı
    max_features=0.8,  # her modelin kullanacağı özellik oranı
    bootstrap=True,  # örneklerin tekrar seçilmesine izin ver
    random_state=42,
)


# model training
bagging_model.fit(X_train, y_train)


# model testing
y_pred = bagging_model.predict(X_test)


# evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
