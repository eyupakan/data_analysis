# import libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris  # kullanılacak veri seti
from sklearn.model_selection import train_test_split  # train test split fonksiyonu
from sklearn.metrics import accuracy_score, confusion_matrix  # doğruluk metriği
import seaborn as sns
import matplotlib.pyplot as plt


# load dataset (iris)
iris = load_iris()
X = iris.data  # features
y = iris.target  # target veriable


# data train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# define model
ada_clf = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=42)

# model training
ada_clf.fit(X_train, y_train)


# prediction
y_pred = ada_clf.predict(X_test)


# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# evaluation with cm
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)
plt.title("Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.show()
