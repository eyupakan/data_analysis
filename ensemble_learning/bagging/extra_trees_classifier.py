# import libraries
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# load dataset: breast cancer
cancer = load_breast_cancer()
X = cancer.data  # features: tümör boyutu, şekli, alanı vb.
y = cancer.target  # hedef değişken 0: malignant (kötü), 1: benign (iyi)

# data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# define extra trees
et_model = ExtraTreesClassifier(
    n_estimators=100,  # ağaç sayısı
    max_depth=10,  # max derinlik
    min_samples_split=5,  # bir düğümü bölmek için min örnek sayısı
    random_state=42,
)

# training
et_model.fit(X_train, y_train)

# testing
y_pred = et_model.predict(X_test)

# evalution: accuracy and classification report
print(f"accuracy: {accuracy_score(y_pred, y_test)}")
print(classification_report(y_pred, y_test))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=cancer.target_names,
    yticklabels=cancer.target_names,
)
plt.title("Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")

# visualize feature importance
feature_importance = et_model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
features = cancer.feature_names

plt.figure()
plt.bar(range(X.shape[1]), feature_importance[sorted_idx], align="center")
plt.xticks(range(X.shape[1]), features[sorted_idx], rotation=90)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()
