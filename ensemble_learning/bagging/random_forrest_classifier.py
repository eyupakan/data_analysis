# import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# load dataset: breast cancer
cancer = load_breast_cancer()
X = cancer.data  # features: tümör boyutu, şekli, alanı vb.
y = cancer.target  # hedef değişken 0: malignant (kötü), 1: benign (iyi)


# data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# create random forest model
rf_model = RandomForestClassifier(
    n_estimators=100,  # ağaç sayısı
    max_depth=10,  # max derinlik
    min_samples_split=5,  # bir düğümü bölmek için min örnek sayısı
    max_leaf_nodes=20,  # max yaprak düğüm sayısı
    random_state=42,
)

# training
rf_model.fit(X_train, y_train)

# testing
y_pred = rf_model.predict(X_test)

# evalution: accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
